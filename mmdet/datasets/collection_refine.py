import json
import mmcv
import os
import os.path as osp
import numpy as np
from lvis import LVIS
from .api_wrappers import COCO
from numpy.core.fromnumeric import shape
from .builder import DATASETS
from .pipelines import Compose
from .lvis import LVISV1Dataset


@DATASETS.register_module()
class CollectionRefine(LVISV1Dataset):
    CLASSES = ('1', '2')

    def __init__(self,
                 coarse_file,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 collection_datasets=None,
                 collection_json=None):
        
        print('loading DT json file...')
        self.coarse_infos = self.load_coarse(coarse_file)
        print('loading GT json file...')
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.seg_suffix = seg_suffix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # processing pipeline
        self.pipeline = Compose(pipeline)

        if collection_datasets is not None:
            self.load_collect(collection_json, collection_datasets)
        if not test_mode:
            self._set_group_flag()
    
    def load_annotations(self, ann_file):
        self.coco = LVIS(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        if self.test_mode:
            self.img_ids = list(self.coarse_infos) 
        else: 
            self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            info['filename'] = 'coco/' + info['filename']
            data_infos.append(info)
        return data_infos

    def load_collect(self, collection_json, collection_datasets):
        collect = json.load(open(collection_json))
        for dataset_name in collection_datasets:
            self.data_infos.extend(collect[dataset_name])

    def load_coarse(self, coarse_file):
        coarse_dts = json.load(open(coarse_file))
        if isinstance(coarse_dts, list):
            coarse_infos = {}
            for dt in coarse_dts:
                img_id = dt['image_id']
                if img_id not in coarse_infos:
                    coarse_infos.update({img_id: []})
                coarse_infos[img_id].append(dt)
        elif isinstance(coarse_dts, dict):
            coarse_infos = coarse_dts
        else:
            raise TypeError('unsupported coarse_dt type')
        return coarse_infos
    
    def get_ann_info(self, img_info):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if 'id' in img_info:
            img_id = img_info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            ann_info = self._parse_ann_info(img_info, ann_info)
        else:
            ann_info = img_info
        return ann_info
    
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ann_ids = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 1024 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            ann_id = str(ann['id'])
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_ann_ids.append(ann_id)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            return None
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        chosen_idx = np.random.choice(len(gt_ann_ids))
        ann = dict(
            ann_id=gt_ann_ids[chosen_idx],
            bboxes=gt_bboxes[chosen_idx],
            labels=gt_labels[chosen_idx],
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann[chosen_idx],
            seg_map=seg_map)

        return ann

    def get_coarse_info(self, ann_info):
        coarse_masks = None
        if 'ann_id' in ann_info:
            maskrcnn_flag = np.random.rand() < 0.3
            if maskrcnn_flag:
                ann_id = ann_info['ann_id']
                if ann_id in self.coarse_infos:
                    dt = self.coarse_infos[ann_id]
                    chosen_idx = np.random.choice(len(dt), size=1).item()
                    coarse_masks = dt[chosen_idx]['segmentation']
        return dict(masks=coarse_masks)
    
    def get_coarse_info_test(self, img_info):
        img_id = img_info['id']
        coarse_masks = []
        bboxes = []
        lables = []
        coarse_dts = self.coarse_infos[img_id]
        for dt in coarse_dts:
            label = self.cat2label[dt['category_id']]
            coarse_masks.append(dt['segmentation'])
            bbox = dt['bbox']
            bbox.append(dt['score'])
            bbox.append(label)
            bboxes.append(bbox)
            lables.append(label)
        bboxes = np.array(bboxes)
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        return dict(masks=coarse_masks, bboxes=bboxes, lables=lables)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys 
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(img_info)
        if ann_info is None:
            return None
        coarse_info = self.get_coarse_info(ann_info)
        results = dict(img_info=img_info, coarse_info=coarse_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        img_info = self.data_infos[idx]

        # img_info = self.coco.load_imgs([252219])[0]
        # img_info['filename'] = img_info['coco_url'].replace(
        #     'http://images.cocodataset.org/', '')
        
        coarse_info = self.get_coarse_info_test(img_info)
        results = dict(img_info=img_info, coarse_info=coarse_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

