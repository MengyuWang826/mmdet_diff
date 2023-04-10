# Copyright (c) OpenMMLab. All rights reserved.
import json
import mmcv
import numpy as np
from .builder import DATASETS
from .pipelines import Compose
from .coco import CocoDataset


@DATASETS.register_module()
class CocoRefineBinary(CocoDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208)]

    def __init__(self,
                 coarse_file=None,
                 max_ins_num=2,
                 **kwargs):
        super(CocoRefineBinary, self).__init__(**kwargs)
        self.max_ins_num = max_ins_num
        if self.test_mode:
            self.coarse_file = coarse_file
            self.coarse_infos = self.load_coarse(self.coarse_file)
            self.coarse_imgs = list(self.coarse_infos)

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
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        num_ins = len(gt_labels)
        if num_ins > self.max_ins_num:
            chosen_idx = np.random.choice(num_ins, size=self.max_ins_num, replace=False)
            gt_bboxes = gt_bboxes[chosen_idx]
            gt_labels = gt_labels[chosen_idx]
            gt_masks = gt_masks_ann
            gt_masks_ann = []
            for idx in chosen_idx:
                gt_masks_ann.append(gt_masks[idx])

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
    
    def load_coarse(self, coarse_file):
        coarse_infos = {}
        coarse_dts = json.load(open(coarse_file))
        for dt in coarse_dts:
            img_id = dt['image_id']
            if img_id not in coarse_infos:
                coarse_infos.update({img_id: []})
            coarse_infos[img_id].append(dt)
        return coarse_infos

    def __len__(self):
        """Total number of samples of data."""
        if not self.test_mode:
            return len(self.data_infos)
        else:
            return len(self.coarse_imgs)

    def get_coarse_info(self, img_id):
        coarse_masks = []
        bboxes = []
        labels = []
        coarse_dts = self.coarse_infos[img_id]
        for dt in coarse_dts:
            label = self.cat2label[dt['category_id']]
            coarse_masks.append(dt['segmentation'])
            bbox = dt['bbox']
            bbox.append(dt['score'])
            bbox.append(label)
            bboxes.append(bbox)
            labels.append(label)
        bboxes = np.array(bboxes)
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
        return dict(masks=coarse_masks, bboxes=bboxes, labels=labels)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        img_id = self.coarse_imgs[idx]
        img_info = self.coco.load_imgs([img_id])[0]
        img_info['filename'] = img_info['file_name']
        coarse_info = self.get_coarse_info(img_id)
        ann_info = self.get_ann_info_test(img_info, img_id)
        results = dict(img_info=img_info, ann_info=ann_info, coarse_info=coarse_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def get_ann_info_test(self, img_info, img_id):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].rsplit('.', 1)[0] + self.seg_suffix

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)
        return ann


