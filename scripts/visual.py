import numpy as np
import cv2
import json
from PIL import Image
import pycocotools.mask as maskUtils
from mmdet.datasets.api_wrappers import COCO, COCOeval
from lvis import LVIS

def output_save(img, ann, mask_c, mask_f, filename, color_map=(220, 20, 60)):
    color_map = np.array(color_map).reshape(1, 1, -1)
    mask_f = np.expand_dims(mask_f, axis=2)
    mask_f = (mask_f * color_map).astype(np.uint8)
    mask_f = cv2.addWeighted(img, 0.5, mask_f, 0.5, 0)
    # mask_f = np.repeat(mask_f, 3, axis=2)
    # mask_f = mask_f * 255
    mask_c = np.expand_dims(mask_c, axis=2)
    mask_c = (mask_c * color_map).astype(np.uint8)
    mask_c = cv2.addWeighted(img, 0.5, mask_c, 0.5, 0)
    # mask_c = np.repeat(mask_c, 3, axis=2)
    # mask_c = mask_c * 255
    ann = np.expand_dims(ann, axis=2)
    ann = (ann * color_map).astype(np.uint8)
    ann = cv2.addWeighted(img, 0.5, ann, 0.5, 0)
    # ann = np.repeat(ann, 3, axis=2)
    # ann = ann * 255
    output1 = np.concatenate((img, ann), axis=0)
    output2 = np.concatenate((mask_c, mask_f), axis=0)
    output = np.concatenate((output1, output2), axis=1)
    # if not os.path.exists(root+img_id):
    #     os.mkdir(root+img_id)
    Image.fromarray(output).save(filename)

def _poly2mask(mask_ann, img_h=None, img_w=None):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

def cal_iou(mask1, mask2):
    si = np.sum(mask1 & mask2)
    su = np.sum(mask1 | mask2)
    if su==0:
        return 0
    else:
        return si/su

if __name__ == '__main__':
    data_root = 'data/coco/'
    gt_json = 'data/refine_annotations/lvis_v1_val_cocofied.json'
    refine_json = 'results.segm.json'
    coarse_json = 'coarse_part.json'
    all_imgs = [397133, 37777, 252219, 174482]
    # [427, 640] [230, 352] [428, 640] [388, 640]
    gt = LVIS(gt_json)
    refine = json.load(open(refine_json))
    coarse = json.load(open(coarse_json))

    refine_dts, coarse_dts = {}, {}
    for dt in refine:
        img_id = dt['image_id']
        cat_id = dt['category_id']
        if img_id not in refine_dts:
            refine_dts.update({img_id: {cat_id: []}})
        elif cat_id not in refine_dts[img_id]:
            refine_dts[img_id].update({cat_id: []})
        refine_dts[img_id][cat_id].append(_poly2mask(dt['segmentation']))

    for dt in coarse:
        img_id = dt['image_id']
        cat_id = dt['category_id']
        if img_id not in coarse_dts:
            coarse_dts.update({img_id: {cat_id: []}})
        elif cat_id not in coarse_dts[img_id]:
            coarse_dts[img_id].update({cat_id: []})
        coarse_dts[img_id][cat_id].append(_poly2mask(dt['segmentation']))

    for img_id in all_imgs:
        img_info = gt.load_imgs([img_id])[0]
        filename = data_root + img_info['coco_url'].replace('http://images.cocodataset.org/', '')
        image = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        h, w = img_info['height'], img_info['width']
        ann_ids = gt.get_ann_ids(img_ids=[img_id])
        ann_info = gt.load_anns(ann_ids)
        for ins_id, ann in enumerate(ann_info):
            cat_id = ann['category_id']
            if cat_id in refine_dts[img_id] and cat_id in coarse_dts[img_id]:
                gt_mask = _poly2mask(ann['segmentation'], img_h=h, img_w=w)
                coarse_masks = coarse_dts[img_id][cat_id]
                refine_masks = refine_dts[img_id][cat_id]
                ious = [cal_iou(gt_mask, _) for _ in coarse_masks]
                max_iou = max(ious)
                idx = ious.index(max_iou)
                iou_f = cal_iou(gt_mask, refine_masks[idx])
                iou_c = round(max_iou, 3)
                iou_f = round(iou_f, 3)
                filename=f'results_100/{img_id}_{ins_id}_c{iou_c}_f{iou_f}.png'
                output_save(image, gt_mask, coarse_masks[idx], refine_masks[idx], filename=filename)

    a = 1
