import json
from tqdm import tqdm
import multiprocessing as mp
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.datasets.api_wrappers import COCO, COCOeval
import pycocotools.mask as mask_utils


def generate_block_target(mask):
    h, w = mask.shape
    boundary_width = max(2, (h+w)//20)
    # boundary_width = 3
    mask_target = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    mask_target = (mask_target>0).float()

    # boundary region
    kernel_size = 2 * boundary_width + 1
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)
    laplacian_kernel[0, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1

    pad_target = F.pad(mask_target, (boundary_width, boundary_width, boundary_width, boundary_width), "constant", 0)

    # pos_boundary
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets

    # neg_boundary
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets

    # generate block target
    block_target = torch.zeros_like(mask_target).float().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets + mask_target) > 0
    block_target[boundary_inds] = 255
    block_target = block_target.squeeze(0).squeeze(0)
    block_target = block_target.numpy().astype(np.uint8)
    return block_target

def cal_ioa(mask_dt, mask_gt):
    si = np.sum(mask_dt & mask_gt)
    s0 = np.sum(mask_gt)
    s1 = np.sum(mask_dt)
    return (s1 > s0) and (s0 != 0) and si/s0 >= 0.98

def run_inst(img_id):
    pos_num = 0
    dts = img_dts[img_id]
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    ann_info = coco.load_anns(ann_ids)
    gts = {}
    for ann in ann_info:
        cls_id = ann['category_id']
        gt_mask = coco.annToMask(ann)
        if cls_id not in gts:
            gts.update({cls_id: []})
        gts[cls_id].append(gt_mask)
    for cls_id_dt in dts:
        if cls_id_dt not in gts:
            continue
        else:
            dt_masks = mask_utils.decode(dts[cls_id_dt])
            dt_masks = list(np.transpose(dt_masks, (2, 0, 1)))
            for dt_mask in dt_masks:
                for gt_mask in gts[cls_id_dt]:
                    if cal_ioa(dt_mask, gt_mask):
                        pos_num += 1
    return pos_num


if __name__ == '__main__':
    gt = 'data/annotations/instances_train2017.json'
    all_dts = json.load(open('data/annotations/maskrcnn_coco_train.json'))
    new_dts_json = 'data/annotations/maskrcnn_coco_train_aug.json'
    coco = COCO(gt)
    all_img_ids = coco.get_img_ids()
    
    img_dts = {}
    for dt in all_dts:
        img_id = dt['image_id']
        cls_id = dt['category_id']
        if img_id not in img_dts:
            img_dts.update({img_id: {cls_id: []}})
        elif cls_id not in img_dts[img_id]:
            img_dts[img_id].update({cls_id: []})
        img_dts[img_id][cls_id].append(dt['segmentation'])
    print('format dts complete!')

    pos_total = 0
    with mp.Pool(processes=20) as p:
        with tqdm(total=len(img_dts)) as pbar:
            for pos in p.imap_unordered(run_inst, img_dts):
                pos_total += pos
                pbar.update(1)
    
    print(f'{pos_total} larger in {len(all_dts)}')

    # all_dt_imgs = set()
    # for dt in dts:
    #     all_dt_imgs.add(dt['image_id'])
    
    # new_imgs = []
    # for img in gt_imgs:
    #     if len(new_imgs) >= 64:
    #         break
    #     if img['id'] in all_dt_imgs:
    #         new_imgs.append(img)
    # gt['images'] = new_imgs
    # with open('data/refine_annotations/lvis_val_easy.json', 'w') as f:
    #     json.dump(gt, f)
    
    # mask = cv2.imread(os.path.join(mask_dir, file_name))
    #         h, w = mask.shape[0], mask.shape[1]
    #         if h>50 and w>50:
    #             ann = cv2.imread(os.path.join(ann_dir, file_name))
    #             img = cv2.imread(os.path.join(img_dir, file_name))
    #             new_mask = generate_block_target(cv2.imread(os.path.join(ann_dir, file_name), flags=0), h, w)
    #             output_save(img, ann, mask, new_mask, str(idx)+'.png')
    #             idx += 1

    