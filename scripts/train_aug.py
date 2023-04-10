import json
from tqdm import tqdm
import multiprocessing as mp
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
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
    block_target[boundary_inds] = 1
    block_target = block_target.squeeze(0).squeeze(0)
    block_target = block_target.numpy().astype(np.uint8)
    return block_target

def mask_save(mask, filename):
    mask = mask * 255
    Image.fromarray(mask).save(filename)

def cal_ioa(mask_dt, mask_gt):
    si = np.sum(mask_dt & mask_gt)
    su = np.sum(mask_dt | mask_gt)
    return su != 0 and si/su > 0

def run_inst(ann):
    fake_coarse = {}
    fake_coarse['category_id'] = ann['category_id']
    fake_coarse['image_id'] = ann['image_id']
    gt_mask = coco.annToMask(ann)
    # mask_save(gt_mask, 'results/mask.png')
    new_mask = generate_block_target(gt_mask)
    # mask_save(new_mask, 'results/new_mask.png')
    new_RLE = mask_utils.encode(new_mask)
    new_RLE['counts'] = str(new_RLE['counts'], encoding='utf-8')
    fake_coarse['segmentation'] = new_RLE
    return fake_coarse


if __name__ == '__main__':
    gt = 'data/annotations/instances_train2017.json'
    all_dts = json.load(open('data/annotations/maskrcnn_coco_train.json'))
    new_dts_json = 'data/annotations/maskrcnn_coco_train_aug.json'
    coco = COCO(gt)
    print(len(all_dts))
    all_anns = list(coco.anns.values())
    
    with mp.Pool(processes=10) as p:
        with tqdm(total=len(all_anns)) as pbar:
            for new_coarse in p.imap_unordered(run_inst, all_anns):
                all_dts.append(new_coarse)
                pbar.update(1)
    # with tqdm(total=len(all_anns)) as pbar:
    #     for ann in all_anns:
    #         new_coarse = run_inst(ann)
    #         pbar.update(1)

    print(len(all_dts))
    with open(new_dts_json, 'w') as f:
        json.dump(all_dts, f)
    # print(f'{pos_total} larger in {len(all_dts)}')

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
    
    
    # mask = cv2.imread(os.path.join(mask_dir, file_name))
    #         h, w = mask.shape[0], mask.shape[1]
    #         if h>50 and w>50:
    #             ann = cv2.imread(os.path.join(ann_dir, file_name))
    #             img = cv2.imread(os.path.join(img_dir, file_name))
    #             new_mask = generate_block_target(cv2.imread(os.path.join(ann_dir, file_name), flags=0), h, w)
    #             output_save(img, ann, mask, new_mask, str(idx)+'.png')
    #             idx += 1

    