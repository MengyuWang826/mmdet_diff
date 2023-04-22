import torch
from torch.cuda import current_device
import torch.nn.functional as F
import numpy as np
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead
from mmdet.core import mask, mask_target_single
import mmcv
from PIL import Image


def mask_save(mask, filename):
    mask = (mask.astype(np.uint8))*255
    Image.fromarray(mask).save(filename)

def img_save(img, filename):
    # gt_mask = gt_mask.cpu().numpy()
    img = img.astype(np.uint8)
    Image.fromarray(img).save(filename)

def uniform_sampler(num_steps, batch_size, device):
    indices_np = np.random.choice(num_steps, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices

@HEADS.register_module()
class DiffusionMergeRoIHead(StandardRoIHead):
    def __init__(self, pad_width, diffusion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diffusion_init(**diffusion)
        self.pad_width = pad_width
        self.max_batch = 8
    
    def _diffusion_init(self,
                        num_pixel_vals, 
                        betas):

        self.eps = 1.e-6
        self.num_pixel_vals = num_pixel_vals
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]
        # self.betas_cumprod = np.cumprod(self.betas, axis=0)

    def forward_train(self,
                      low_lvl_feat,
                      fpn_feats,
                      img_metas, 
                      gt_masks,
                      coarse_masks):
        # mask head forward and loss
        losses = dict()
        current_device = low_lvl_feat.device
        proposals, x_start, x_last = self._get_proposals_and_mask(gt_masks, coarse_masks, current_device)

        # gt_save(gt_masks[0].masks)
        # coarse_save(coarse_masks[0].masks)
        # gt_roi_save(x_start)
        # coarse_roi_save(x_last)

        # for t in list(range(self.num_timesteps))[::-1]:
        #     t_step = torch.tensor([t]*len(x_start), dtype=torch.long, device=current_device)
        #     x_t = self.q_sample(x_start, x_last, t_step, current_device)
        #     sample_save(x_t, t)

        t = uniform_sampler(self.num_timesteps, x_start.shape[0], x_start.device)
        x_t = self.q_sample(x_start, x_last, t, current_device)
        mask_feats = self.mask_roi_extractor(low_lvl_feat, fpn_feats, proposals)
        x_with_c = torch.cat((x_t, mask_feats), dim=1)
       
        mask_pred = self.mask_head(x_with_c, t)
        mask_targets = x_start.unsqueeze(1)
        loss_mask = self.mask_head.loss_mask(mask_pred, mask_targets)
        losses.update({'loss_mask': loss_mask})
        return losses
    
    def simple_test(self, 
                    low_lvl_feat,
                    fpn_feats,
                    coarse_masks,
                    img_metas):
        current_device = low_lvl_feat.device
        proposals, x_last = self._get_proposals_and_mask_test(coarse_masks, current_device)
        mask_feats = self.mask_roi_extractor(low_lvl_feat, fpn_feats, proposals)
        indices = list(range(self.num_timesteps))[::-1]
        num_ins = x_last.shape[0]
        res = []
        if num_ins > 16:
            for idx in range(0, num_ins, 16):
                x = x_last[idx:idx+16]
                x = x.unsqueeze(1)
                feats = mask_feats[idx:idx+16]
                cur_fine_probs = torch.zeros_like(x)
                for i in indices:
                    t = torch.tensor([i] * x.shape[0], device=current_device)
                    x, cur_fine_probs = self.p_sample(x,
                                                       feats,
                                                       cur_fine_probs,
                                                       t)
                res.append(x)
        else:
            x = x_last.unsqueeze(1)
            feats = mask_feats
            cur_fine_probs = torch.zeros_like(x)
            for i in indices:
                t = torch.tensor([i] * x.shape[0], device=current_device)
                x, cur_fine_probs = self.p_sample(x,
                                                    feats,
                                                    cur_fine_probs,
                                                    t)
            res = [x]
        res = torch.cat(res, dim=0)
        res = _do_paste_mask(res, proposals, img_metas)
        return res
    
    def _get_proposals_and_mask(self, gt_masks, coarse_masks, current_device):
        proposals, x_start, x_last = [], [], []
        img_id = 0
        for img_gt_masks, img_coarse_masks in zip(gt_masks, coarse_masks):
            assert len(img_gt_masks) == len(img_coarse_masks)
            num_ins = len(img_gt_masks)
            if num_ins > 0:
                H, W = img_gt_masks.height, img_gt_masks.width
                img_proposals = mask2bbox(img_gt_masks.masks, current_device, H, W, pad_width=self.pad_width)
                with torch.no_grad():
                    roi_gt_masks = mask_target_single(img_proposals, torch.tensor(range(num_ins), dtype=torch.long), img_gt_masks, self.train_cfg)
                    roi_coarse_masks = mask_target_single(img_proposals, torch.tensor(range(num_ins), dtype=torch.long), img_coarse_masks, self.train_cfg)
                img_ids = torch.tensor([img_id]*num_ins, device=current_device, dtype=img_proposals.dtype).reshape(-1, 1)
                proposals.append(torch.cat((img_ids, img_proposals), dim=-1))
                x_start.append(roi_gt_masks)
                x_last.append(roi_coarse_masks)
                img_id += 1
        proposals = torch.cat(proposals, dim=0)
        x_start = torch.cat(x_start, dim=0)
        x_last = torch.cat(x_last, dim=0)
        batch_size = len(x_last)
        # print(batch_size)
        if batch_size > self.max_batch:
            proposals = torch.cat((proposals[:self.max_batch//2], proposals[-self.max_batch//2:]), dim=0)
            x_start = torch.cat((x_start[:self.max_batch//2], x_start[-self.max_batch//2:]), dim=0)
            x_last = torch.cat((x_last[:self.max_batch//2], x_last[-self.max_batch//2:]), dim=0)
        return proposals, x_start, x_last
    
    def _get_proposals_and_mask_test(self, coarse_masks, current_device):
        proposals, x_last = [], []
        img_id = 0
        for img_coarse_masks in coarse_masks:
            num_ins = len(img_coarse_masks)
            if num_ins > 0:
                H, W = img_coarse_masks.height, img_coarse_masks.width
                img_proposals = mask2bbox(img_coarse_masks.masks, current_device, H, W, pad_width=self.pad_width)
                with torch.no_grad():
                    roi_coarse_masks = mask_target_single(img_proposals, torch.tensor(range(num_ins), dtype=torch.long), img_coarse_masks, self.test_cfg)
                img_ids = torch.tensor([img_id]*num_ins, device=current_device, dtype=img_proposals.dtype).reshape(-1, 1)
                proposals.append(torch.cat((img_ids, img_proposals), dim=-1))
                x_last.append(roi_coarse_masks)
                img_id += 1
        proposals = torch.cat(proposals, dim=0)
        x_last = torch.cat(x_last, dim=0)
        return proposals, x_last
    
    def _get_proposals_and_mask_from_bbox(self, masks, current_device):
        proposals, mask_targets = [], []
        img_id = 0
        for img_meta, img_bboxes, img_masks in zip(img_metas, bboxes, masks):
            img_bboxes = torch.tensor(img_bboxes[:, :4], device=current_device, dtype=torch.float32)
            num_ins = img_bboxes.shape[0]
            H, W = img_meta['img_shape'][:-1]
            img_bboxes[:, 0] = (img_bboxes[:, 0] - 10).clamp(min=0)
            img_bboxes[:, 1] = (img_bboxes[:, 1] - 10).clamp(min=0)
            img_bboxes[:, 2] = (img_bboxes[:, 2] + 10).clamp(max=W)
            img_bboxes[:, 3] = (img_bboxes[:, 3] + 10).clamp(max=H)
            with torch.no_grad():
                roi_masks = mask_target_single(img_bboxes, torch.tensor(range(num_ins), dtype=torch.long), img_masks, self.test_cfg)
            img_ids = torch.tensor([img_id]*num_ins, device=current_device, dtype=img_bboxes.dtype).reshape(-1, 1)
            proposals.append(torch.cat((img_ids, img_bboxes), dim=-1))
            mask_targets.append(roi_masks)
            img_id += 1
        proposals = torch.cat(proposals, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)
        return proposals, mask_targets

    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample.unsqueeze(1)
    
    def p_sample(self, x, feats, cur_fine_probs, t):
        x_with_c = torch.cat((x, feats), dim=1)
        x_start_logits = self.mask_head(x_with_c, t)
        pred_x_start = (x_start_logits >= 0).float()
        x_start_fine_probs = 2 * torch.abs(x_start_logits.sigmoid() - 0.5)

        t = t[0].item()
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f

        sample_noise = torch.rand(size=x.shape, device=x.device)
        fine_map = (sample_noise < cur_fine_probs).float()
        x_prev = pred_x_start * fine_map + x * (1 - fine_map)

        dim_4_cuda_mask_save(pred_x_start, f'results/x_start_{t}')
        dim_4_cuda_mask_save(fine_map, f'results/fine_map{t}')
        # dim_4_cuda_mask_save(x_prev, f'results/x_prev{t}')
        return x_prev, cur_fine_probs


def mask2bbox(np_masks, device, H, W, pad_width=20):
    """Obtain tight bounding boxes of binary masks.

    Args:
        masks (Tensor): Binary mask of shape (n, h, w).

    Returns:
        Tensor: Bboxe with shape (n, 4) of \
            positive region in binary mask.
    """
    masks = torch.tensor(np_masks, dtype=torch.bool, device=device)
    N = masks.shape[0]
    bboxes = torch.zeros((N, 4), device=device, dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(N):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i, :] = bboxes.new_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1])
    bboxes[:, 0] = (bboxes[:, 0] - pad_width).clamp(min=0)
    bboxes[:, 1] = (bboxes[:, 1] - pad_width).clamp(min=0)
    bboxes[:, 2] = (bboxes[:, 2] + pad_width).clamp(max=W)
    bboxes[:, 3] = (bboxes[:, 3] + pad_width).clamp(max=H)
    return bboxes

def _do_paste_mask(masks, proposals, img_metas):
    device = masks.device
    scale_factor = img_metas[0]['scale_factor']
    scale_factor = torch.from_numpy(scale_factor).to(proposals)
    scale_factor = scale_factor.reshape(1, 4)
    img_h, img_w = img_metas[0]['ori_shape'][:2]
    boxes = proposals[:, 1:]
    boxes = boxes / scale_factor
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)
    return img_masks[:, 0]


def dim_4_cuda_mask_save(masks, file_name):
    masks = masks.squeeze(1)
    for idx, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(file_name + f'_{idx}.png')
