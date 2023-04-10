import torch
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
    indices_np += 1
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices

@HEADS.register_module()
class DiffusionDownRoIHead(StandardRoIHead):
    def __init__(self, diffusion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diffusion_init(**diffusion)
    
    def _diffusion_init(self, timesteps, pad_width):
        self.num_timesteps = timesteps
        self.pad_width = pad_width

    # def forward_train(self,
    #                   low_lvl_feat,
    #                   fpn_feats,
    #                   img_metas, 
    #                   gt_bboxes,
    #                   gt_masks):
    #     # mask head forward and loss
    #     losses = dict()
    #     proposals, mask_targets = self._get_proposals_and_mask(img_metas, gt_bboxes, gt_masks)
    #     t = uniform_sampler(self.num_timesteps, mask_targets.shape[0], mask_targets.device)
    #     x_t = self.q_sample(mask_targets, t)
    #     mask_feats = self.mask_roi_extractor(low_lvl_feat, fpn_feats, proposals)
    #     x_with_c = torch.cat((x_t, mask_feats), dim=1)
       
    #     mask_pred = self.mask_head(x_with_c, t)
    #     mask_targets = mask_targets.unsqueeze(1)
    #     loss_mask = self.mask_head.loss_mask(mask_pred, mask_targets)
    #     losses.update({'loss_mask': loss_mask})
    #     return losses

    def forward_train(self,
                      low_lvl_feat,
                      fpn_feats,
                      img_metas, 
                      gt_bboxes,
                      gt_masks):
        # mask head forward and loss
        losses = dict()
        proposals, mask_targets = self._get_proposals_and_mask(img_metas, gt_bboxes, gt_masks)
        gt_save(mask_targets)
        batch_size = mask_targets.shape[0]
        current_device = mask_targets.device
        t = torch.tensor([self.num_timesteps] * batch_size, device=current_device)
        x = self.q_sample(mask_targets, t)
        coarse_save(x)
        mask_feats = self.mask_roi_extractor(low_lvl_feat, fpn_feats, proposals)
        indices = list(range(1, self.num_timesteps+1))[::-1]
        for i in indices:
                t = torch.tensor([i] * batch_size, device=current_device)
                x = self.p_sample(x,
                                  mask_feats,
                                  t)
       
        mask_pred = self.mask_head(x_with_c, t)
        mask_targets = mask_targets.unsqueeze(1)
        loss_mask = self.mask_head.loss_mask(mask_pred, mask_targets)
        losses.update({'loss_mask': loss_mask})
        return losses
    
    def simple_test(self, 
                    low_lvl_feat,
                    fpn_feats,
                    coarse_masks,
                    dt_bboxes,
                    img_metas):
        current_device = low_lvl_feat.device
        proposals, mask_targets = self._get_proposals_and_mask_test(img_metas, coarse_masks, dt_bboxes, current_device)
        mask_feats = self.mask_roi_extractor(low_lvl_feat, fpn_feats, proposals)
        indices = list(range(1, self.num_timesteps+1))[::-1]
        num_ins = mask_targets.shape[0]
        x_last = mask_targets
        if num_ins > 50:
            nn_inputs = []
            nn_inputs.append([x_last[:50], mask_feats[:50], 50])
            nn_inputs.append([x_last[50:], mask_feats[50:], num_ins-50])
        else:
            nn_inputs = [[x_last, mask_feats, num_ins]]

        res = []
        for nn_input in nn_inputs:
            x, feats, batch_size = nn_input
            gt_save(x)
            t_max = torch.tensor([self.num_timesteps] * batch_size, device=current_device)
            x = self.q_sample(x, t_max)
            coarse_save(x)
            for i in indices:
                t = torch.tensor([i] * batch_size, device=current_device)
                x = self.p_sample(x,
                                  feats,
                                  t)
            res.append(x)
        res = torch.cat(res, dim=0)
        res = _do_paste_mask(res, proposals, img_metas)
        return res
    
    def _get_proposals_and_mask(self, img_metas, bboxes, masks):
        proposals, mask_targets = [], []
        img_id = 0
        for img_meta, img_bboxes, img_masks in zip(img_metas, bboxes, masks):
            num_ins = img_bboxes.shape[0]
            if num_ins > 0:
                H, W = img_meta['img_shape'][:-1]
                img_bboxes[:, 0] = (img_bboxes[:, 0] - 10).clamp(min=0)
                img_bboxes[:, 1] = (img_bboxes[:, 1] - 10).clamp(min=0)
                img_bboxes[:, 2] = (img_bboxes[:, 2] + 10).clamp(max=W)
                img_bboxes[:, 3] = (img_bboxes[:, 3] + 10).clamp(max=H)
                with torch.no_grad():
                    roi_masks = mask_target_single(img_bboxes, torch.tensor(range(num_ins), dtype=torch.long), img_masks, self.train_cfg)
                img_ids = torch.tensor([img_id]*num_ins, device=img_bboxes.device, dtype=img_bboxes.dtype).reshape(-1, 1)
                proposals.append(torch.cat((img_ids, img_bboxes), dim=-1))
                mask_targets.append(roi_masks)
                img_id += 1
        proposals = torch.cat(proposals, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)
        return proposals, mask_targets
    
    def _get_proposals_and_mask_test(self, img_metas, masks, bboxes, current_device):
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

    def q_sample(self, mask_targets, t, test_mode=False):
        if test_mode:
            stride = 2 ** t[0].item()
            # mask_save(x_start.squeeze(0).squeeze(0).cpu().numpy(), f'results/{idx}_0.png')
            x_t = F.interpolate(
                F.interpolate(mask_targets, scale_factor=1/stride, mode='bilinear'),
                scale_factor=stride, 
                mode='bilinear')
            x_t = (x_t >= 0.5).float()
            # mask_save(x_t_i.squeeze(0).squeeze(0).cpu().numpy(), f'results/{idx}_t.png')
        else:
            strides = 2 ** t
            x_t = []
            for idx, stride in enumerate(strides):
                stride = stride.item()
                x_start = mask_targets[idx].clone().detach().unsqueeze(0).unsqueeze(0)
                # mask_save(x_start.squeeze(0).squeeze(0).cpu().numpy(), f'results/{idx}_0.png')
                x_t_i = F.interpolate(
                    F.interpolate(x_start, scale_factor=1/stride, mode='bilinear'),
                    scale_factor=stride, 
                    mode='bilinear')
                x_t_i = (x_t_i >= 0.5).float()
                # mask_save(x_t_i.squeeze(0).squeeze(0).cpu().numpy(), f'results/{idx}_t.png')
                x_t.append(x_t_i)
            x_t = torch.cat(x_t, dim=0)
        return x_t
    
    def p_sample(self, x, feats, t):
        x_with_c = torch.cat((x, feats), dim=1)
        pred_x_start = self.mask_head(x_with_c, t)
        pred_x_start = (pred_x_start >= 0).float()
        x_start_save(pred_x_start, t[0].item())
        degraded_x_cur = self.q_sample(pred_x_start, t, test_mode=True)
        degraded_x_prev = self.q_sample(pred_x_start, t-1, test_mode=True)
        sample = x - degraded_x_cur + degraded_x_prev
        x_prev_save(sample, t[0].item())
        return sample

def _do_paste_mask(masks, proposals, img_metas):
    device = masks.device
    img_h, img_w = img_metas[0]['ori_shape'][:2]
    boxes = proposals[:, 1:]
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

def x_start_save(masks, t):
    masks = masks.squeeze(1)
    for idx, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(f'results/x_start_{t}_{idx}.png')

def x_prev_save(masks, t):
    masks = masks.squeeze(1)
    for idx, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(f'results/x_prev_{t}_{idx}.png')

def coarse_save(masks):
    masks = masks.squeeze(1)
    for idx, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(f'results/coarse_{idx}.png')

def gt_save(masks):
    for idx, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(f'results/gt_{idx}.png')