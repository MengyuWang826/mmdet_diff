import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_loss
import numpy as np

from PIL import Image

def uniform_sampler(num_steps, batch_size, device):
    indices_np = np.random.choice(num_steps, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices      

def mask2bboxcenter(input_masks, device, H, W, scale_factor=4):
    if isinstance(input_masks, np.ndarray):
        masks = torch.tensor(input_masks, dtype=torch.bool, device=device)
    elif isinstance(input_masks, torch.Tensor):
        masks = input_masks.squeeze(1)
    else:
        raise TypeError(f'unsupport mask type')
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
    bboxes = bboxes * scale_factor
    # center_x = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
    # center_y = 0.5 * (bboxes[:, 1] + bboxes[:, 3])
    # center_coors = torch.stack((center_x, center_y), dim=1)
    return bboxes

@DETECTORS.register_module()
class SamRefinementor(BaseDetector):
    def __init__(self,
                 image_encoder,
                 prompt_encoder,
                 mask_decoder,
                 diffusion_cfg,
                 train_cfg,
                 test_cfg,
                 mask_loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 iou_loss=dict(type='MSELoss', loss_weight=1.0),
                 num_classes=77,
                 init_cfg=None):
        super(SamRefinementor, self).__init__(init_cfg=init_cfg)

        self.image_encoder = build_backbone(image_encoder)
        self.prompt_encoder = build_backbone(prompt_encoder)
        self.mask_decoder =  build_head(mask_decoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_cfg)
        self.mask_loss = build_loss(mask_loss)
        self.iou_loss = build_loss(iou_loss)
        self.num_classes = num_classes
    
    def _diffusion_init(self, diffusion_cfg):
        self.diff_iter = diffusion_cfg['diff_iter']
        betas = diffusion_cfg['betas']
        self.eps = 1.e-6
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]    
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      coarse_masks):
        current_device = img.device
        img_embddings = self.extract_feat(img)
        areas, x_start, x_last, img_ids = self._get_refine_input(gt_masks, coarse_masks, current_device)
        img_feats = torch.zeros((areas.shape[0], img_embddings.shape[1], img_embddings.shape[2], img_embddings.shape[3]), device=current_device)
        for i in range(img_embddings.shape[0]):
            idx = img_ids == i
            img_feats[idx] = img_embddings[i]
        del img_embddings
        t = uniform_sampler(self.num_timesteps, x_start.shape[0], x_start.device)
        x_t = self.q_sample(x_start, x_last, t, current_device)
        sparse_embeddings, dense_embeddings, time_embeddings = self.prompt_encoder(bboxes, x_t, t)
        low_res_masks, iou_predictions = self.mask_decoder(
            img_feats,
            self.prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            time_embeddings)
        losses = dict()
        mask_targets = x_start.unsqueeze(1)
        iou_targets = self.cal_iou(mask_targets, low_res_masks)
        losses['loss_mask'] = self.mask_loss(low_res_masks, mask_targets)
        losses['loss_iou'] = self.iou_loss(iou_predictions, iou_targets)
        losses['iou'] = iou_targets.mean()
        return losses
    
    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample.unsqueeze(1)
    
    @torch.no_grad()
    def cal_iou(self, target, mask, eps=1e-3):
        target = target >= 0.5
        mask = mask >= 0
        si = (target & mask).sum(-1).sum(-1)
        su = (target | mask).sum(-1).sum(-1)
        return (si / su + eps)

    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        x = self.image_encoder(img)
        return x
    
    def _get_refine_input(self, gt_masks, coarse_masks, current_device):
        areas, x_start, x_last, img_ids = [], [], [], []
        img_id = 0
        for img_gt_masks, img_coarse_masks in zip(gt_masks, coarse_masks):
            assert len(img_gt_masks) == len(img_coarse_masks)
            num_ins = len(img_gt_masks)
            if num_ins > 0:
                areas.append(img_coarse_masks.areas)
                x_start.append(torch.tensor(img_gt_masks.masks, device=current_device))
                x_last.append(torch.tensor(img_coarse_masks.masks, device=current_device))
                img_ids.append(torch.tensor([img_id]*num_ins, device=current_device))
                img_id += 1
        areas = np.concatenate(areas, axis=0)
        x_start = torch.cat(x_start, dim=0)
        x_last = torch.cat(x_last, dim=0)
        img_ids = torch.cat(img_ids, dim=0)
        batch_size = len(x_last)
        # print(batch_size)
        if batch_size > 64:
            chosen_idx = np.random.choice(batch_size, size=32, replace=False)
            areas = areas[chosen_idx]
            x_start = x_start[chosen_idx]
            x_last = x_last[chosen_idx]
            img_ids = img_ids[chosen_idx]
        return areas, x_start, x_last, img_ids

    def simple_test(self, img, img_metas, coarse_masks, dt_bboxes, rescale=False):
        """Test without augmentation."""
        current_device = img.device
        img_embddings = self.extract_feat(img)

        H, W = coarse_masks[0].height, coarse_masks[0].width
        x_last = torch.tensor(coarse_masks[0].masks, device=current_device)
        x = x_last.unsqueeze(1).float()
        if self.diff_iter:
            cur_fine_probs = torch.zeros_like(x)
            indices = list(range(self.num_timesteps))[::-1]
            for i in indices:
                bboxes, _ = mask2bboxcenter(x, current_device, H, W)
                t = torch.tensor([i] * x.shape[0], device=current_device)
                x, cur_fine_probs = self.p_sample(x, cur_fine_probs, t, bboxes, img_embddings)
        else:
            bboxes, _ = mask2bboxcenter(x, current_device, H, W)
            t = torch.tensor([0] * x.shape[0], device=current_device)
            img_feats = torch.repeat_interleave(img_embddings, bboxes.shape[0], dim=0)
            sparse_embeddings, dense_embeddings, time_embeddings = self.prompt_encoder(bboxes, x, t)
            x, iou_predictions = self.mask_decoder(
                img_feats,
                self.prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                time_embeddings)
        ori_shape = img_metas[0]['ori_shape'][:2]
        img_shape = img_metas[0]['img_shape'][:2]
        pad_shape = img_metas[0]['pad_shape'][:2]
        refine_mask = F.interpolate(x, size=pad_shape, mode="bilinear")
        refine_mask = refine_mask[:, :, :img_shape[0], :img_shape[1]]
        refine_mask = F.interpolate(refine_mask, size=ori_shape, mode="bilinear").squeeze(1)
        refine_mask = (refine_mask >= 0).int()

        x_last = F.interpolate(x_last.float().unsqueeze(1), size=pad_shape, mode="bilinear")
        x_last = x_last[:, :, :img_shape[0], :img_shape[1]]
        x_last = F.interpolate(x_last, size=ori_shape, mode="bilinear").squeeze(1)
        x_last = x_last >= 0.5

        refine_save(x_last, refine_mask)

        dt_bboxes = dt_bboxes.cpu().numpy()
        bboxes = dt_bboxes[0][:, :5]
        labels = dt_bboxes[0][:, 5]
        labels = labels.astype(int)
        bbox_results = self._format_bboxes_results(bboxes, labels)
        mask_results = self._format_mask_results(refine_mask, labels)
        return [(bbox_results, mask_results)]

    def p_sample(self, x, cur_fine_probs, t, bboxes, img_embddings):
        img_feats = torch.repeat_interleave(img_embddings, bboxes.shape[0], dim=0)
        sparse_embeddings, dense_embeddings, time_embeddings = self.prompt_encoder(bboxes, x, t)
        low_res_masks, iou_predictions = self.mask_decoder(
            img_feats,
            self.prompt_encoder.get_dense_pe(),
            sparse_embeddings,
            dense_embeddings,
            time_embeddings)
        t = t[0].item()
        pred_x_start = (low_res_masks >= 0).float()
        x_start_fine_probs = 2 * torch.abs(low_res_masks.sigmoid() - 0.5)
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        sample_noise = torch.rand(size=x.shape, device=x.device)
        fine_map = (sample_noise < cur_fine_probs).float()
        x_prev = pred_x_start * fine_map + x * (1 - fine_map)

        # single_mask_save(pred_x_start, t, 'start')
        # single_mask_save(x_prev, t, 'prev')
        if t > 0:
            return x_prev, cur_fine_probs
        else:
            return low_res_masks, iou_predictions

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def _format_bboxes_results(self,bboxes, labels):
        '''
        return [[]*num_class]
        outer: batch
        inter: class, each is a shape[n, 5] np.array(float32)
        '''
        cls_bboxes = []
        for i in range(self.num_classes):
            cur_idx = (labels == i)
            cls_bboxes.append(bboxes[cur_idx])
        return cls_bboxes
    
    def _format_mask_results(self,masks, labels):
        '''
        inter: [[]*n]
        each is a ori_shape binary mask, shape[h, w] bool np.array
        '''
        masks = masks.detach().cpu().numpy()
        cls_masks = [[] for _ in range(self.num_classes)]
        for i in range(len(masks)):
            cls_masks[labels[i]].append(masks[i])
        return cls_masks

def mask_save(gt, coarse, refine):
    refine = refine.squeeze(1)
    gt = gt.cpu().numpy().astype(np.uint8)
    gt = gt * 255
    coarse = coarse.cpu().numpy().astype(np.uint8)
    coarse = coarse * 255
    refine = refine.cpu().numpy().astype(np.uint8)
    refine = refine * 255
    idx = 0
    for gt_mask, coarse_mask, refine_mask in zip(gt, coarse, refine):
        empty = np.zeros_like(gt_mask)
        out1 = np.concatenate((gt_mask, coarse_mask), axis=0)
        out2 = np.concatenate((refine_mask, empty), axis=0)
        out = np.concatenate((out1, out2), axis=1)
        Image.fromarray(out).save(f'results/{idx}.png')
        idx += 1

def refine_save(coarse, refine):
    coarse = coarse.cpu().numpy().astype(np.uint8)
    coarse = coarse * 255
    refine = refine.cpu().numpy().astype(np.uint8)
    refine = refine * 255
    idx = 0
    for coarse_mask, refine_mask in zip(coarse, refine):
        out2 = np.concatenate((coarse_mask, refine_mask), axis=1)
        Image.fromarray(out2).save(f'results/{idx}.png')
        idx += 1

def single_mask_save(masks, idx, file_name):
    masks = masks.squeeze(1)
    masks = masks.cpu().numpy().astype(np.uint8)
    masks = masks * 255
    for i, mask in enumerate(masks):
        Image.fromarray(mask).save(f'results/{file_name}_{idx}_{i}.png')
