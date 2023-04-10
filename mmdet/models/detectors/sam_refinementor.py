import torch
import numpy as np
from mmcv.runner import BaseModule
from ..builder import DETECTORS, build_backbone, build_head
import numpy as np


def uniform_sampler(num_steps, batch_size, device):
    indices_np = np.random.choice(num_steps, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices

def mask2bbox(np_masks, device, H, W, pad_width=0):
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

@DETECTORS.register_module()
class SamRefinementor(BaseModule):
    def __init__(self,
                 image_encoder,
                 prompt_encoder,
                 mask_decoder,
                 diffusion_betas,
                 train_cfg,
                 test_cfg,
                 num_classes=77,
                 init_cfg=None):
        super(SamRefinementor, self).__init__(init_cfg=init_cfg)

        self.image_encoder = build_backbone(image_encoder)
        self.prompt_encoder = build_backbone(prompt_encoder)
        self.mask_decoder =  build_head(mask_decoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_betas)
        self.num_classes = num_classes
    
    def _diffusion_init(self, betas):
        self.eps = 1.e-6
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]

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

        # dim_4_cuda_mask_save(pred_x_start, f'results/x_start_{t}')
        # dim_4_cuda_mask_save(fine_map, f'results/fine_map{t}')
        # dim_4_cuda_mask_save(x_prev, f'results/x_prev{t}')
        return x_prev, cur_fine_probs
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      coarse_masks):
        current_device = img.device
        img_embddings = self.image_encoder(img)
        proposals, x_start, x_last = self._get_refine_input(gt_masks, coarse_masks, current_device)
        t = uniform_sampler(self.num_timesteps, x_start.shape[0], x_start.device)
        x_t = self.q_sample(x_start, x_last, t, current_device)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(proposals, x_t,)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=img_embddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings)

        losses = dict()
        return losses
    
    def _get_refine_input(self, gt_masks, coarse_masks, current_device):
        proposals, x_start, x_last = [], [], []
        for img_gt_masks, img_coarse_masks in zip(gt_masks, coarse_masks):
            assert len(img_gt_masks) == len(img_coarse_masks)
            num_ins = len(img_gt_masks)
            if num_ins > 0:
                H, W = img_gt_masks.height, img_gt_masks.width
                img_proposals = mask2bbox(img_coarse_masks.masks, current_device, H, W)
                proposals.append(img_proposals)
                x_start.append(torch.tensor(img_gt_masks.masks, device=current_device))
                x_last.append(torch.tensor(img_coarse_masks.masks, device=current_device))
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

    def simple_test(self, img, img_metas, coarse_masks, dt_bboxes, rescale=False):
        """Test without augmentation."""

        low_lvl_feat, fpn_feats = self.extract_feat(img)
        if len(coarse_masks[0]):
            mask_pred =  self.roi_head.simple_test(
                low_lvl_feat,
                fpn_feats,
                coarse_masks,
                img_metas)
            dt_bboxes = dt_bboxes.cpu().numpy()
            bboxes = dt_bboxes[0][:, :5]
            labels = dt_bboxes[0][:, 5]
            labels = labels.astype(int)
            bbox_results = self._format_bboxes_results(bboxes, labels)
            mask_results = self._format_mask_results(mask_pred, labels)
        else:
            bbox_results = [np.zeros([0, 4]) for _ in range(self.num_classes)]
            mask_results = [[] for _ in range(self.num_classes)]
        return [(bbox_results, mask_results)]
    
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
        masks = masks >= 0.5
        masks = masks.detach().cpu().numpy()
        cls_masks = [[] for _ in range(self.num_classes)]
        for i in range(len(masks)):
            cls_masks[labels[i]].append(masks[i])
        return cls_masks
