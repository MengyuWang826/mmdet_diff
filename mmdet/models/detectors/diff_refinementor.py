import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseDetector
from ..builder import DETECTORS, build_head, build_loss
import numpy as np

from PIL import Image

def uniform_sampler(num_steps, batch_size, device):
    indices_np = np.random.choice(num_steps, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices      

def mask2bboxcenter(input_masks, device, scale_factor=4):
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
class DiffRefinementor(BaseDetector):
    def __init__(self,
                 denoise_model,
                 diffusion_cfg,
                 train_cfg,
                 test_cfg,
                 mask_loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
                 num_classes=77,
                 init_cfg=None):
        super(DiffRefinementor, self).__init__(init_cfg=init_cfg)

        self.denoise_model = build_head(denoise_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_cfg)
        self.mask_loss = build_loss(mask_loss)
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
                      object_img,
                      object_gt_masks,
                      object_coarse_masks,
                      patch_img,
                      patch_gt_masks,
                      patch_coarse_masks,
                      img_metas):
        current_device = object_img.device
        img = torch.cat((object_img, patch_img), dim=0)
        x_start = torch.cat((self._bitmapmasks_to_tensor(object_gt_masks, current_device),
                             self._bitmapmasks_to_tensor(patch_gt_masks, current_device)), dim=0)
        x_last = torch.cat((self._bitmapmasks_to_tensor(object_coarse_masks, current_device),
                            self._bitmapmasks_to_tensor(patch_coarse_masks, current_device)), dim=0)
        x_t = self.q_sample(x_start, x_last, t)
        losses = dict()
        # mask_targets = x_start.unsqueeze(1)
        # iou_targets = self.cal_iou(mask_targets, low_res_masks)
        # losses['loss_mask'] = self.mask_loss(low_res_masks, mask_targets)
        # losses['loss_iou'] = self.iou_loss(iou_predictions, iou_targets)
        # losses['iou'] = iou_targets.mean()
        return losses
    
    def _bitmapmasks_to_tensor(bitmapmasks, current_device):
        tensor_masks = []
        for bitmapmask in bitmapmasks:
            tensor_masks.append(bitmapmask.masks)
        tensor_masks = np.stack(tensor_masks)
        tensor_masks = torch.tensor(tensor_masks, device=current_device)
        return tensor_masks
    
    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        # quad_mask_save(x_start, x_last, new_x_start, sample)
        return sample
    
    @torch.no_grad()
    def cal_iou(self, target, mask, eps=1e-3):
        target = target >= 0.5
        mask = mask >= 0
        si = (target & mask).sum(-1).sum(-1)
        su = (target | mask).sum(-1).sum(-1)
        return (si / su + eps)

    def simple_test(self, img, img_metas, coarse_masks, dt_bboxes, rescale=False):
        """Test without augmentation."""
        if len(coarse_masks[0]) == 0:
            bbox_results = [np.zeros([0, 4]) for _ in range(self.num_classes)]
            mask_results = [[] for _ in range(self.num_classes)]
            return [(bbox_results, mask_results)]
        current_device = img.device
        img_embddings = self.extract_feat(img)

        H, W = coarse_masks[0].height, coarse_masks[0].width
        x_last = torch.tensor(coarse_masks[0].masks, device=current_device)
        x = x_last.unsqueeze(1).float()
        if self.diff_iter:
            cur_fine_probs = torch.zeros_like(x)
            indices = list(range(self.num_timesteps))[::-1]
            for i in indices:
                bboxes = mask2bboxcenter(x, current_device)
                t = torch.tensor([i] * x.shape[0], device=current_device)
                x, cur_fine_probs = self.p_sample(x, cur_fine_probs, t, bboxes, img_embddings)
        else:
            bboxes = mask2bboxcenter(x, current_device)
            t = torch.tensor([0] * x.shape[0], device=current_device)
            img_feats = torch.repeat_interleave(img_embddings, bboxes.shape[0], dim=0)
            sparse_embeddings, dense_embeddings, time_embeddings = self.prompt_encoder(bboxes, None, t)
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
        refine_mask = F.interpolate(refine_mask, size=ori_shape, mode="bilinear")
        refine_mask = (refine_mask >= 0.5).int()

        x_last = F.interpolate(x_last.float().unsqueeze(1), size=pad_shape, mode="bilinear")
        x_last = x_last[:, :, :img_shape[0], :img_shape[1]]
        x_last = F.interpolate(x_last, size=ori_shape, mode="bilinear").squeeze(1)
        x_last = x_last >= 0.5

        multi_mask_save(x_last, refine_mask)
        refine_mask = refine_mask[:, 0]

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
            return low_res_masks.sigmoid(), iou_predictions
    
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

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError


@torch.no_grad()
def generate_block_target(mask, area):
    expand_flag = np.random.rand(1)[0] <= 0.5
    # width_frac = np.random.rand(1)
    # boundary_width = max((int(np.sqrt(area) / 5 * width_frac[0]), 2))
    boundary_width = max((int(np.sqrt(area) / 5), 2))
    mask_target = mask.float().unsqueeze(0).unsqueeze(0)

    # boundary region
    kernel_size = 2 * boundary_width + 1
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)
    laplacian_kernel[:, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1

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
    if expand_flag:
        boundary_inds = (pos_boundary_targets + neg_boundary_targets + mask_target) > 0
    else:
        boundary_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target = block_target.squeeze(0).squeeze(0)
    return block_target

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

def multi_mask_save(coarse, refine):
    coarse = coarse.cpu().numpy().astype(np.uint8)
    coarse = coarse * 255
    refine = refine.cpu().numpy().astype(np.uint8)
    refine = refine * 255
    idx = 0
    for coarse_mask, refine_masks in zip(coarse, refine):
        empty = np.zeros_like(coarse_mask)
        out1 = np.concatenate((coarse_mask, empty), axis=0)
        out2 = np.concatenate((refine_masks[0], refine_masks[1]), axis=0)
        out3 = np.concatenate((refine_masks[2], refine_masks[3]), axis=0)
        out = np.concatenate((out1, out2, out3), axis=1)
        Image.fromarray(out).save(f'results/{idx}.png')
        idx += 1

def quad_mask_save(gt, coarse, new_gt, q_sample):
    gt = gt.cpu().numpy().astype(np.uint8)
    gt = gt * 255
    coarse = coarse.cpu().numpy().astype(np.uint8)
    coarse = coarse * 255
    new_gt = new_gt.cpu().numpy().astype(np.uint8)
    new_gt = new_gt * 255
    q_sample = q_sample.cpu().numpy().astype(np.uint8)
    q_sample = q_sample * 255
    idx = 0
    for gt_mask, coarse_mask, nre_gt_mask, q in zip(gt, coarse, new_gt, q_sample):
        out1 = np.concatenate((gt_mask, coarse_mask), axis=0)
        out2 = np.concatenate((nre_gt_mask, q), axis=0)
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

def quad_refine_save(coarse, refine):
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
