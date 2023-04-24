import torch
import torch.nn.functional as F
import numpy as np
from .base_refinementor import BaseRefinementor
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
    bboxes = torch.zeros((N, 4), device=device, dtype=torch.float32)
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

def mask2bbox(mask):
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    x_1, x_2 = x[0], x[-1] + 1
    y_1, y_2 = y[0], y[-1] + 1
    return x_1, y_1, x_2, y_2

@DETECTORS.register_module()
class DiffRefinementor(BaseRefinementor):
    def __init__(self,
                 task,
                 denoise_model,
                 diffusion_cfg,
                 train_cfg,
                 test_cfg,
                 loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_texture=dict(type='TextureL1Loss', loss_weight=5.0),
                 num_classes=80,
                 init_cfg=None):
        super(DiffRefinementor, self).__init__(task=task, init_cfg=init_cfg)

        self.denoise_model = build_head(denoise_model)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(diffusion_cfg)
        self.loss_mask = build_loss(loss_mask)
        self.loss_texture = build_loss(loss_texture)
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
                      img_metas,
                      object_img,
                      object_gt_masks,
                      object_coarse_masks,
                      patch_img,
                      patch_gt_masks,
                      patch_coarse_masks):
        current_device = object_img.device
        img = torch.cat((object_img, patch_img), dim=0)
        x_start = torch.cat((self._bitmapmasks_to_tensor(object_gt_masks, current_device),
                             self._bitmapmasks_to_tensor(patch_gt_masks, current_device)), dim=0)
        x_last = torch.cat((self._bitmapmasks_to_tensor(object_coarse_masks, current_device),
                            self._bitmapmasks_to_tensor(patch_coarse_masks, current_device)), dim=0)
        t = uniform_sampler(self.num_timesteps, img.shape[0], current_device)
        # t = torch.tensor([5]*2, device=current_device)
        x_t = self.q_sample(x_start, x_last, t, current_device)
        # train_save(img, x_start, x_last, x_t, img_metas, t)
        z_t = torch.cat((img, x_t), dim=1)
        pred_logits = self.denoise_model(z_t, t)
        # train_save(img, x_last, x_t, (pred_logits>=0).float(), img_metas, t)
        iou_pred = self.cal_iou(x_start, pred_logits)
        losses = dict()
        losses['loss_mask'] = self.loss_mask(pred_logits, x_start)
        losses['loss_texture'] = self.loss_texture(pred_logits, x_start, t)
        losses['iou'] = iou_pred.mean()
        return losses
    
    def _bitmapmasks_to_tensor(self, bitmapmasks, current_device):
        tensor_masks = []
        for bitmapmask in bitmapmasks:
            tensor_masks.append(bitmapmask.masks)
        tensor_masks = np.stack(tensor_masks)
        tensor_masks = torch.tensor(tensor_masks, device=current_device, dtype=torch.float32)
        return tensor_masks
    
    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        # quad_mask_save(x_start, x_last, new_x_start, sample)
        return sample
    
    @torch.no_grad()
    def cal_iou(self, target, mask, eps=1e-3):
        target = target.clone().detach() >= 0.5
        mask = mask.clone().detach() >= 0
        si = (target & mask).sum(-1).sum(-1)
        su = (target | mask).sum(-1).sum(-1)
        return (si / (su + eps))

    def simple_test_instance(self, 
                             img_metas, 
                             img, 
                             coarse_masks,
                             dt_bboxes,
                             **kwargs):
        """Test without augmentation."""
        
        coarse_masks, tiny_coarse_masks, dt_bboxes = self._filter_tiny_instance(coarse_masks, dt_bboxes)

        if len(coarse_masks) == 0:
            assert len(tiny_coarse_masks) > 0
            img_masks = tiny_coarse_masks
        else:
            current_device = img.device
            object_imgs, object_masks, orisize_flag, object_coors = self._get_object_input(img, coarse_masks, img_metas, current_device)
            batch_max = self.test_cfg.get('batch_max', 0)
            num_ins = len(object_masks)
            if num_ins <= batch_max:
                xs = [(object_masks, object_imgs)]
            else:
                xs = []
                for idx in range(0, num_ins, batch_max):
                    end = min(num_ins, idx + batch_max)
                    xs.append((object_masks[idx: end], object_imgs[idx:end]))
            res, fine_probs = self.p_sample_loop(xs, current_device)
            img_masks = _do_paste_mask(res, object_coors, img_metas)

            use_local_step = self.test_cfg.get('use_local_step', False)
            if use_local_step:
                if not orisize_flag.all():
                    refined_masks = img_masks[~ orisize_flag]
                    local_refine_masks = img_masks[orisize_flag]
                    local_refine_masks = self.local_refine(local_refine_masks)
                    img_masks = torch.cat((refined_masks, local_refine_masks), dim=0)
            img_masks = img_masks >= 0.5
            img_masks = img_masks.cpu().numpy().astype(np.uint8)
            img_masks = np.concatenate((img_masks, tiny_coarse_masks), axis=0)
            
        # refine_save(object_masks, x)

        
        # single_mask_save(x.squeeze(1), 'object')
        # single_mask_save(img_masks, 'pan')

        assert len(img_masks) == len(dt_bboxes)
        bboxes = dt_bboxes[:, :5]
        labels = dt_bboxes[:, 5]
        labels = labels.astype(int)
        bbox_results = self._format_bboxes_results(bboxes, labels)
        mask_results = self._format_mask_results(img_masks, labels)
        return [(bbox_results, mask_results)]
    
    def _filter_tiny_instance(self, coarse_masks, dt_bboxes):
        area_thr = self.test_cfg.get('area_thr', 0)
        valid_idx = coarse_masks[0].areas >= area_thr
        invalid_idx = ~ valid_idx
        tiny_dt_bboxes = dt_bboxes[0][invalid_idx]
        dt_bboxes = dt_bboxes[0][valid_idx]
        dt_bboxes = np.concatenate((dt_bboxes, tiny_dt_bboxes), axis=0)
        return coarse_masks[0].masks[valid_idx], coarse_masks[0].masks[invalid_idx], dt_bboxes

    def _get_pan_input(self, img, coarse_masks, img_metas, current_device):
        img_h, img_w = img_metas[0]['img_shape'][:2]
        coarse_masks = coarse_masks[0].resize((256, 256))
        img = F.interpolate(img[:, :, :img_h, :img_w], size=(256, 256), mode='bilinear')
        x_last = torch.tensor(coarse_masks.masks, device=current_device)
        return img, x_last
    
    def _get_object_crop_coor(self, x_1, x_2, w, object_size):
        x_start = max(object_size/2, x_2 - object_size/2)
        x_end = min(x_1 + object_size/2, w - object_size/2)
        x_c = (x_start + x_end) / 2
        x_1_ob = max(int(x_c - object_size/2), 0)
        x_2_ob = min(int(x_c + object_size/2), w)
        return x_1_ob, x_2_ob
    
    def _get_object_input(self, img, coarse_masks, img_metas, current_device):
        img_h, img_w = img_metas[0]['img_shape'][:2]
        object_imgs, object_masks, orisize_flag, object_coors = [], [], [], []
        object_out_size = self.test_cfg.get('objtct_size', 256)
        pad_width = self.test_cfg.get('pad_width', 0)
        for mask in coarse_masks:
            x_1, y_1, x_2, y_2 = mask2bbox(mask)
            object_h, object_w = y_2 - y_1, x_2 - x_1
            if object_h > object_out_size or object_w > object_out_size:
                object_size = max(object_h, object_w) + pad_width
            else:
                object_size = object_out_size
            if img_h < object_size or img_w < object_size:
                object_imgs.append(F.interpolate(img, size=(object_out_size, object_out_size), mode='bilinear'))
                mask = torch.tensor(mask, device=current_device, dtype=torch.float32)
                object_masks.append(F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(object_out_size, object_out_size), mode='bilinear'))
                orisize_flag.append(object_out_size >= img_h and object_out_size >= img_w)
                object_coors.append(torch.tensor((0, 0, img_w, img_h), device=current_device))
            else:
                x_1_ob, x_2_ob = self._get_object_crop_coor(x_1, x_2, img_w, object_size)
                y_1_ob, y_2_ob = self._get_object_crop_coor(y_1, y_2, img_h, object_size)
                object_img = img[:, :, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
                object_mask = torch.tensor(mask[y_1_ob: y_2_ob, x_1_ob: x_2_ob], 
                                           device=current_device,
                                           dtype=torch.float32)
                object_imgs.append(F.interpolate(object_img, size=(object_out_size, object_out_size), mode='bilinear'))
                object_masks.append(F.interpolate(object_mask.unsqueeze(0).unsqueeze(0), size=(object_out_size, object_out_size), mode='bilinear'))
                orisize_flag.append(object_out_size >= (y_2_ob-y_1_ob) and object_out_size >= (x_2_ob-x_1_ob))
                object_coors.append(torch.tensor((x_1_ob, y_1_ob, x_2_ob, y_2_ob), device=current_device))
        object_imgs = torch.cat(object_imgs, dim=0)
        object_masks = torch.cat(object_masks, dim=0)
        object_coors = torch.stack(object_coors, dim=0)
        orisize_flag = torch.tensor(orisize_flag, device=current_device)
        return object_imgs, object_masks, orisize_flag, object_coors
    
    def local_refine(self, masks):
        return masks
    
    def p_sample_loop(self, xs, current_device):
        indices = list(range(self.num_timesteps))[::-1]
        res, fine_probs = [], []
        for data in xs:
            x, img = data
            cur_fine_probs = torch.zeros_like(x)
            for i in indices:
                t = torch.tensor([i] * x.shape[0], device=current_device)
                x, cur_fine_probs = self.p_sample(img, x, cur_fine_probs, t)
            res.append(x)
            fine_probs.append(cur_fine_probs)
        res = torch.cat(res, dim=0)
        fine_probs = torch.cat(fine_probs, dim=0)
        return res, fine_probs

    def p_sample(self, img, x, cur_fine_probs, t):
        z = torch.cat((img, x), dim=1)
        pred_logits = self.denoise_model(z, t)
        t = t[0].item()
        pred_x_start = (pred_logits >= 0).float()
        x_start_fine_probs = 2 * torch.abs(pred_logits.sigmoid() - 0.5)
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        if t > 0:
            sample_noise = torch.rand(size=x.shape, device=x.device)
            fine_map = (sample_noise < cur_fine_probs).float()
            x_prev = pred_x_start * fine_map + x * (1 - fine_map)
            # single_mask_save(cur_fine_probs, t, 'fine_probs')
            # sample_save(x, pred_x_start, x_prev)
            return x_prev, cur_fine_probs
        else:
            return pred_logits.sigmoid(), cur_fine_probs
    
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
        cls_masks = [[] for _ in range(self.num_classes)]
        for i in range(len(masks)):
            cls_masks[labels[i]].append(masks[i])
        return cls_masks

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError


def _do_paste_mask(masks, object_coors, img_metas):
    # masks: tensor(N, 1, H, W)
    # proposals: tensor(N, 5)
    device = masks.device
    img_h, img_w = img_metas[0]['ori_shape'][:2]
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(object_coors, 1, dim=1)  # each is Nx1

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

def train_save(img, x_start, x_last, x_t, img_metas, t):
    x_start = torch.cat([x_start]*3, dim=1)
    x_last = torch.cat([x_last]*3, dim=1)
    x_t = torch.cat([x_t]*3, dim=1)
    img = img.cpu().numpy()
    img = img.transpose((0, 2, 3, 1))
    img_mean = img_metas[0]['img_norm_cfg']['mean'].reshape(1, 1, 1, 3)
    img_std = img_metas[0]['img_norm_cfg']['std'].reshape(1, 1, 1, 3)
    img = img * img_std + img_mean
    img = img.astype(np.uint8)
    x_start = x_start.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    x_start = x_start * 255
    x_last = x_last.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    x_last = x_last * 255
    x_t = x_t.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    x_t = x_t * 255
    idx = 0
    for img, gt_mask, coarse_mask, q in zip(img, x_start, x_last, x_t):
        out1 = np.concatenate((img, gt_mask), axis=0)
        out2 = np.concatenate((coarse_mask, q), axis=0)
        out = np.concatenate((out1, out2), axis=1)
        ct = t[idx].item()
        Image.fromarray(out).save(f'results/{idx}_train.png')
        idx += 1

def refine_save(coarse, refine):
    coarse = coarse.squeeze(1).cpu().numpy().astype(np.uint8)
    coarse = coarse * 255
    refine = refine.squeeze(1).cpu().numpy().astype(np.uint8)
    refine = refine * 255
    idx = 0
    for coarse_mask, refine_mask in zip(coarse, refine):
        out2 = np.concatenate((coarse_mask, refine_mask), axis=1)
        Image.fromarray(out2).save(f'results/{idx}.png')
        idx += 1

def sample_save(x, x_start, x_prev):
    x = x.squeeze(1).cpu().numpy().astype(np.uint8)
    x = x * 255
    x_start = x_start.squeeze(1).cpu().numpy().astype(np.uint8)
    x_start = x_start * 255
    x_prev = x_prev.squeeze(1).cpu().numpy().astype(np.uint8)
    x_prev = x_prev * 255
    idx = 0
    for xt, coarse_mask, refine_mask in zip(x, x_start, x_prev):
        out2 = np.concatenate((xt, coarse_mask, refine_mask), axis=-1)
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

def single_mask_save(masks, t, file_name):
    masks = masks.squeeze(1).cpu().numpy()
    masks = (masks * 255).astype(np.uint8)
    for i, mask in enumerate(masks):
        Image.fromarray(mask).save(f'results/{file_name}_{i}_{t}.png')
