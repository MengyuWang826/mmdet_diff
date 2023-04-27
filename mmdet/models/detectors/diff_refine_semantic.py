import torch
import torch.nn.functional as F
import numpy as np
from .base_refinementor import BaseRefinementor
from ..builder import DETECTORS, build_head, build_loss
import numpy as np
from mmcv.ops import nms

from PIL import Image
import cv2

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
class DiffRefineSemantic(BaseRefinementor):
    def __init__(self,
                 task,
                 denoise_model,
                 diffusion_cfg,
                 train_cfg,
                 test_cfg,
                 loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_texture=dict(type='TextureL1Loss', loss_weight=5.0),
                 num_classes=77,
                 init_cfg=None):
        super(DiffRefineSemantic, self).__init__(task=task, init_cfg=init_cfg)

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
    
    def simple_test_semantic(self, img_metas, img, coarse_masks, **kwargs):
        if coarse_masks[0].masks.sum() == 0:
            return [(np.zeros_like(coarse_masks[0].masks[0]), img_metas[0])]
        current_device = img.device
        ori_shape = img_metas[0]['ori_shape'][:2]
        indices = list(range(self.num_timesteps))[::-1]
        object_indices = indices
        local_1_indices = indices
        local_2_indices = indices

        # object_step
        object_imgs, object_masks, object_coors = self._get_object_input(img, coarse_masks, ori_shape, current_device)
        object_masks, object_fine_probs = self.p_sample_loop([(object_masks, object_imgs, None)], 
                                                        object_indices, 
                                                        current_device, 
                                                        use_first_step_flag=True,
                                                        use_last_step_flag=True)
        object_masks = (object_masks >= 0.5)

        # local_save(object_imgs, object_masks, object_masks, object_fine_probs, img_metas, 'object')

        img_masks = _do_paste_mask(object_masks, object_coors, img_metas)
        img_mask = img_masks.any(dim=0).float()

        # local_step_1
        patch_size = self.test_cfg.get('patch_size_1', 0)
        patch_imgs, patch_masks, patch_coors = self._get_local_input(img, img_mask, ori_shape, patch_size)
        batch_max = self.test_cfg.get('batch_max', 0)
        num_ins = len(patch_imgs)
        if num_ins <= batch_max:
            xs = [(patch_masks, patch_imgs, None)]
        else:
            xs = []
            for idx in range(0, num_ins, batch_max):
                end = min(num_ins, idx + batch_max)
                xs.append((patch_masks[idx: end], patch_imgs[idx:end], None))
        local_masks, _ = self.p_sample_loop(xs, 
                                            local_1_indices, 
                                            patch_imgs.device,
                                            use_first_step_flag=False,
                                            use_last_step_flag=True)
        mask = self.paste_local_patch(local_masks, img_mask, patch_coors, patch_size)

        # one_mask_save(mask, 'step1')

        # local_step_2
        patch_size = self.test_cfg.get('patch_size_2', 0)
        patch_imgs, patch_masks, patch_coors = self._get_local_input(img, mask, ori_shape, patch_size)
        batch_max = self.test_cfg.get('batch_max', 0)
        num_ins = len(patch_imgs)
        if num_ins <= batch_max:
            xs = [(patch_masks, patch_imgs, None)]
        else:
            xs = []
            for idx in range(0, num_ins, batch_max):
                end = min(num_ins, idx + batch_max)
                xs.append((patch_masks[idx: end], patch_imgs[idx:end], None))
        local_masks, _ = self.p_sample_loop(xs, 
                                            local_2_indices, 
                                            patch_imgs.device,
                                            use_first_step_flag=True,
                                            use_last_step_flag=True)
        mask = self.paste_local_patch(local_masks, img_mask, patch_coors, patch_size)

        return [(mask.cpu().numpy(), img_metas[0])]
        
    def _get_object_input(self, img, coarse_masks, ori_shape, current_device):
        img_h, img_w = ori_shape
        ob_pad_width = self.test_cfg.get('ob_pad_width', 0)
        size_thr = self.test_cfg.get('size_thr', 0)
        do_blur = self.test_cfg.get('do_blur', False)
        model_size = self.test_cfg.get('model_size', 256)
        object_imgs, object_masks, object_coors = [], [], []
        
        mask = coarse_masks[0].masks[0]
        img_mask = torch.tensor(coarse_masks[0].masks, dtype=torch.float32, device=current_device)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x_1, y_1, w, h = cv2.boundingRect(cnt)
            if w > size_thr or h > size_thr:
                object_size = max(w, h, model_size) + ob_pad_width
                x_1_ob, x_2_ob = self._get_object_coor(x_1 + w/2, img_w, object_size)
                y_1_ob, y_2_ob = self._get_object_coor(y_1 + h/2, img_h, object_size)
                object_img = img[:, :, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
                object_img = F.interpolate(object_img, size=model_size, mode='nearest')
                object_mask = img_mask[:, y_1_ob: y_2_ob, x_1_ob: x_2_ob]
                object_mask = F.interpolate(object_mask.unsqueeze(0), size=model_size)
                object_mask = (object_mask >= 0.5).float()
                object_coor = (x_1_ob, y_1_ob, x_2_ob, y_2_ob)
                if do_blur:
                    object_img = self.uniform_blur(object_img, current_device)
                object_imgs.append(object_img)
                object_masks.append(object_mask)
                object_coors.append(object_coor)
        object_imgs = torch.cat(object_imgs, dim=0)
        object_masks = torch.cat(object_masks, dim=0)
        return object_imgs, object_masks, object_coors
    
    def _get_object_coor(self, x_c, w, object_size):
        x_1_ob = int(max(x_c - object_size/2, 0))
        x_2_ob = int(min(x_c + object_size/2, w))
        return x_1_ob, x_2_ob

    def uniform_blur(self, object_img, current_device, fileter_size=(5,5)):
        object_img = object_img.cpu().numpy()[0]
        object_img = object_img.transpose(1, 2, 0)
        object_img = cv2.blur(object_img, fileter_size)
        object_img = torch.tensor(object_img.transpose(2, 0, 1), dtype=torch.float32, device=current_device).unsqueeze(0)
        return object_img

    def _get_local_input(self, img, img_mask, ori_shape, patch_size):
        x_1, y_1, x_2, y_2 = mask2bbox(img_mask.cpu().numpy())
        img_h, img_w = ori_shape
        overlap_width = self.test_cfg.get('overlap_fraction', 0) * patch_size
        # x_start = max(x_1, patch_size // 2)
        # x_end = min(x_2, img_w - patch_size // 2)
        # y_start = max(y_1, patch_size // 2)
        # y_end = min(y_2, img_h - patch_size // 2)
        x_start = patch_size // 2
        x_end = img_w - patch_size // 2
        y_start =patch_size // 2
        y_end = img_h - patch_size // 2
        x = np.arange(x_start, x_end, patch_size-overlap_width)
        y = np.arange(y_start, y_end, patch_size-overlap_width)
        X, Y = np.meshgrid(x, y)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        x_1 = X - patch_size//2
        x_2 = X + patch_size//2
        y_1 = Y - patch_size//2
        y_2 = Y + patch_size//2
        patch_coors = np.concatenate((x_1, y_1, x_2, y_2), axis=-1).astype(np.int16)
        x = img_mask.unsqueeze(0).unsqueeze(0)
        return self._crop_patch(img, x, patch_coors)
    
    def _crop_patch(self, img, mask, patch_coors):
        model_size = self.test_cfg.get('objtct_size', 256)
        patch_imgs, patch_masks, new_patch_coors = [], [], []
        for coor in patch_coors:
            patch_mask = mask[:, :, coor[1]:coor[3], coor[0]:coor[2]]
            if patch_mask.sum():
                patch_imgs.append(img[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                # patch_fine_probs.append(cur_fine_probs[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_masks.append(patch_mask)
                new_patch_coors.append(coor)
        patch_imgs = F.interpolate(torch.cat(patch_imgs, dim=0), size=(model_size, model_size))
        patch_masks = F.interpolate(torch.cat(patch_masks, dim=0), size=(model_size, model_size))
        # patch_fine_probs = F.interpolate(torch.cat(patch_fine_probs, dim=0), size=(model_size, model_size))
        patch_masks = (patch_masks >= 0.5).float()
        return patch_imgs, patch_masks, new_patch_coors
    
    def paste_local_patch(self, local_masks, mask, patch_coors, patch_size):
        zeor_mask = torch.zeros_like(mask)
        weight = torch.zeros_like(mask)
        # local_masks = local_masks.squeeze(1)
        local_masks = F.interpolate(local_masks, size=patch_size).squeeze(1)
        local_weight = torch.ones_like(local_masks[0])
        for local_mask, coor in zip(local_masks, patch_coors):
            zeor_mask[coor[1]:coor[3], coor[0]:coor[2]] += local_mask
            weight[coor[1]:coor[3], coor[0]:coor[2]] += local_weight
        zeor_mask = zeor_mask / weight
        zeor_mask = (zeor_mask >= 0.5).float()
        for coor in patch_coors:
            mask[coor[1]:coor[3], coor[0]:coor[2]] = zeor_mask[coor[1]:coor[3], coor[0]:coor[2]]
        return mask
    
    def p_sample_loop(self, xs, indices, current_device, use_first_step_flag=False, use_last_step_flag=True):
        res, fine_probs = [], []
        for data in xs:
            x, img, cur_fine_probs = data
            if cur_fine_probs is None:
                cur_fine_probs = torch.zeros_like(x)
            for i in indices:
                t = torch.tensor([i] * x.shape[0], device=current_device)
                first_step_flag = (use_first_step_flag and i==indices[0])
                last_step_flag = (use_last_step_flag and i==indices[-1])
                x, cur_fine_probs = self.p_sample(img, x, cur_fine_probs, t, first_step_flag, last_step_flag)
            res.append(x)
            fine_probs.append(cur_fine_probs)
        res = torch.cat(res, dim=0)
        fine_probs = torch.cat(fine_probs, dim=0)
        return res, fine_probs

    def p_sample(self, img, x, cur_fine_probs, t, first_step_flag, last_step_flag):
        if first_step_flag:
            sample_noise = torch.rand(size=x.shape, device=img.device)
            zero = torch.zeros_like(x)
            sample_map = (sample_noise <= 0.5).float()
            x_last = x * sample_map + (1 - sample_map) * zero
            z = torch.cat((img, x_last), dim=1)
        else:
            z = torch.cat((img, x), dim=1)
        pred_logits = self.denoise_model(z, t)
        t = t[0].item()
        pred_x_start = (pred_logits >= 0).float()
        x_start_fine_probs = 2 * torch.abs(pred_logits.sigmoid() - 0.5)
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        if last_step_flag:
            return pred_logits.sigmoid(), cur_fine_probs
        else:
            sample_noise = torch.rand(size=x.shape, device=x.device)
            fine_map = (sample_noise < cur_fine_probs).float()
            x_prev = pred_x_start * fine_map + x * (1 - fine_map)
            # single_mask_save(x_prev, t, 'x_prev')
            # single_mask_save(pred_x_start, t, 'start')
            return x_prev, cur_fine_probs

    

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError


def _do_paste_mask(masks, object_coors, img_metas):
    # masks: tensor(N, 1, H, W)
    # proposals: tensor(N, 5)
    device = masks.device
    if not isinstance(object_coors, torch.Tensor):
        object_coors = torch.tensor(object_coors, device=device).reshape(-1, 4)
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
def find_float_boundary(mask, width=3):
    # Extract boundary from instance mask
    maskdt = mask.unsqueeze(0).unsqueeze(0).float()
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder, stride=1, padding=width//2)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    fbmask = fbmask[0, 0]
    y_c, x_c = torch.where(fbmask)
    scores = fbmask[y_c, x_c]
    return y_c, x_c, scores

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
    x_start = x_start >= 0.5
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
        Image.fromarray(out).save(f'results/{idx}_train.png')
        idx += 1

def local_save(img, x_start, x_last, x_t, img_metas, filename):
    x_start = torch.cat([x_start]*3, dim=1)
    x_last = torch.cat([x_last]*3, dim=1)
    x_t = torch.cat([x_t]*3, dim=1)
    img = img.cpu().numpy()
    img = img.transpose((0, 2, 3, 1))
    img_mean = img_metas[0]['img_norm_cfg']['mean'].reshape(1, 1, 1, 3)
    img_std = img_metas[0]['img_norm_cfg']['std'].reshape(1, 1, 1, 3)
    img = img * img_std + img_mean
    imgs = img.astype(np.uint8)
    x_start = x_start.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    x_start = x_start * 255
    x_last = x_last.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    x_last = x_last * 255
    x_t = x_t*255
    x_t = x_t.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
    idx = 0
    for img, gt_mask, coarse_mask, q in zip(imgs, x_start, x_last, x_t):
        out1 = np.concatenate((img, gt_mask), axis=0)
        out2 = np.concatenate((coarse_mask, q), axis=0)
        out = np.concatenate((out1, out2), axis=1)
        Image.fromarray(out).save(f'results/{idx}_{filename}.png')
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
        Image.fromarray(mask).save(f'results/{i}_{t}_{file_name}.png')

def single_img_save(imgs, img_metas):
    imgs = imgs.cpu().numpy()
    imgs = imgs.transpose((0, 2, 3, 1))
    img_mean = img_metas[0]['img_norm_cfg']['mean'].reshape(1, 1, 1, 3)
    img_std = img_metas[0]['img_norm_cfg']['std'].reshape(1, 1, 1, 3)
    imgs = imgs * img_std + img_mean
    imgs = imgs.astype(np.uint8)
    for idx, img in enumerate(imgs):
        Image.fromarray(img).save(f'results/{idx}_img.png')
        idx += 1

def one_mask_save(mask, filename):
    out = mask.cpu().numpy()
    out = (out * 255).astype(np.uint8)
    Image.fromarray(out).save(f"results/{filename}.png")
