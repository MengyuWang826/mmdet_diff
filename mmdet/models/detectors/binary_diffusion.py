import copy
from numpy.random.mtrand import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch._C import device
from einops import rearrange, reduce, repeat

from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from .base import BaseDetector

from PIL import Image
import cv2

def output_save(img, gt, mask_c, mask_f, filename, color_map=(220, 20, 60)):

    color_map = np.array(color_map).reshape(1, 1, -1)
    img = img.astype(np.uint8)

    mask_f = torch.stack([mask_f]*3, dim=-1)
    mask_f = mask_f.cpu().numpy()
    mask_f = (mask_f * color_map).astype(np.uint8)
    mask_f = cv2.addWeighted(img, 0.5, mask_f, 0.5, 0)
    # mask_f = np.repeat(mask_f, 3, axis=2)
    # mask_f = mask_f * 255
    mask_c = torch.stack([mask_c]*3, dim=-1)
    mask_c = mask_c.cpu().numpy()
    mask_c = (mask_c * color_map).astype(np.uint8)
    mask_c = cv2.addWeighted(img, 0.5, mask_c, 0.5, 0)
    # mask_c = np.repeat(mask_c, 3, axis=2)
    # mask_c = mask_c * 255
    ann = torch.stack([gt]*3, dim=-1)
    ann = ann.cpu().numpy()
    ann = (ann * color_map).astype(np.uint8)
    ann = cv2.addWeighted(img, 0.5, ann, 0.5, 0)
    # ann = ann * 255
    # ann = np.zeros_like(img)
    output1 = np.concatenate((img, ann), axis=0)
    output2 = np.concatenate((mask_c, mask_f), axis=0)
    output = np.concatenate((output1, output2), axis=1)
    # if not os.path.exists(root+img_id):
    #     os.mkdir(root+img_id)
    Image.fromarray(output).save(filename)

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

def pos_grid(bs, h, w, current_device):
    """
    This is a more standard version of the position embedding, very similar to
    the one used by the Attention is all you need paper, generalized to work on
    images.
    """
    mask = torch.ones((bs, h, w), device=current_device)
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    return torch.stack((x_embed, y_embed), dim=1)

def get_diffusion_betas(diffusion_betas):
    """Get betas from the hyperparameters."""
    if diffusion_betas['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return np.linspace(diffusion_betas['start'], diffusion_betas['stop'], diffusion_betas['num_timesteps'])
    elif diffusion_betas['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = (
            np.arange(diffusion_betas['num_timesteps'] + 1, dtype=np.float64) /
            diffusion_betas['num_timesteps'])
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas
    elif diffusion_betas['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / np.linspace(diffusion_betas['num_timesteps'], 1., diffusion_betas['num_timesteps'])
    else:
        raise NotImplementedError(diffusion_betas['type'])


@DETECTORS.register_module()
class BinaryDiffusionRefine(BaseDetector):
    def __init__(self,
                 backbone,
                 denoise_model,
                 diffusion,
                #  loss_cfg,
                 max_sample_points=1000,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BinaryDiffusionRefine, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.ups = nn.ModuleList()
        for i in range(neck['num_outs']):
            self.ups.append(nn.Upsample(scale_factor=2**(i+1)))
        denoise_model.update(train_cfg=copy.deepcopy(train_cfg))
        denoise_model.update(test_cfg=copy.deepcopy(test_cfg))
        self.denoise_model = build_head(denoise_model)
        self.max_sample_points = max_sample_points
        self.num_classes = 2
        # for i in range(4)
        # self.loss = build_loss(loss_cfg)
        # self.loss_dice = build_loss(loss_dice)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(**diffusion)
    
    def _diffusion_init(self, 
                        betas, 
                        loss_type,
                        hybrid_coeff=None):

        self.loss_type = loss_type  # kl, hybrid, cross_entropy_x_start
        self.hybrid_coeff = hybrid_coeff

        # Data \in {0, ..., num_pixel_vals-1}
        self.num_pixel_vals = self.num_classes
        self.eps = 1.e-6

        # Computations here in float64 for accuracy
        self.betas = get_diffusion_betas(betas)
        self.num_timesteps = self.betas.shape[0]

        # Construct transition matrices for q(x_t|x_{t-1})
        # NOTE: t goes from {0, ..., T-1}
        q_one_step_mats = [self._get_transition_mat(t) for t in range(0, self.num_timesteps)]
        self.q_onestep_mats = np.stack(q_one_step_mats, axis=0) 
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                            self.num_pixel_vals,
                                            self.num_pixel_vals)

        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
        # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = np.matmul(q_mat_t, self.q_onestep_mats[t])
            q_mats.append(q_mat_t)
        self.q_mats = np.stack(q_mats, axis=0)
        assert self.q_mats.shape == (self.num_timesteps, self.num_pixel_vals, self.num_pixel_vals)
        self.q_mats = torch.tensor(self.q_mats, device='cpu')

        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        self.transpose_q_onestep_mats = np.transpose(self.q_onestep_mats,
                                                    axes=(0, 2, 1))
        self.transpose_q_onestep_mats = torch.tensor(self.transpose_q_onestep_mats, device='cpu')
        del self.q_onestep_mats
    
    def _get_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        | 1-0.5beta_t   0.5beta_t |
        |  0.5beta_t   1-0.5beta_t|

        Contrary to the band diagonal version, this method constructs a transition
        matrix with uniform probability to all other states.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
        """
        beta_t = self.betas[t]
        mat = np.full(shape=(self.num_pixel_vals, self.num_pixel_vals),
                    fill_value=beta_t/float(self.num_pixel_vals),
                    dtype=np.float64)
        diag_indices = np.diag_indices_from(mat)
        diag_val = 1. - beta_t * (self.num_pixel_vals-1.)/self.num_pixel_vals
        mat[diag_indices] = diag_val
        return mat
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_bboxes):

        # Add noise to data
        current_device = img.device
        img_feats, pos_ids, zero_masks, x_start = self.extract_feat(img, gt_masks, gt_bboxes, current_device)
        sample_shape = (x_start.shape[0], x_start.shape[1], self.num_pixel_vals)

        noise = torch.rand(size=sample_shape, device=current_device)
        t = uniform_sampler(self.num_timesteps, x_start.shape[0], current_device)

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_with_c = torch.cat((x_t.unsqueeze(-1), img_feats), dim=-1)

        pred_logits = self.denoise_model(x_with_c, pos_ids, t)
        pred_x_start = F.softmax(pred_logits, dim=-1)

        losses = dict()
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits = self.q_posterior_logits(pred_x_start, x_t, t, x_start_logits=True)
        vb_loss = self.categorical_kl_logits(true_logits, model_logits)
        zero_time_masks = (t > 0).float().reshape(-1, 1, 1)
        losses['vb_loss'] = (vb_loss * zero_masks * zero_time_masks).mean()
        pred_logits = pred_logits.reshape(-1, 2)
        x_start = x_start.reshape(-1).type(torch.int64)
        ce_loss = F.cross_entropy(pred_logits, x_start, reduction='none')
        losses['ce_loss'] = (ce_loss * zero_masks.reshape(-1)).mean()

        return losses
    
    def categorical_kl_logits(self, logits1, logits2):
        """KL divergence between categorical distributions.
        Distributions parameterized by logits.
        Args:
            logits1: logits of the first distribution. Last dim is class dim.
            logits2: logits of the second distribution. Last dim is class dim.
            eps: float small number to avoid numerical issues.
        Returns:
            KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
        """
        out = (
            F.softmax(logits1 + self.eps, dim=-1) *
            (F.log_softmax(logits1 + self.eps, dim=-1) -
            F.log_softmax(logits2 + self.eps, dim=-1)))
        return out
    
    def simple_test(self,
                    img,
                    img_metas,
                    gt_masks,
                    coarse_masks,
                    labels,
                    max_batch=50,
                    rescale=False):
        new_accs, old_accs = [], []
        current_device = img.device
        img_feats, poses, coarse_results, x_starts = self.extract_feat_test(img, coarse_masks, gt_masks, current_device)
        refine_masks = []
        idx = 0
        img = F.interpolate(img, scale_factor=0.5, mode='bilinear')
        img = img[0].permute(1, 2, 0).cpu().numpy()
        mean = img_metas[0]['img_norm_cfg']['mean'].reshape(1, 1, -1)
        std = img_metas[0]['img_norm_cfg']['std'].reshape(1, 1, -1)
        img_id = int(img_metas[0]['ori_filename'][:-4])
        img = img * std + mean
        for ins_feat, pos, x_start, coarse_result in zip(img_feats, poses, x_starts, coarse_results):
            ins_feat = ins_feat.transpose(0, 1).unsqueeze(0)
            # x = x_start
            gt_points = coarse_result[2][coarse_result[0]]
            ones = torch.ones_like(gt_points)
            old_acc = ((gt_points == x_start).sum() / ones.sum()).item()
            old_accs.append(old_acc)
            old_acc = round(old_acc, 4)
            pos = pos.transpose(0, 1).unsqueeze(0)
            mask = coarse_result[1]
            coarse_mask = mask.clone().detach()
            # mask_save(coarse_mask, f'results/coarse_{idx}.png')
            # indices = list(range(self.num_timesteps))[::-1]
            # for i in indices:
            #     t = torch.tensor([i], device=current_device)
            #     x = x.unsqueeze(0).unsqueeze(-1)
            #     x_with_c = torch.cat((x, ins_feat), dim=-1)
            #     x = self.denoise_model(x_with_c, pos, t).squeeze(0).squeeze(-1)
            #     x = F.softmax(x, dim=-1)
            #     x_pred = x.argmax(dim=-1)
            #     mask[coarse_result[0]] = x
            #     refine_mask = mask.cpu().numpy()
            #     mask_save(refine_mask, f'results/refine_{idx}_{i}.png')

            sample = self.p_sample_loop(ins_feat, pos, x_start, current_device)
            sample = torch.argmax(sample, dim=-1).squeeze(0)
            sample = sample.float()

            
            new_acc = ((gt_points == sample).sum() / ones.sum()).item()
            new_accs.append(new_acc)
            new_acc = round(new_acc, 4)

            # sample = torch.ones(size=x_start.shape, device=current_device, dtype=torch.float32)

            # coarse_mask = mask.cpu().numpy()
            # mask_save(coarse_mask, f'results/coarse_{idx}.png')
            mask[coarse_result[0]] = sample
            refine_mask = mask
            output_save(img, coarse_result[2], coarse_mask, refine_mask, filename=f'results/{img_id}_{old_acc}_{new_acc}.png')
            # output = torch.cat((coarse_mask, refine_mask), dim=-1)
            # output = output.cpu().numpy()
            # mask_save(output, f'results/{idx}.png')
            refine_masks.append(mask)
            idx += 1
        return new_accs, old_accs

    def extract_feat(self, img, masks, bboxes, current_device):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        low_lvl_feat = x[0]
        fpn_feats = self.neck(x[1:])
        multi_lvl_feats = []
        for idx, up in enumerate(self.ups):
            multi_lvl_feats.append(up(fpn_feats[idx]))
        multi_lvl_feats = torch.cat([low_lvl_feat, *multi_lvl_feats], dim=1)
        bs, h, w = multi_lvl_feats.shape[0], multi_lvl_feats.shape[-2], multi_lvl_feats.shape[-1]
        pos = pos_grid(bs, h, w, current_device)
        x_y_max = torch.tensor((w, h), device=current_device).unsqueeze(-1)
        feats, pos_ids, zero_masks, x_starts = [], [], [], []
        for batch_id, bitmapmasks in enumerate(masks):
            img_bboxes = bboxes[batch_id]
            center_coors = torch.stack(
                ((img_bboxes[:, 0] + img_bboxes[:, 2]) / 2 , (img_bboxes[:, 1] + img_bboxes[:, 3]) / 2),
                dim=-1).int()
            for ins_id in range(len(bitmapmasks.areas)):
                if ins_id > 50:
                    break
                mask = bitmapmasks.masks[ins_id]
                area = bitmapmasks.areas[ins_id]
                boundary_region, boundary_target = generate_boundary_region(mask, current_device, area)
                f = multi_lvl_feats[batch_id, :, boundary_region]
                p = pos[batch_id, :, boundary_region]
                center_coor = center_coors[ins_id].unsqueeze(-1)
                p = (p - center_coor) / x_y_max
                num_points = p.shape[-1]
                if num_points >= self.max_sample_points:
                    sample_points_idx = np.random.choice(num_points, size=self.max_sample_points, replace=False)
                    f = f[:, sample_points_idx]
                    p = p[:, sample_points_idx]
                    boundary_target = boundary_target[sample_points_idx]
                    zero_mask = torch.ones(size=(self.max_sample_points, ), device=current_device)
                elif num_points > 1:
                    f = torch.cat((f, torch.zeros(size=(f.shape[0], self.max_sample_points-num_points), device=current_device)), dim=-1)
                    p = torch.cat((p, torch.zeros(size=(p.shape[0], self.max_sample_points-num_points), device=current_device)), dim=-1)
                    boundary_target = torch.cat((boundary_target, torch.zeros(size=(self.max_sample_points-num_points, ), device=current_device)), dim=-1)
                    zero_mask = torch.zeros(size=(self.max_sample_points, ), device=current_device)
                    zero_mask[:num_points] = 1
                else:
                    continue
                feats.append(f)
                pos_ids.append(p)
                zero_masks.append(zero_mask)
                x_starts.append(boundary_target)
        # print(len(feats))
        feats = torch.stack(feats, dim=0).transpose(1, 2)
        pos_ids = torch.stack(pos_ids, dim=0).transpose(1, 2)
        zero_masks = torch.stack(zero_masks, dim=0).unsqueeze(-1)
        x_starts = torch.stack(x_starts, dim=0)
        return feats, pos_ids, zero_masks, x_starts
    
    # def extract_feat_test(self, img, coarse_masks, gt_masks, current_device):
    #     """Directly extract features from the backbone and neck."""
    #     x = self.backbone(img)
    #     low_lvl_feat = x[0]
    #     fpn_feats = self.neck(x[1:])
    #     multi_lvl_feats = []
    #     for idx, up in enumerate(self.ups):
    #         multi_lvl_feats.append(up(fpn_feats[idx]))
    #     multi_lvl_feats = torch.cat([low_lvl_feat, *multi_lvl_feats], dim=1)
    #     bs, h, w = multi_lvl_feats.shape[0], multi_lvl_feats.shape[-2], multi_lvl_feats.shape[-1]
    #     pos = pos_grid(bs, h, w, current_device)
    #     x_y_max = torch.tensor((w, h), device=current_device).unsqueeze(-1)
    #     feats, points_pos, x_starts, coarse_results = [], [], [], []
    #     for batch_id, bitmapmasks in enumerate(coarse_masks):
    #         for ins_id in range(len(bitmapmasks.areas)):
    #             mask = bitmapmasks.masks[ins_id]
    #             area = bitmapmasks.areas[ins_id]
    #             boundary_region, boundary_target, coarse_mask = generate_boundary_region(mask, current_device, area, test_mode=True)
    #             x_any, y_any = torch.any(coarse_mask, dim=0), torch.any(coarse_mask, dim=1)
    #             x = torch.where(x_any)[0]
    #             y = torch.where(y_any)[0]
    #             x_center, y_center = (x[0] + x[-1] + 1) / 2, (y[0] + y[-1] + 1) / 2
    #             center_coor = torch.tensor([x_center, y_center], dtype=torch.int32, device=current_device).unsqueeze(-1)
    #             f = multi_lvl_feats[batch_id, :, boundary_region]
    #             p = pos[batch_id, :, boundary_region]
    #             p = (p - center_coor) / x_y_max
    #             feats.append(f)
    #             points_pos.append(p)
    #             x_starts.append(boundary_target)
    #             coarse_results.append((boundary_region, coarse_mask))
    #     return feats, points_pos, coarse_results, x_starts

    def extract_feat_test(self, img, coarse_masks, gt_masks, current_device):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        low_lvl_feat = x[0]
        fpn_feats = self.neck(x[1:])
        multi_lvl_feats = []
        for idx, up in enumerate(self.ups):
            multi_lvl_feats.append(up(fpn_feats[idx]))
        multi_lvl_feats = torch.cat([low_lvl_feat, *multi_lvl_feats], dim=1)
        bs, h, w = multi_lvl_feats.shape[0], multi_lvl_feats.shape[-2], multi_lvl_feats.shape[-1]
        pos = pos_grid(bs, h, w, current_device)
        x_y_max = torch.tensor((w, h), device=current_device).unsqueeze(-1)
        feats, points_pos, x_starts, coarse_results = [], [], [], []
        gt_masks = list(gt_masks[0].to(current_device))
        coarse_masks = list(coarse_masks[0].to(current_device))
        for gt_mask, coarse_mask in zip(gt_masks, coarse_masks):
            area = coarse_mask.sum().item()
            boundary_region, boundary_target = generate_boundary_region(coarse_mask, current_device, area, test_mode=True)
            x_any, y_any = torch.any(coarse_mask, dim=0), torch.any(coarse_mask, dim=1)
            x = torch.where(x_any)[0]
            y = torch.where(y_any)[0]
            x_center, y_center = (x[0] + x[-1] + 1) / 2, (y[0] + y[-1] + 1) / 2
            center_coor = torch.tensor([x_center, y_center], dtype=torch.int32, device=current_device).unsqueeze(-1)
            f = multi_lvl_feats[0, :, boundary_region]
            p = pos[0, :, boundary_region]
            p = (p - center_coor) / x_y_max
            feats.append(f)
            points_pos.append(p)
            x_starts.append(boundary_target)
            coarse_results.append((boundary_region, coarse_mask, gt_mask))
        return feats, points_pos, coarse_results, x_starts
    
    def aug_test(self, img, img_metas, rescale=False):
        pass

    def _at(self, a, t, x):
        """Extract coefficients at diffusion_betasified timesteps t and conditioning data x.

        Args:
        a: np.ndarray: plain NumPy float64 array of constants indexed by time.
        t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
        x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one hot representation, but have integer
            values representing the class values.

        Returns:
        a[t, x]: jnp.ndarray: Jax array.
        """
        a = a.to(x.device)
        t_broadcast = t.unsqueeze(-1).type(torch.long)
        x_idx = x.type(torch.long)

        # x.shape = (bs, height, width, channels)
        # t_broadcast_shape = (bs, 1, 1, 1)
        # a.shape = (num_timesteps, num_pixel_vals, num_pixel_vals)
        # out.shape = (bs, height, width, channels, num_pixel_vals)
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t_broadcast, x_idx]

    def _at_onehot(self, a, t, x):
        """Extract coefficients at diffusion_betasified timesteps t and conditioning data x.

        Args:
        a: np.ndarray: plain NumPy float64 array of constants indexed by time.
        t: jnp.ndarray: Jax array of time indices, shape = (bs,).
        x: jnp.ndarray: jax array, shape (bs, ..., num_pixel_vals), float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, ..., num_pixel_vals)
        """
        at = a[t].to(x.device).type(torch.float32)
        return torch.matmul(x, at)

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
        x_start: tensor of shape [bs, c, h, w]
        t: tensor of shape (bs,).

        """
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

        Args:
        x_start: jnp.array: original clean data, in integer form (not onehot).
            shape = (bs, ...).
        t: :jnp.array: timestep of the diffusion process, shape (bs,).
        noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
            Should be of shape (*x_start.shape, num_pixel_vals).

        Returns:
        sample: jnp.ndarray: same shape as x_start. noisy data.
        """
        # q_prob = self.q_probs(x_start, t)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        gumbel_noise = - torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start)."""

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        # for i in range(1, 10):
        #     fact1 = self._at_onehot(self.transpose_q_onestep_mats, t-i, fact1.type(torch.float32))
        if x_start_logits:
            x_start = F.softmax(x_start, dim=-1)
            fact2 = self._at_onehot(self.q_mats, t-1, x_start)
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        # pos_prob = fact1 * fact2
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        return out

    def p_logits(self,
                 x,
                 feats,
                 pos,
                 t):
        """Compute logits of p(x_{t-1} | x_t)."""
        x_with_c = torch.cat((x, feats), dim=-1)
        pred_x_start_logits = self.denoise_model(x_with_c, pos, t)
        if t[0].item() > 0:
            model_logits = self.q_posterior_logits(pred_x_start_logits, x.squeeze(-1), t, x_start_logits=True)
        else:
            model_logits = pred_x_start_logits

        # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
        # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
        return model_logits, pred_x_start_logits

    # === Sampling ===

    def p_sample_loop(self,
                      feats,
                      pos,
                      x_start,
                      current_device):
        """Ancestral sampling."""
        x_start = x_start.unsqueeze(0)
        indices = list(range(1000))[::-1]
        sample_shape = x_start.shape
        noise_shape = (sample_shape[0], sample_shape[1], self.num_pixel_vals)
        batch_size = sample_shape[0]
        x = x_start
        for i in indices:
            # if i % 100 == 0:
            #     print(f'current_iter: {i}')
            x = x.unsqueeze(-1)
            t = torch.tensor([i] * batch_size, device=current_device)
            noise = torch.rand(size=noise_shape, device=current_device)
            x = self.p_sample(
                x=x,
                feats=feats,
                pos=pos,
                t=t,
                noise=noise)
        return x
    
    def p_sample(self, 
                 x,
                 feats,
                 pos,
                 t,
                 noise):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        model_logits, pred_x_start_logits = self.p_logits(x=x, feats=feats, pos=pos, t=t)

        # No noise when t == 0
        # NOTE: for t=0 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case.
        if t[0].item() > 0:
            gumbel_noise = -torch.log(-torch.log(noise))
            sample = torch.argmax(model_logits + gumbel_noise, dim=-1)
        else:
            sample = pred_x_start_logits

        return sample

def generate_boundary_region(mask, current_device, area, test_mode=False):
    # mask_target = torch.tensor(mask, dtype=torch.float32, device=current_device)
    mask_target = mask.unsqueeze(0).unsqueeze(0)

    boundary_width = max((int(np.sqrt(area) / 10), 2))

    # boundary region
    kernel_size = 2 * boundary_width + 1
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=current_device).requires_grad_(False)
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
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    boundary_target = mask_target[boundary_inds]
    return boundary_inds.squeeze(0).squeeze(0), boundary_target.squeeze(0).squeeze(0)