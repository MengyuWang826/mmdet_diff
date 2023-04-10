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

def gt_mask_save(gt_mask, root):
    # gt_mask = gt_mask.cpu().numpy()
    gt_mask = (gt_mask.astype(np.uint8))*255
    Image.fromarray(gt_mask).save(root+'mask.png')

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def uniform_sampler(num_steps, batch_size, device):
    indices_np = np.random.choice(num_steps, size=(batch_size,))
    indices = torch.from_numpy(indices_np).long().to(device)
    return indices

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def cal_iou(mask, bbox):
    bbox_map = np.zeros_like(mask)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if x2==x1 or y2 ==y1:
        return 0
    bbox_map[y1:y2, x1:x2] = 1
    si = np.sum(mask & bbox_map)
    su = np.sum(mask | bbox_map)
    if su==0:
        return 0
    else:
        return si/su

ModelMeanType = set(['prev_x', 'start_x', 'epsilon'])
ModelVarType = set(['learned', 'fixed_small', 'fixed_large', 'learned_range'])
LossType = set(['mse', 'rescaled_mse', 'kl', 'rescaled_kl'])


@DETECTORS.register_module()
class DiffusionRefine(BaseDetector):
    def __init__(self,
                 backbone,
                 denoise_model,
                 diffusion,
                 loss_cfg,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DiffusionRefine, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        denoise_model.update(train_cfg=copy.deepcopy(train_cfg))
        denoise_model.update(test_cfg=copy.deepcopy(test_cfg))
        self.denoise_model = build_head(denoise_model)
        down_chs = self.denoise_model.down_chs[:-1]
        self.num_classes = self.denoise_model.num_classes
        self.cond_blocks = nn.ModuleList()
        for level, ch in enumerate(down_chs):
            self.cond_blocks.append(
                nn.Conv2d(256*(2**level), ch, 1, padding=0)
            )
        self.loss = build_loss(loss_cfg)
        # self.loss_dice = build_loss(loss_dice)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._diffusion_init(**diffusion)

    def _diffusion_init(
        self,
        steps,
        schedule_name,
        model_mean_type,
        model_var_type,
        loss_type,
        resample_steps=None,
        num_bits=7,
        use_ddim=False,
        rescale_timesteps=False
    ):
        self.use_ddim = use_ddim
        self.num_bits = num_bits
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        ori_betas = np.array(get_named_beta_schedule(schedule_name, steps), dtype=np.float64)
        if resample_steps:
            skip_step = steps // resample_steps 
            ori_alphas = 1.0 - ori_betas
            ori_alphas_cumprod = np.cumprod(ori_alphas, axis=0)
            last_alpha_cumprod = 1.0
            betas = []
            for i, alpha_cumprod in enumerate(ori_alphas_cumprod):
                if i % skip_step == 0:
                    betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
            betas = np.array(betas, dtype=np.float64)
        else:
            betas = ori_betas
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def decimal_to_bits(self, x):
        """ expects tensor x ranging from 0 to 100(int), outputs bit tensor ranging from -1 to 1 """
        mask = 2 ** torch.arange(self.num_bits - 1, -1, -1, dtype=torch.int32, device=x.device)
        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b c h w -> b c 1 h w')

        bits = ((x & mask) != 0).float()
        bits = rearrange(bits, 'b c d h w -> b (c d) h w')
        bits = bits * 2 - 1
        return bits

    def bits_to_decimal(self, x):
        """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
        device = x.device

        x = (x > 0).int()
        mask = 2 ** torch.arange(self.num_bits - 1, -1, -1, device = device, dtype = torch.int32)
        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b (c d) h w -> b c d h w', d = self.num_bits)
        dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
        return dec
    
    def _get_refine_results(self, samples, bboxes, img_metas, iou_thr=0.2):
        # root = 'results/'
        batch_size = len(img_metas)
        bbox_results = []
        mask_results = []
        for batch_id in range(batch_size):
            refine_bboxes = [[] for _ in range(self.num_classes)]
            refine_masks = [[] for _ in range(self.num_classes)]
            img_shape = img_metas[batch_id]['img_shape']
            ori_shape = img_metas[batch_id]['ori_shape']
            masks = samples[batch_id].unsqueeze(0)
            masks = masks[:, :, 0:img_shape[0], 0:img_shape[1]]
            masks = F.interpolate(masks, size=ori_shape[:-1], mode='bilinear', align_corners=True)
            masks = (masks >= 0).int
            img_bbox = bboxes[batch_id]
            for mask in masks:
                mask = mask.cpu().numpy()
                ious = [cal_iou(mask, _) for _ in img_bbox]
                max_iou = max(ious)
                if max_iou < iou_thr:
                    continue

                # gt_mask_save(mask, root)

                idx = ious.index(max_iou)
                bbox = img_bbox[idx, 0:5]
                label = int(img_bbox[idx, 5])
                refine_masks[label].append(mask.astype(bool))
                refine_bboxes[label].append(bbox)
            bbox_results.append(refine_bboxes)
            mask_results.append(refine_masks)
        return list(zip(bbox_results, mask_results))
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        img_feats = []
        # img_feats_shape = []
        for i, module in enumerate(self.cond_blocks):
            feat = module(x[i])
            img_feats.append(feat)
            # img_feats_shape.append(feat.shape[-2:])
        return img_feats
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      coarse_masks,
                      labels,
                      **kwargs):

        current_device = img.device
        num_per_batch = [len(mask) for mask in gt_masks]
        num = 0
        idx2batch = {}
        for idx, len_batch in enumerate(num_per_batch):
            for _ in range(len_batch):
                idx2batch[num] = idx
                num += 1
        x_start = torch.cat(gt_masks, dim=0).to(current_device)
        x_start = x_start.unsqueeze(1)
        coarse_masks = torch.cat(coarse_masks, dim=0).to(current_device)
        coarse_masks = coarse_masks.unsqueeze(1)
        labels = torch.cat(labels).to(current_device)
    
        # get img_features
        img_feats = self.extract_feat(img) #(1/4, 1/8)

        # get x_t
        noise = torch.randn_like(x_start)
        t = uniform_sampler(self.num_timesteps, x_start.shape[0], img.device)
        x_t = self.q_sample(x_start, t, noise=noise)

        #cal diffusion loss
        losses = dict()
        model_output = self.denoise_model(x_t, self._scale_timesteps(t), coarse_masks, img_feats, labels, idx2batch)

        if self.model_mean_type == 'prev_x':
            target = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0]
        elif self.model_mean_type == 'start_x':
            target = x_start
        elif self.model_mean_type == 'epsilon':
            target = noise
        else:
            raise NotImplementedError(f'unsupported mean type {self.model_mean_type}')

        # model_out = model_output.clone().detach() >= 0
        # target = target > 0
        # si = (model_out & target).sum()
        # su = (model_out | target).sum()
        
        fg_mask = target >= 0.5
        bg_mask = target < 0.5
        fg_weight = bg_mask.sum() / fg_mask.sum()
        pixel_weight = fg_mask.float() * fg_weight + bg_mask.float()
        # fg_num, bg_num = fg_mask.sum(), bg_mask.sum()
        # eps = torch.finfo(torch.float32).eps
        losses["model_mean_loss"] = self.loss(model_output, target, weight=pixel_weight)
        # loss =  F.mse_loss(model_output, target, reduction='none')
        
        # losses['mse_loss'] = (loss * pixel_weight).mean()
        # losses['dice_loss'] = self.loss_dice(model_output, target)
        # losses['train_iou'] = si / su

        return losses
    
    def simple_test(self,
                    img,
                    img_metas,
                    coarse_masks,
                    bboxes,
                    max_batch=50,
                    rescale=False):
        current_device = img.device
        batch_size = len(img)
        num_per_batch = [len(mask) for mask in coarse_masks]
        idx2batch = {}
        num = 0
        for idx, len_batch in enumerate(num_per_batch):
            for _ in range(len_batch):
                idx2batch[num] = idx
                num += 1
        img_feats, img_feats_shape = self.extract_feat(img)
        coarse_masks = torch.cat(coarse_masks, dim=0).to(current_device)
        coarse_masks = coarse_masks.unsqueeze(1)
        mask_batch = len(coarse_masks)
        if mask_batch <= max_batch:
            cond = dict(img_feats=img_feats, coarse_masks=coarse_masks, idx2batch=idx2batch)
            sample_shape = coarse_masks.shape
            samples = self.p_sample_loop (self.denoise_model, sample_shape, cond)
        else:
            samples = []
            if (mask_batch / max_batch - mask_batch // max_batch) > 0:
                batch_divide = mask_batch // max_batch + 1
            else:
                batch_divide = mask_batch // max_batch
            for i in range(batch_divide):
                coarse_mask = coarse_masks[i*max_batch: min(mask_batch, (i+1)*max_batch)]
                cur_batch = len(coarse_mask)
                cur_idx2batch = {}
                for j in range(cur_batch):
                    cur_idx2batch[j] = idx2batch[j + i*max_batch]
                cond = dict(img_feats=img_feats, coarse_masks=coarse_mask, idx2batch=cur_idx2batch)
                sample_shape = coarse_mask.shape
                sample = self.p_sample_loop (self.denoise_model, sample_shape, cond)
                samples.append(sample)
            samples = torch.cat(samples, dim=0)

        bbox_results = []
        mask_results = []
        batch_start = 0
        img_ids = []
        for batch_id, batch_bbox in enumerate(bboxes):
            num_cur_batch = len(batch_bbox)
            refine_bboxes = [[] for _ in range(self.num_classes)]
            refine_masks = [[] for _ in range(self.num_classes)]
            img_shape = img_metas[batch_id]['img_shape']
            ori_shape = img_metas[batch_id]['ori_shape']
            img_id = img_metas[batch_id]['ori_filename']
            img_id = img_id.split('.')[0][-12:]
            img_ids.append(int(img_id))
            h_np, w_np = img_shape[0] // 2, img_shape[1] // 2
            masks = samples[batch_start: batch_start + num_cur_batch]
            # print(masks.shape)
            masks = masks[:, :, 0:h_np, 0:w_np]
            # print(masks.shape)
            masks = F.interpolate(masks, size=ori_shape[:-1], mode='bilinear', align_corners=True)
            masks = masks.squeeze(1)
            masks = (masks >= 0)
            masks = masks.cpu().numpy()
            for idx, bbox in enumerate(batch_bbox):
                new_bbox = bbox[0:5]
                label = int(bbox[5])
                mask = masks[idx]
                refine_masks[label].append(mask)
                refine_bboxes[label].append(new_bbox)
            bbox_results.append(refine_bboxes)
            mask_results.append(refine_masks)
            batch_start += num_cur_batch
        return list(zip(bbox_results, mask_results)), img_ids

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, cond, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, cond, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def aug_test(self, img, img_metas, rescale=False):
        pass

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, cond, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **cond)

        if self.model_var_type in ['learned', 'learned_range']:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == 'learned':
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                'fixed_large': (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                'fixed_small': (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == 'prev_x':
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in ['start_x', 'epsilon']:
            if self.model_mean_type == 'start_x':
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, cond, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        cond,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            cond,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        cond,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]
