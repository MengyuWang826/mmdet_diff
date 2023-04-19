checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrain/sam_pre_bbox_0.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
image_size = 1024
prompt_embed_dim = 256
vit_patch_size = 16
image_embedding_size = 64
model = dict(
    type='SamRefinementor',
    image_encoder=dict(
        type='SamImageEncoder',
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=dict(eps=1e-06),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256),
    prompt_encoder=dict(
        type='SamPromptEncoder',
        embed_dim=256,
        input_image_size=(1024, 1024),
        image_embedding_size=(64, 64),
        mask_in_chans=16),
    mask_decoder=dict(
        type='SamMaskDecoder',
        transformer=dict(
            depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256),
    diffusion_cfg=dict(
        betas=dict(type='linear', start=0.8, stop=0, num_timesteps=6),
        diff_iter=True),
    train_cfg=dict(pad_width=20),
    test_cfg=dict(pad_width=20))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_mask=True),
    dict(type='LoadCoarseMasks'),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        mask_scale_factor=0.25,
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', with_bbox=True, test_mode=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        mask_scale_factor=0.25,
        keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks', 'dt_bboxes'])
]
dataset_type = 'LVISRefine'
img_root = 'data/coco/'
ann_root = 'data/lvis_annotations/'
train_dataloader = dict(samples_per_gpu=4, workers_per_gpu=4)
test_dataloader = dict(samples_per_gpu=1, workers_per_gpu=1)
data = dict(
    train=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=False,
                with_label=False,
                with_mask=True),
            dict(type='LoadCoarseMasks'),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                mask_scale_factor=0.25,
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks'])
        ],
        type='LVISRefine',
        ann_file='data/lvis_annotations/lvis_v1_train.json',
        coarse_file='data/lvis_annotations/maskrcnn_lvis_train_matched.json',
        img_prefix='data/coco/'),
    train_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
    val=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadCoarseMasks', with_bbox=True, test_mode=True),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                mask_scale_factor=0.25,
                keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'coarse_masks', 'dt_bboxes'])
        ],
        type='LVISRefine',
        ann_file='data/lvis_annotations/lvis_v1_val_cocofied.json',
        coarse_file='data/lvis_annotations/maskrcnn_lvis_val_cocofied.json',
        img_prefix='data/coco/'),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    test=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadCoarseMasks', with_bbox=True, test_mode=True),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                mask_scale_factor=0.25,
                keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'coarse_masks', 'dt_bboxes'])
        ],
        type='LVISRefine',
        ann_file='data/lvis_annotations/lvis_v1_val_cocofied.json',
        coarse_file='data/lvis_annotations/maskrcnn_lvis_val_cocofied.json',
        img_prefix='data/coco/'),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
optimizer = dict(
    type='AdamW', lr=1e-05, weight_decay=0, eps=1e-08, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    gamma=0.1,
    step=[8, 11],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=0.01,
    warmup_iters=150)
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = './work_dirs/bi_sam_diff_lvis'
auto_resume = False
gpu_ids = range(0, 4)
