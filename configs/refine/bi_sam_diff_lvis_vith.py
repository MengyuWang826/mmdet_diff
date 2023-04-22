_base_ = [
    '../_base_/default_runtime.py'
]

image_size = 1024
prompt_embed_dim = 256
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size

model = dict(
    type='SamRefinementor',
    image_encoder=dict(
        type='SamImageEncoder',
        depth=32,
        embed_dim=1280,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=dict(eps=1e-6),
        num_heads=16,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[7, 15, 23, 31],
        window_size=14,
        out_chans=256),
    prompt_encoder=dict(
        type='SamPromptEncoder',
        embed_dim=prompt_embed_dim,
        input_image_size=(image_size, image_size),
        image_embedding_size=(image_embedding_size, image_embedding_size),
        mask_in_chans=16,
        sam_zero_shot=True),
    mask_decoder=dict(
        type='SamMaskDecoder',
        transformer=dict(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        multi_mask_output=True),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,  # 1e-4 gauss, 0.02 uniform
            stop=0,  # 0.02, gauss, 1. uniform
            num_timesteps=6),
        diff_iter=False),
    # model training and testing settings
    train_cfg=dict(
        pad_width=20),
    test_cfg=dict(
        pad_width=20))  

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True),
    dict(type='LoadCoarseMasks'),
    dict(type='Resize', img_scale=(image_size, image_size), mask_scale_factor=0.25, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(image_size, image_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', with_bbox=True, test_mode=True),
    dict(type='Resize', img_scale=(image_size, image_size), mask_scale_factor=0.25, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(image_size, image_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks', 'dt_bboxes'])
]


dataset_type = 'LVISRefine'
img_root = 'data/coco/'
ann_root = 'data/lvis_annotations/'
train_dataloader=dict(
    samples_per_gpu=4,
    workers_per_gpu=4)
test_dataloader=dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
data = dict(
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'lvis_v1_train.json',
        coarse_file=ann_root + 'maskrcnn_lvis_train_matched.json',
        img_prefix=img_root),
    train_dataloader=train_dataloader,
    val=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'lvis_v1_val_cocofied.json',
        coarse_file=ann_root + 'maskrcnn_lvis_val_cocofied.json',
        img_prefix=img_root),
    val_dataloader=test_dataloader,
    test=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'lvis_v1_val_cocofied.json',
        coarse_file=ann_root + 'maskrcnn_lvis_val_cocofied.json',
        img_prefix=img_root),
    test_dataloader=test_dataloader)
evaluation = dict(metric=['bbox', 'segm'])


optimizer = dict(
    type='AdamW',
    lr=1e-6,
    weight_decay=0,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'prompt_encoder': dict(lr_mult=10)
        }))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    step=[8, 11],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=0.01,  # no warmup
    warmup_iters=150)


runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook', by_epoch=False)
    ])

workflow = [('train', 1)]
checkpoint_config = dict(interval=1)

evaluation = dict(
    interval=1,
    metric=['bbox', 'segm'])

load_from = 'pretrain/sam_pre_bbox_0.pth'