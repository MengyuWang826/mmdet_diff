_base_ = [
    '../_base_/default_runtime.py'
]

model = dict(
    type='DiffusionRefine',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    denoise_model=dict(
        type='DenoiseUNetHead',
        in_channels=2,
        out_channels=1,
        model_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        channel_mult = (1, 1, 2, 2, 4, 4),
        dropout=0.0,
        num_classes=80,
        use_checkpoint=False),
    diffusion=dict(
        steps=1000,
        schedule_name='cosine',
        model_mean_type='start_x',
        model_var_type='fixed_small',
        loss_type='mse',
        resample_steps=250,
        rescale_timesteps=False),
    loss_cfg=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True),
    # loss_dice=dict(
    #     type='DiceLoss',
    #     use_sigmoid=True,
    #     activate=True,
    #     reduction='mean',
    #     naive_dice=True,
    #     eps=1.0),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())  

image_size = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True),
    dict(type='LoadCoarseMasks'),
    dict(type='Resize', img_scale=image_size, mask_scale_factor=0.5, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='PreforDiffRefine', gt_norm=False, max_out=8),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks', 'coarse_masks', 'labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', with_bbox=True),
    dict(type='Resize', img_scale=image_size, mask_scale_factor=0.5, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='PreforDiffRefine', mode='test'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks', 'bboxes'])
]

dataset_type = 'CocoRefine'
img_root = 'data/coco/'
ann_root = 'data/annotations/'
train_dataloader=dict(
    samples_per_gpu=4,
    workers_per_gpu=2)
test_dataloader=dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
data = dict(
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'instances_train2017.json',
        coarse_file=ann_root + 'maskrcnn_coco_train.json',
        img_prefix=img_root + 'train2017/'),
    train_dataloader=train_dataloader,
    val=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'instances_val2017.json',
        coarse_file=ann_root + 'maskrcnn_coco_val.json',
        img_prefix=img_root + 'val2017/'),
    val_dataloader=test_dataloader,
    test=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=ann_root + 'instances_val2017.json',
        coarse_file=ann_root + 'maskrcnn_coco_val.json',
        img_prefix=img_root + 'val2017/'),
    test_dataloader=test_dataloader)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

max_iters = 400000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
interval = 5000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=10)

evaluation = dict(
    interval=interval,
    metric=['bbox', 'segm'])

# resume_from = 'checkpoints/iter_10000.pth'