_base_ = [
    '../_base_/default_runtime.py'
]

model = dict(
    type='DiffRefinementor',
    task='instance',
    denoise_model=dict(
        type='DenoiseUNet',
        in_channels=4,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        learn_time_embd=False,
        channel_mult = (1, 1, 2, 2, 4, 4),
        dropout=0.0,
        use_checkpoint=False),
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

object_size = 256
patch_size = 128
test_scale=(1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_mask=True),
    dict(type='LoadCoarseMasksNew'),
    dict(type='LoadPatchData', object_size=object_size, patch_size=patch_size),
    dict(type='Resize', img_scale=(object_size, object_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_coarse_masks',
                               'patch_img', 'patch_gt_masks', 'patch_coarse_masks'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasksNew', test_mode=True),
    dict(type='Resize', img_scale=test_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=test_scale, img_only=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks'])
]


dataset_type = 'CollectionRefine'
img_root = 'data/'
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
        collection_datasets=['dut', 'ecssd', 'fss', 'msra'],
        collection_json=img_root + 'collection.json',
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
    lr=1e-4,
    weight_decay=0,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.01),
            'neck': dict(lr_mult=0.01)
        }))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[8000, 9000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

max_iters = 10000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
interval = 1000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=40)

evaluation = dict(
    interval=interval,
    metric=['bbox', 'segm'])

