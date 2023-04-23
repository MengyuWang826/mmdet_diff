checkpoint_config = dict(
    interval=5000, by_epoch=False, save_last=True, max_keep_ckpts=40)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 5000)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
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
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0.0,
        use_checkpoint=False),
    diffusion_cfg=dict(
        betas=dict(type='linear', start=0.8, stop=0, num_timesteps=6),
        diff_iter=False),
    train_cfg=dict(pad_width=20),
    test_cfg=dict(pad_width=20))
object_size = 256
patch_size = 128
test_scale = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_mask=True),
    dict(type='LoadCoarseMasksNew'),
    dict(type='LoadPatchData', object_size=256, patch_size=128),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'object_img', 'object_gt_masks', 'object_coarse_masks',
            'patch_img', 'patch_gt_masks', 'patch_coarse_masks'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasksNew', test_mode=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024), img_only=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks'])
]
dataset_type = 'CollectionRefine'
img_root = 'data/'
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
            dict(type='LoadCoarseMasksNew'),
            dict(type='LoadPatchData', object_size=256, patch_size=128),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'object_img', 'object_gt_masks', 'object_coarse_masks',
                    'patch_img', 'patch_gt_masks', 'patch_coarse_masks'
                ])
        ],
        type='CollectionRefine',
        ann_file='data/lvis_annotations/lvis_v1_train.json',
        coarse_file='data/lvis_annotations/maskrcnn_lvis_train_matched.json',
        collection_datasets=['dut', 'ecssd', 'fss', 'msra'],
        collection_json='data/collection.json',
        img_prefix='data/'),
    train_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
    val=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadCoarseMasksNew', test_mode=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024), img_only=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'coarse_masks'])
        ],
        type='CollectionRefine',
        ann_file='data/lvis_annotations/lvis_v1_val_cocofied.json',
        coarse_file='data/lvis_annotations/maskrcnn_lvis_val_cocofied.json',
        img_prefix='data/'),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
    test=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadCoarseMasksNew', test_mode=True),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024), img_only=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'coarse_masks'])
        ],
        type='CollectionRefine',
        ann_file='data/lvis_annotations/lvis_v1_val_cocofied.json',
        coarse_file='data/lvis_annotations/maskrcnn_lvis_val_cocofied.json',
        img_prefix='data/'),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1))
evaluation = dict(interval=5000, metric=['bbox', 'segm'])
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0, eps=1e-08, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[160000, 180000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,
    warmup_iters=10)
max_iters = 200000
runner = dict(type='IterBasedRunner', max_iters=200000)
interval = 5000
work_dir = './work_dirs/bi_diff_r50_multi_dataset'
auto_resume = False
gpu_ids = range(0, 8)
