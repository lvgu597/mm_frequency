_base_ = './cascade_rcnn_r50_fpn_70e_733.py'
# dataset settings
dataset_type = 'FabricDataset'
data_root = '/home/jkx/project/smallq/tianchidata_coco_base/'  # Root path of data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(type='LoadAnnotations', with_bbox=True), # Second pipeline to load annotations for current image
    dict(
        type='Resize', # Augmentation pipeline that resize the images and their annotations
        img_scale=[(3400, 800), (3400, 1200)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5), #翻转 flip_ratio 为翻转概率
    dict(type='Normalize', **img_norm_cfg), #规范化image
    dict(type='Pad', size_divisor=32), # padding设置，填充图片可被32整出除
    dict(type='DefaultFormatBundle'), # Default format bundle to gather data in the pipeline
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), #决定将哪些关键数据传给detection的管道
]
test_pipeline = [
    dict(type='LoadImageFromFile'), #加载图片的pipline
    dict(
        type='MultiScaleFlipAug', 
        img_scale=(3400, 1000), #最大test scale
        flip=True, #测试时是否翻转图片
        transforms=[
            dict(type='Resize', keep_ratio=True), #保持原始比例的resize
            dict(type='RandomFlip'), #
            dict(type='Normalize', **img_norm_cfg), #规范化
            dict(type='Pad', size_divisor=32), 
            dict(type='ImageToTensor', keys=['img']), #将图片转为tensor
            dict(type='Collect', keys=['img']), #获取关键信息的pipline
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/Duck_inject_normal_2.json',
        img_prefix=data_root + 'defect_images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val0331.json',
        img_prefix=data_root + 'defect_images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val0331.json',
        img_prefix=data_root + 'defect_images/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[40, 60])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# runtime settings
total_epochs = 70
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_with_duck_2.py'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_733/latest.pth' #加载预训练网络参数
resume_from = None
workflow = [('train', 1)]