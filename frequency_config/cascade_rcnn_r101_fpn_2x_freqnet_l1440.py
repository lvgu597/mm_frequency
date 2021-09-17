_base_ = '../frequency_config/cascade_rcnn_r50_fpn_40e_l1440.py'
model = dict(
    type='CascadeRCNN',
    # pretrained='modelzoo://resnet50',
    backbone=dict(
        type='FreqNet',
        depth=101,
        num_stages=4,
        reduction=16,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(
            type='BN', #设置bn层
            requires_grad=True),
        norm_eval=True,
        style='pytorch'),)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[32, 38])
total_epochs = 40
checkpoint_config = dict(interval=5)
work_dir = '../data/work_dirs/cascade_rcnn_r101_fpn_2x_freqnet_l1440'
load_from = '../data/pretrained/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth' #加载预训练网络参数