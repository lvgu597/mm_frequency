_base_ = './cascade_rcnn_r50_fpn_70e_733.py'
model = dict(
    backbone = dict(
        norm_cfg=dict(
            type='BN', #设置bn层
            requires_grad=True),
        norm_eval=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), #dcnv2 c3-c5
        stage_with_dcn=(False, True, True, True),
    )
)
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_dcnv2'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_733/epoch_41.pth'