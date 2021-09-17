_base_ = './cascade_rcnn_r50_fpn_70e_733.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 4),
            stages=(False, True, True, True),
            position='after_conv3')
    ]))
work_dir = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_gcnet'
load_from = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_733/epoch_41.pth' #加载预训练网络参数