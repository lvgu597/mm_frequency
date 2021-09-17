from mmdet.apis import init_detector, inference_detector
import mmcv
config_file = '../data/work_dirs/cascade_rcnn_r50_detectors_70e/cascade_rcnn_r50_detectors_70e.py'
checkpoint_file = '../data/work_dirs/cascade_rcnn_r50_detectors_70e/epoch_30.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
model.show_result(img, result, out_file='result.jpg')