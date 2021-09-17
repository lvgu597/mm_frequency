# python test.py ../data/work_dirs/faster_rcnn_r50_fpn_2x/faster_rcnn_r50_fpn_2x.py \
#     ../data/work_dirs/faster_rcnn_r50_fpn_2x/latest.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_fpn_2x/cascade_rcnn_r50_fpn_2x.py \
#     ../data/work_dirs/cascade_rcnn_r50_fpn_2x/latest.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_detectors_70e/cascade_rcnn_r50_detectors_70e.py \
#     ../data/work_dirs/cascade_rcnn_r50_detectors_70e/epoch_30.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_detectors_dcn/cascade_rcnn_r50_detectors_dcn.py \
#     ../data/work_dirs/cascade_rcnn_r50_detectors_dcn/epoch_23.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_detectors_dcnv2/cascade_rcnn_r50_detectors_dcnv2.py \
#     ../data/work_dirs/cascade_rcnn_r50_detectors_dcnv2/latest.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_detectors_70e_gc/cascade_rcnn_r50_detectors_70e_gc.py \
#     ../data/work_dirs/cascade_rcnn_r50_detectors_70e_gc/epoch_26.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_detectors_2x_oversample/cascade_rcnn_r50_detectors_2x_oversample.py \
#     ../data/work_dirs/cascade_rcnn_r50_detectors_2x_oversample/epoch_1.pth \
#     --eval bbox

# python test.py ../data/work_dirs/cascade_rcnn_r50_1x_DetectoRS/cascade_rcnn_r50_1x_DetectoRS.py \
#     ../data/work_dirs/cascade_rcnn_r50_1x_DetectoRS/latest.pth \
#     --format-only \
#     --eval-options "jsonfile_prefix=../data/results/DtetctoRS_b_results"

# python json2submit.py \
#     --test_json ../data/results/DtetctoRS_b_results.bbox.json

# python test.py ../data/work_dirs/cascade_rcnn_r50_fpn_70e_733/cascade_rcnn_r50_fpn_70e_733.py \
#     ../data/work_dirs/cascade_rcnn_r50_fpn_70e_733/latest.pth \
#     --out ../data/results/base_result.pkl

# python test.py ../data/work_dirs/cascade_rcnn_r50_fpn_70e_dcn/cascade_rcnn_r50_fpn_70e_dcn.py \
#     ../data/work_dirs/cascade_rcnn_r50_fpn_70e_dcn/latest.pth \
#     --eval bbox

python test.py ../data/work_dirs/faster_rcnn_r50_fpn_2x_freqnet_l1440_crop/faster_rcnn_r50_fpn_2x_freqnet_l1440_crop.py \
    ../data/work_dirs/faster_rcnn_r50_fpn_2x_freqnet_l1440_crop/latest.pth \
    --eval bbox

python test.py ../data/work_dirs/cascade_rcnn_r50_fpn_2x_freqnet_l1440_crop/cascade_rcnn_r50_fpn_2x_freqnet_l1440_crop.py \
    ../data/work_dirs/cascade_rcnn_r50_fpn_2x_freqnet_l1440_crop/latest.pth \
    --eval bbox