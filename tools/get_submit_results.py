import time, os
import json
import mmcv 
from mmdet.apis import init_detector, inference_detector

def main():

    config_file = '../config/cascade_rcnn_r50_fpn_70e_duck_aug.py'  # 修改成自己的配置文件
    checkpoint_file = '../data/work_dirs/cascade_rcnn_r50_fpn_70e_duck_aug/latest.pth' # 修改成自己的训练权重

    test_path = '/home/jkx/project/smallq/tianchidata/normal_aug/'  # 官方测试集图片路径

    csv_name = "result_"+""+time.strftime("%Y%m%d%H%M%S", time.localtime())+".csv"
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    csv_file = open(csv_name, 'w')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    for i, img_name in enumerate(img_list, 1):
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        for i, bboxes in enumerate(predict, 1):
            print(i, bboxes)
            assert True==0
            if len(bboxes)>0:
                defect_label = i
                print(i)
                image_name = img_name
                for bbox in bboxes:
                    xmin, ymin, w, h, confidence = bbox.tolist()
                    xmax = xmin + w
                    ymax = ymin + h
                    xmin = round(x1,2)
                    ymin = round(y1,2)
                    csv_file.write(defect_label + ',' + image_name + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
    csv_file.close()
        
if __name__ == "__main__":
    main()