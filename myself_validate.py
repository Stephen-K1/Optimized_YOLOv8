'''
测试检测模型的cls分支输出准确程度
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml').load('weights/yolov8n.pt')
model = YOLO('runs/detect/yolov8n_coco_100epoch/weights/best.pt')

