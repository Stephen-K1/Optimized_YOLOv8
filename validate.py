import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml').load('weights/yolov8n.pt')
model = YOLO('runs/detect/yolov8s_coco_100epoch_AP58/weights/best.pt')
# model = YOLO('runs/detect/yolov8m_no_pretrained_20k_without_vivo/weights/best.pt')

# Customize validation settings
metrics = model.val(data='coco.yaml', imgsz=640, batch=128, device='0', workers=8, save_json=True)
metrics.box.map    # map50-95
metrics.box.all_ap
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category  