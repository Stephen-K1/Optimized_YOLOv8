import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8l.yaml').load('runs/detect/train22/weights/last.pt')  # build a new model from YAML
# model = YOLO('/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/runs/detect/train37/weights/last.pt')
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml', task='detect').load('weights/yolov8n.pt')
# model = YOLO('yolov8n.yaml').load('runs/detect/yolov8_coco_100epoch_2x_c2f/weights/best.pt')
# model = YOLO('runs/detect/train329/weights/last.pt')   # for resume
# model = YOLO('runs/detect/yolov8n_coco_100epoch_new_dfl_no_offset/weights/best.pt')

# model.ckpt['train_args']['device'] = '2,3'
# Train the model
results = model.train(data='coco.yaml', imgsz=640)
# results = model.train(resume=True, data='coco.yaml')