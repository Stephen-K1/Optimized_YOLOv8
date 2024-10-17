import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
from tqdm import tqdm
import time


path = '/data/vjuicefs_sz_ocr_001/72179367/DSP/to_zip/10k_0910/images'
txt_path = '/data/vjuicefs_sz_ocr_001/72179367/DSP/to_zip/10k_0910/labels'
model = YOLO('runs/detect/yolov8s_3Classes_3100Images/weights/best.pt')
time_0 = time.time()
for img in tqdm(os.listdir(path)):
    # if not img.endswith('.jpg'):
    #     print(f'{img} does not end with jpg')
    #     continue
    txt_lines = []
    img_path = os.path.join(path, img)
    result = model.cuda()(img_path)
    cls, boxes = result[0].boxes.cls.to('cpu'), result[0].boxes.xywhn.to('cpu')
    cls_list = [int(i) for i in list(cls)]
    boxes_list = [[str(element) for element in row] for row in boxes.tolist()]
    for i, j in zip(cls_list, boxes_list):
        txt_lines.append(str(i) + ' ' + ' '.join(j) + '\n')
    with open(os.path.join(txt_path, img.replace('.jpg', '.txt')), 'w') as txt_file:
        txt_file.writelines(txt_lines)
time_1 = time.time()
print(time_1-time_0)