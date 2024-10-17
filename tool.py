from collections import defaultdict
import os
from tqdm import tqdm
import json
import shutil
import pandas as pd

# convert iphone detection annotation classes
# 1 -> 3, 2 -> 1, 3 -> 2
# label_path = r'/data/vjuicefs_sz_ocr_001/72179367/Training_data/glip_test_apple/yolo_format/test/labels'
# for label in tqdm(os.listdir(label_path)):
#     line_list = []
#     with open(os.path.join(label_path,  label), 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             new_line = line
#             if line[0] == '1':
#                 new_line = '3' + line[1:]
#             elif line[0] == '2':
#                 new_line = '1' + line[1:]
#             elif line[0] == '3':
#                 new_line = '2' + line[1:]
#             line_list.append(new_line)
#     with open(os.path.join(label_path, label), 'w') as file:
#         file.writelines(line_list)


# convert prediction to coco instances_val.json format, that is class 0-80 -> 1-90
map = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
}
new_json = []
with open('runs/detect/yolov8n_coco_100epoch_remax=13_(dfl+false_loss)/predictions.json', 'r') as file:
    pred = json.load(file)
    for item in tqdm(pred):
        category = item['category_id']
        new_cat = map[category]
        item['category_id'] = new_cat
        new_json.append(item)

with open('runs/detect/yolov8n_coco_100epoch_remax=13_(dfl+false_loss)/predictions_converted.json', 'w') as json_file:
    json.dump(new_json, json_file, indent=2)
