import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
from tqdm import tqdm
import time
from PIL import Image
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
import threading

"""speed test"""
# model = YOLO('yolov8n.yaml')
# # model = RTDETR('rtdetr-l.yaml')
# # model = YOLO('yolov8l.yaml')
# model.model.cuda()
# data = torch.randn(8,3,640,640).cuda()
# time_0 = time.time()
# for i in tqdm(range(1000)):
#     out = model(data)
#     # result = model.cuda()(r'ultralytics/assets/bus.jpg')
# time_1 = time.time()
# print(time_1-time_0)


"""test detection result, save images"""
# model = YOLO('runs/detect/yolov8l_20k_without_vivo/weights/best.pt')
# images_path = '/data/vjuicefs_sz_ocr_001/72179367/DSP/download/10k'
# save_path = 'predicted_results/yolov8l_20k_without_vivo'
# # device = torch.device('cuda:3')   # 'cuda:3'中间不能有空格
# for image_name in tqdm(os.listdir(images_path)):
#     image = os.path.join(images_path, image_name)
#     result = model.cuda()(image)
#     im_bgr = result[0].plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
#     im_rgb.save(os.path.join(save_path, image_name))


"""predict the background images and save the result, mark those false postive images in fp list"""
fp_list = []
model = YOLO('runs/detect/yolov8s_3Classes_3100Images/weights/best.pt')
images_path = '/data/vjuicefs_sz_ocr_001/72179367/DSP/download/300k'
save_path = 'predicted_results/20k_0910'
# device = torch.device('cuda:3')   # 'cuda:3'中间不能有空格
images = os.listdir(images_path)
images.sort()
for image_name in tqdm(images[80000:100000]):   # 300k数据，目前到80k
    image = os.path.join(images_path, image_name)
    result = model.cuda()(image)
    im_bgr = result[0].plot()     # BGR-order numpy array
    conf = result[0].boxes.conf.tolist()
    if len(conf) > 0:
        fp_list.append(result[0].path.split('/')[-1].split('.')[0])   # save the name of the image
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        im_rgb.save(os.path.join(save_path, image_name))
print(fp_list, '\n', f'length of the list: {len(fp_list)}')   # save to logs


"""save the positve and negative preditions"""
# tp_list = []
# fp_list = []
# model = YOLO('/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/runs/detect/yolov8l_all_100epoch/weights/best.pt')
# images_path = r'/data/vjuicefs_sz_ocr_001/72179367/Training_data/Apple_20k/val_only_apple/images'
# tp_path = r'/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/predicted_results/18k_apple/TP'
# fn_path = r'/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/predicted_results/18k_apple/FP'
# # device = torch.device('cuda:3')   # 'cuda:3'中间不能有空格
# for image_name in tqdm(os.listdir(images_path)):
#     image = os.path.join(images_path, image_name)
#     result = model.cuda()(image)
#     im_bgr = result[0].plot()  # BGR-order numpy array
#     conf = result[0].boxes.conf.tolist()
#     if len(conf) > 0:
#         tp_list.append(result[0].path.split('/')[-1].split('.')[0])   # save the name of the image
#         im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
#         im_rgb.save(os.path.join(tp_path, image_name))
#     else:
#         fp_list.append(result[0].path.split('/')[-1].split('.')[0])   # save the name of the image
#         shutil.copy(result[0].path, fn_path)
# print(tp_list, '\n', f'length of the list: {len(tp_list)}')   # save to logs


"""save the negative preditions"""
# model = YOLO('/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/runs/detect/yolov8l_all_100epoch/weights/best.pt')
# images_path = r'/data/vjuicefs_sz_ocr_001/72179367/DSP/download/10k_testset'
# save_path = 'predicted_results/10k_testset'
# # device = torch.device('cuda:3')   # 'cuda:3'中间不能有空格
# for image_name in tqdm(os.listdir(images_path)):
#     image = os.path.join(images_path, image_name)
#     result = model.cuda()(image)
#     im_bgr = result[0].plot()  # BGR-order numpy array
#     conf = result[0].boxes.conf.tolist()
#     if len(conf) > 0:
#         im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
#         im_rgb.save(os.path.join(save_path, image_name))


"""visualize a YOLOv8 module (defined in 'ultralytics/nn/modules')"""
# from ultralytics.nn.modules import *
# import torch
# import os

# x = torch.ones(1, 128, 40, 40)
# m = Conv(128, 128)
# f = f'{m._get_name()}.onnx'
# torch.onnx.export(m, x, f)
# os.system(f'onnxsim {f} {f} && xdg-open {f}')


"""predict with multi-thread"""
# model = YOLO('runs/detect/yolov8s_3classes_180eopch_2000Images/weights/best.pt').cuda()
# images_path = '/data/vjuicefs_sz_ocr_001/72179367/DSP/download/300k'
# save_path = 'predicted_results/30k'
# os.makedirs(save_path, exist_ok=True)
# fp_list = []
# def process_image(image_name):
#     image = os.path.join(images_path, image_name)
#     result = model(image)
#     im_bgr = result[0].plot()  # BGR-order numpy array
#     conf = result[0].boxes.conf.tolist()
#     if len(conf) > 0:
#         fp_list.append(result[0].path.split('/')[-1].split('.')[0])  # Save the name of the image
#         im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
#         im_rgb.save(os.path.join(save_path, image_name))
#     return image_name

# with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your CPU/GPU capabilities
#     futures = {executor.submit(process_image, image_name): image_name for image_name in os.listdir(images_path)}
#     for future in tqdm(as_completed(futures), total=len(futures)):
#         image_name = futures[future]
#         try:
#             future.result()
#         except Exception as exc:
#             print(f'{image_name} generated an exception: {exc}')

# print(fp_list, '\n', f'Length of the list: {len(fp_list)}')



