# -*- coding: utf-8 -*-
import cv2
import os
import json
import random
import time
from ultralytics import YOLO
from flask import Flask, jsonify, request
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

app = Flask(__name__)

@app.route('/competitor_apple', methods=['POST'])    
def run():
    model = YOLO('/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/runs/detect/yolov8l_20k_without_vivo/weights/best.pt')
    start = time.time()
    sess = request.form.to_dict()
    # image = '/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/pics/20230223_iphone_3_.jpg'
    image = sess.get('image','')
    result = model(image)
    print(result[0].boxes.cls, result[0].boxes.conf)
    try:
        image = sess.get('image','')
        sessid = sess.get('sessid','')
        businessid = sess.get('businessId','')
        if not image: 
            print('null image')
            return jsonify({"result": {},"error_code": 2,"error_msg": "no parameter image"})
        else:
            print('request start,input no problem!')
            print(sum(param.numel() for param in model.parameters()))
            result = model(image)
            print(f'the cls in result: {result[0].boxex.cls}')
            res = {"result": result,"error_code": 0,"error_msg": "succ","sessionid": sessid}
            res = json.dumps(res)
            # print(res)
            stop = time.time()
            print('request finished, time cost [{}ms]'.format(stop-start))
            return res
    except:
        print('except error!')
        return jsonify({"result": {},"error_code": 1,"error_msg": "porno fail","sessionid": sessid})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5051)


# class model_session:
#     def __init__(self):
#         self.model = YOLO('/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/runs/detect/yolov8l_20k_without_vivo/weights/best.pt')


#     def __call__(self, img, businessid):
#         #convert base64 to numpy

#         img = self.img_b64_to_arr(img)
        
#        # image = Image.fromarray(img,mode='RGB')
#         img = main.preprocess_images(img,299,299)
        
#         # the format of result is like result={'level':'1','rate':'0.999','level_sexy':'0','rate_sexy':'0.0001'}
#         result = self.pornographic.pornographic_recognition(img, businessid)
#         print(result)

#         return result
#         #---------------------返回格式为json finished!---------------------#


# if __name__ == '__main__':
    
#     import time 
#     with open('temp1.jpg','rb') as f :
#         w = base64.b64encode(f.read())
#     #print w
#     vio = Poron_session()
#     a = time.time()
#     for x in range(1):
#         vio.recg(w,'thisistest')
#     b = time.time()
#     print(b-a)
#    # logger.info(b-a)
