import os
import base64
import time
import requests
import json

class_dict = {0: 'apple logo', 1: 'apple home', 2: 'iphone front camera', 3: 'iphone back camera'}
imgfile = '/data/vjuicefs_sz_ocr_001/72179367/codes/ultralytics-main/pics/20230223_iphone_3_.jpg'

params = {
     'image': imgfile,
     'sessid':'test',
     'businessId':'thisistest',
}

url = 'http://0.0.0.0:5000/competitor_apple'

a = time.time()
res = requests.post(url, data=params)

out = json.loads(res.text)
out['cls'] = [class_dict[i] for i in out['cls']] 

print("using time:",time.time()-a)
print(out)