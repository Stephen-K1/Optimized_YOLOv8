from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval

anno_json = r'/data/vjuicefs_sz_ocr_001/72179367/codes/Co-DETR-main/data/coco/annotations/instances_val2017.json'  # coco val2017 annotation
pred_json = r'runs/detect/yolov8n_coco_100epoch_remax=13_(dfl+false_loss)/predictions_converted.json'    # convert code in tool.py
anno = COCO(str(anno_json)) # Load your JSON annotations
pred = anno.loadRes(str(pred_json))   # Load predictions.json
val = COCOeval(anno, pred, "bbox")
val.evaluate()
val.accumulate()
val.summarize()