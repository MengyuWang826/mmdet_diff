from lvis import LVIS, LVISEval, LVISResults
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json


def val_lvis(result_path, ann_path, val_type):
    lvis_eval = LVISEval(ann_path, result_path, val_type)
    lvis_eval.run()
    lvis_eval.print_results()

def val_coco(result_path, ann_path, val_type):
    cocoGt = COCO(ann_path)        #标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes(result_path)  #自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, val_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ =='__main__':
    gt_json = 'data/lvis_annotations/lvis_v1_val_cocofied.json'
    dt_json = 'data/lvis_annotations/maskrcnn_lvis_val_cocofied.json'
    dataset = 'lvis'
    if dataset == 'coco':
        a = val_coco(dt_json, gt_json, 'segm')
    else:
        a = val_lvis(dt_json, gt_json, 'segm')
