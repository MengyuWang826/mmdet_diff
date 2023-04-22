import json
import pycocotools.mask as maskUtils

coarse_dts = json.load(open('data/lvis_annotations/maskrcnn_lvis_val_cocofied.json'))
refine_dts = json.load(open('results.segm.json'))

for idx, coarse_dt in enumerate(coarse_dts):
    coarse_mask = maskUtils.decode(coarse_dt['segmentation'])
    area = coarse_mask.sum()
    if area < 512:
        refine_dts.append(coarse_dt)

with open('refine_paste_tiny.json', 'w') as f:
    json.dump(refine_dts, f)


a = 1