import json
import pycocotools.mask as maskUtils
from tqdm import tqdm

coarse_dts = json.load(open('all_json/coarse_json_cocofiedlvis/mask2former_r50_50e.json'))
refine_dts = json.load(open('results.segm.json'))


new_refine_dts = []
with tqdm(total=len(refine_dts)) as p:
    for idx, refine_dt in enumerate(refine_dts):
        coarse_mask = maskUtils.decode(refine_dt['segmentation'])
        area = coarse_mask.sum()
        if area > 1024: 
            new_refine_dts.append(refine_dt)
        p.update()

with tqdm(total=len(coarse_dts)) as p:
    for idx, coarse_dt in enumerate(coarse_dts):
        coarse_mask = maskUtils.decode(coarse_dt['segmentation'])
        area = coarse_mask.sum()
        if area <= 1024: 
            new_refine_dts.append(coarse_dt)
        p.update()

with open('refine_paste_tiny.json', 'w') as f:
    json.dump(new_refine_dts, f)


a = 1