import json
import pycocotools.mask as maskUtils
from tqdm import tqdm

coarse_dts = json.load(open('all_json/coarse_json_cocofiedlvis/transfiner_r50_3x_deform.json'))
refine_dts = json.load(open('results.segm.json'))

with tqdm(total=len(coarse_dts)) as p:
    for idx, coarse_dt in enumerate(coarse_dts):
        coarse_mask = maskUtils.decode(coarse_dt['segmentation'])
        area = coarse_mask.sum()
        if area < 512: 
            refine_dts.append(coarse_dt)
        p.update()

with open('refine_paste_tiny.json', 'w') as f:
    json.dump(refine_dts, f)


a = 1