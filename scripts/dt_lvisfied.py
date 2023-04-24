from lvis import LVIS
import json
from tqdm import tqdm

if __name__ == '__main__':
    gt_json = 'data/lvis_annotations/lvis_v1_val_cocofied.json'
    dt_json = 'all_json/coarse_json/mask2former_r50_50e.json'

    gt = LVIS(gt_json)
    dts = json.load(open(dt_json))

    new_dts = []

    with tqdm(total=len(dts)) as p:
        for dt in dts:
            if dt['image_id'] in gt.imgs and dt['category_id'] in gt.cats:
                new_dts.append(dt)
            p.update() 

    with open('all_json/coarse_json_cocofiedlvis/mask2former_r50_50e.json', 'w') as f:
        json.dump(new_dts, f)