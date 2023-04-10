import json
from tqdm import tqdm
import copy

# train = json.load(open('data/LVIS/lvis_v1_train.json'))
coarse = json.load(open('data/refine_annotations/maskrcnn_val.json'))
# val = json.load(open('data/annotations/instances_val2017.json'))
# coarse = json.load(open('data/annotations/maskrcnn_coco_val.json'))
all_imgs = set([397133, 37777, 252219, 174482])
# a = 1

new_coarse = []
# new_val = []

for dt in coarse:
    if dt['image_id'] in all_imgs:
        new_coarse.append(dt)
# for gt in val['annotations']:
#     gt['score'] = 0.9

a = 1
# with open('data/annotations/coarse_part.json', 'w') as f:
#     json.dump(new_coarse, f)
with open('coarse_part.json', 'w') as f:
    json.dump(new_coarse, f)