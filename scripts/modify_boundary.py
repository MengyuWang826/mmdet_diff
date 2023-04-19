import cv2
import numpy as np
import random
import math
from lvis import LVIS
import pycocotools.mask as maskUtils
from PIL import Image


def get_random_structure(size):
    # The provided model is trained with 
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target = 0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate. 
    # input: np array of size [H,W] image
    # output: same shape as input
    
    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #only modified contours is needed actually. 
    sampled_contours = []   
    modified_contours = [] 

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        #remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)
        
        idx_dist = []
        for i in range(number_of_vertices-number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
            
        idx_dist = sorted(idx_dist, key=lambda x:x[1])
        
        remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]
        
       #remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
        contour = new_contour
        

        #sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            #modify contours
            for idx, coor in enumerate(modified_contour):

                change = np.random.normal(0,move_rate) # 0.1 means change position of vertex to 10 percent farther from center
                x,y = coor[0]
                new_x = x + (x-center[0]) * change
                new_y = y + (y-center[1]) * change

                modified_contour[idx] = [new_x,new_y]
        modified_contours.append(modified_contour)
        
    #draw boundary
    gt = np.copy(image)
    image = np.zeros((image.shape[0], image.shape[1], 3))

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    image = image[:, :, 0]
    image = perturb_seg(image, iou_target)
    
    return image

def _poly2mask(mask_ann, img_h=None, img_w=None):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

if __name__ == '__main__':
    gt_json = 'data/lvis_annotations/lvis_v1_val.json'
    data_root = 'data/coco/'
    gt = LVIS(gt_json)
    all_imgs = gt.imgs
    for img_id in all_imgs:
        img_info = gt.load_imgs([img_id])[0]
        filename = data_root + img_info['coco_url'].replace('http://images.cocodataset.org/', '')
        image = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
        h, w = img_info['height'], img_info['width']
        ann_ids = gt.get_ann_ids(img_ids=[img_id])
        ann_info = gt.load_anns(ann_ids)
        for ins_id, ann in enumerate(ann_info):
            cat_id = ann['category_id']
            gt_mask = _poly2mask(ann['segmentation'], img_h=h, img_w=w)
            if gt_mask.sum() < 1024:
                continue
            gt_mask = (gt_mask * 255).astype(np.uint8)
            a = modify_boundary(gt_mask)
            a = a.astype(np.uint8)
            Image.fromarray(gt_mask).save(f'results/{ins_id}_gt.png')
            Image.fromarray(a).save(f'results/{ins_id}_modify.png')
        b = 1