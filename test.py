# import torch
# import numpy as np
# from einops import rearrange, reduce, repeat
# import json

# BITS = 7

# def decimal_to_bits(x, bits = BITS):
#     """ expects tensor x ranging from 0 to 100(int), outputs bit tensor ranging from -1 to 1 """
#     device = x.device
#     mask = 2 ** torch.arange(bits - 1, -1, -1, device = device)
#     mask = rearrange(mask, 'd -> d 1 1')
#     x = rearrange(x, 'b c h w -> b c 1 h w')

#     bits = ((x & mask) != 0).float()
#     bits = rearrange(bits, 'b c d h w -> b (c d) h w')
#     bits = bits * 2 - 1
#     return bits

# def bits_to_decimal(x, bits = BITS):
#     """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
#     device = x.device

#     x = (x > 0).int()
#     mask = 2 ** torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)
#     mask = rearrange(mask, 'd -> d 1 1')
#     x = rearrange(x, 'b (c d) h w -> b c d h w', d = BITS)
#     dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
#     return dec


# if __name__ == '__main__':
#     a = np.array(range(100))

#     x = np.random.choice(a, 18)
#     x = torch.tensor(x).reshape(-1, 1, 3, 3)
#     


#     a = 1

#     z = x == y

#     print(x[1, 0])
#     b = x[1, 0, 0, 0].item()
#     print(z[1][0])
#     print(z[1][b])
#     print(z)

#     b = torch.arange(7 - 1, -1, -1, device = 'cpu', dtype = torch.int32)

#     a = 1

# def decimal_to_bits(x):
#     """ expects tensor x ranging from 0 to 100(int), outputs bit tensor ranging from -1 to 1 """
#     mask = 2 ** torch.arange(7 - 1, -1, -1, dtype=torch.int32, device=x.device)
#     mask = rearrange(mask, 'd -> d 1 1')
#     print(x[0, 0])
#     x = rearrange(x, 'b c h w -> b c 1 h w')

#     bits = x & mask
#     for i in range(7):
#         print(bits[0, 0, i])
#     bits = (bits) != 0
#     bits = bits.float()
#     bits = rearrange(bits, 'b c d h w -> b (c d) h w')
#     bits = bits * 2 - 1
#     return bits

# def cal_iou(mask, bbox):
#     bbox_map = np.zeros_like(mask)
#     bbox_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
#     si = np.sum(mask & bbox_map)
#     su = np.sum(mask | bbox_map)
#     if su==0:
#         return 0
#     else:
#         return si/su

import torch
import json

if __name__ == '__main__':
    a = torch.tensor([1, 2, 3], device='cuda:0')

    gt = json.load(open('data/refine_annotations/lvis_v1_val_cocofied.json'))

    pass
    # a = np.array(range(100))

    # a = torch.tensor(55)
    # b = torch.tensor(23)

    # c = a & b
    # d = b & a

    # x = np.random.choice(a, 18)
    # x = torch.tensor(x).reshape(-1, 1, 3, 3)
    # print(x[0, 0, 0, 0])
    # # y = torch.tensor(range(100)).reshape(-1, 1, 1)

    # y = decimal_to_bits(x)
    # print(y[0, :, 0, 0])
    
    # a = 1
    # a = [[] for i in range(80)]
    # a[0] = []
    # a = torch.load('work_dirs/diff_r50_fpn/epoch_1.pth')

    # b = 2
    # img_bbox = np.array([[3, 5, 10, 8, 1],
    #                      [2, 1, 6, 10, 8]],
    #                      dtype=np.int32)
    # mask = torch.rand([4, 11, 11], device='cuda:0')
    # mask = (mask > 0.5).int()
    # iou_thr = 0.2

    

    # for i, bbox in enumerate(img_bbox):
    #     bbox_map[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

    # a = (mask & bbox_map)

    # a = 1
    # a = torch.rand((50, 1, 50, 50))
    # img_shape = (32, 50, 3)
    # a = a[:, :, 0:img_shape[0], 0:img_shape[1]]

    # b = 1

    # a = torch.zeros((3, 1, 5, 5))
    # b = torch.ones((1, 1, 5, 5))
    # c = a + b

    # print(c[0])
    # print(c[1])
    # print(c[-1])

    # x = 1

    # a = np.random.choice(2, size=(4, 1, 3, 3))
    # a = torch.tensor(a)
    # t = np.random.choice(10, size=4)
    # t = torch.tensor(t)
    # # t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    # q = np.random.randn(10, 2, 2)
    # q = torch.tensor(q)

    # q_probs = q[t]
    # h, w, num = q_probs.shape[-3:]
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(num):
    #             x = q_probs[0, 0, i, j, k]
    #             y = q[t[0].item()][a[0, 0, i, j]][k]
    #             print(x == y)

    # print(q_probs)
    # b = (1, 3, 3)
    # c = 3


    # print(a == b, c)

    # pass

    a = torch.ones(size=(100,))
    pass

    a = json.load(open('/opt/data/private/wmy/mmdetection/data/annotations/maskrcnn_coco_val.json'))
    b = json.load(open('/opt/data/private/wmy/mmdetection/data/lvis_annotations/maskrcnn_lvis_val.json'))
    c = json.load(open('/opt/data/private/wmy/mmdetection/data/refine_annotations/maskrcnn_val.json'))

    pass