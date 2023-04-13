import torch

sam_pth = 'checkpoints/sam/sam_vit_b_01ec64.pth'
diff_pth = 'work_dirs/bi_sam_diff_lvis/iter_1.pth'

sam = torch.load(sam_pth, map_location=torch.device('cpu'))
diff = torch.load(diff_pth, map_location=torch.device('cpu'))['state_dict']

print('check diff change')
for name in diff:
    if name not in sam:
        print(f'new_module {name}: {diff[name].shape}')
    elif diff[name].shape != sam[name].shape:
        print(f'changed_module {name}: diff_{diff[name].shape}, sam_{sam[name].shape}')

print('check sam del')
for name in sam:
    if name not in diff:
        print(f'del_module {name}: {sam[name].shape}')
