import torch

pth_file = 'checkpoints/sam/sam_vit_b_01ec64.pth'

a = torch.load(pth_file, map_location=torch.device('cpu'))

b = 1