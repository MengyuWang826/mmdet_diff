import torch

sam_pth = 'checkpoints/sam/sam_vit_b_01ec64.pth'
diff_pth = 'work_dirs/bi_sam_diff_lvis/epoch_1.pth'

sam = torch.load(sam_pth, map_location=torch.device('cpu'))
diff = torch.load(diff_pth, map_location=torch.device('cpu'))['state_dict']

changed_modules = ['mask_decoder.mask_tokens.weight', 
                   'mask_decoder.iou_prediction_head.layers.2.weight', 
                   'mask_decoder.iou_prediction_head.layers.2.bias']
changed_modules = set(changed_modules)

diff_pre = dict()

for name in sam:
    if name not in diff:
        if name == 'prompt_encoder.point_embeddings.2.weight':
            diff_pre.update({'prompt_encoder.point_embeddings.0.weight': sam[name]})
        elif name == 'prompt_encoder.point_embeddings.3.weight':
            diff_pre.update({'prompt_encoder.point_embeddings.1.weight': sam[name]})
        elif 'mask_decoder.output_hypernetworks_mlps.0' in name:
            new_name = name.replace('mask_decoder.output_hypernetworks_mlps.0', 'mask_decoder.output_hypernetworks_mlp')
            diff_pre.update({new_name: sam[name]})
        elif 'no_mask_embed' in name:
            diff_pre.update({name: sam[name]})
        else:
            continue
    elif name in changed_modules:
        if name == 'mask_decoder.mask_tokens.weight':
            diff_pre.update({name: sam[name]})
        else:
            diff_pre.update({name: sam[name][0].unsqueeze(0)})
    else:
        diff_pre.update({name: sam[name]})

torch.save(diff_pre, 'pretrain/sam_pre_bbox_muiti.pth')
