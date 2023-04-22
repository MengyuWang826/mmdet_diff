import torch

sam_pth = 'checkpoints/sam/sam_vit_h_4b8939.pth'

sam = torch.load(sam_pth, map_location=torch.device('cpu'))


sam['prompt_encoder.point_embeddings.0.weight'] = sam['prompt_encoder.point_embeddings.2.weight']
sam['prompt_encoder.point_embeddings.1.weight'] = sam['prompt_encoder.point_embeddings.3.weight']
sam.pop('prompt_encoder.point_embeddings.2.weight')
sam.pop('prompt_encoder.point_embeddings.3.weight')
sam.pop('prompt_encoder.not_a_point_embed.weight')

torch.save(sam, 'checkpoints/sam/sam_vit_h_multi.pth')
