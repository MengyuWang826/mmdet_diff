from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import numpy as np


@DETECTORS.register_module()
class Refinementor(TwoStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 num_classes=77,
                 pretrained=None,
                 init_cfg=None):
        super(Refinementor, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_classes = num_classes
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        low_lvl_feat = x[0]
        fpn_feats = self.neck(x[1:])
        return low_lvl_feat, fpn_feats
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      coarse_masks):
        low_lvl_feat, fpn_feats = self.extract_feat(img)

        losses = dict()
        roi_losses = self.roi_head.forward_train(
            low_lvl_feat, 
            fpn_feats, 
            img_metas, 
            gt_masks,
            coarse_masks)
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, coarse_masks, dt_bboxes, rescale=False):
        """Test without augmentation."""

        low_lvl_feat, fpn_feats = self.extract_feat(img)
        if len(coarse_masks[0]):
            mask_pred =  self.roi_head.simple_test(
                low_lvl_feat,
                fpn_feats,
                coarse_masks,
                img_metas)
            dt_bboxes = dt_bboxes.cpu().numpy()
            bboxes = dt_bboxes[0][:, :5]
            labels = dt_bboxes[0][:, 5]
            labels = labels.astype(int)
            bbox_results = self._format_bboxes_results(bboxes, labels)
            mask_results = self._format_mask_results(mask_pred, labels)
        else:
            bbox_results = [np.zeros([0, 4]) for _ in range(self.num_classes)]
            mask_results = [[] for _ in range(self.num_classes)]
        return [(bbox_results, mask_results)]
    
    def _format_bboxes_results(self,bboxes, labels):
        '''
        return [[]*num_class]
        outer: batch
        inter: class, each is a shape[n, 5] np.array(float32)
        '''
        cls_bboxes = []
        for i in range(self.num_classes):
            cur_idx = (labels == i)
            cls_bboxes.append(bboxes[cur_idx])
        return cls_bboxes
    
    def _format_mask_results(self,masks, labels):
        '''
        inter: [[]*n]
        each is a ori_shape binary mask, shape[h, w] bool np.array
        '''
        masks = masks >= 0.5
        masks = masks.detach().cpu().numpy()
        cls_masks = [[] for _ in range(self.num_classes)]
        for i in range(len(masks)):
            cls_masks[labels[i]].append(masks[i])
        return cls_masks
