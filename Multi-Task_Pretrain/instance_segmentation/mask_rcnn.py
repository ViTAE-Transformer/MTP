# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from torch import Tensor
from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig

from .two_stage import MTP_IS_TwoStageDetector


@MODELS.register_module()
class MTP_IS_MaskRCNN(MTP_IS_TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone: ConfigDict = None,
                 rpn_head = ConfigDict(
                        type='MTP_IS_RPNHead',
                        in_channels=256,
                        feat_channels=256,
                        anchor_generator=dict(
                            type='AnchorGenerator',
                            scales=[8],
                            ratios=[0.5, 1.0, 2.0],
                            strides=[4, 8, 16, 32, 64]),
                        bbox_coder=dict(
                            type='DeltaXYWHBBoxCoder',
                            target_means=[.0, .0, .0, .0],
                            target_stds=[1.0, 1.0, 1.0, 1.0]),
                        loss_cls=dict(
                            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                 roi_head = ConfigDict(
                        type='MTP_IS_StandardRoIHead',
                        bbox_roi_extractor=dict(
                            type='SingleRoIExtractor',
                            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                            out_channels=256,
                            featmap_strides=[4, 8, 16, 32]),
                        bbox_head=dict(
                            type='MTP_IS_Shared2FCBBoxHead',
                            in_channels=256,
                            fc_out_channels=1024,
                            roi_feat_size=7,
                            #num_classes=-1,
                            bbox_coder=dict(
                                type='DeltaXYWHBBoxCoder',
                                target_means=[0., 0., 0., 0.],
                                target_stds=[0.1, 0.1, 0.2, 0.2]),
                            reg_class_agnostic=False,
                            loss_cls=dict(
                                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                        mask_roi_extractor=dict(
                            type='SingleRoIExtractor',
                            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                            out_channels=256,
                            featmap_strides=[4, 8, 16, 32]),
                        mask_head=dict(
                            type='MTP_IS_FCNMaskHead',
                            num_convs=4,
                            in_channels=256,
                            conv_out_channels=256,
                            #num_classes=-1,
                            loss_mask=dict(
                                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
                 train_cfg = ConfigDict(
                            rpn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.7,
                                    neg_iou_thr=0.3,
                                    min_pos_iou=0.3,
                                    match_low_quality=True,
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=256,
                                    pos_fraction=0.5,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=False),
                                allowed_border=-1,
                                pos_weight=-1,
                                debug=False),
                            rpn_proposal=dict(
                                nms_pre=2000,
                                max_per_img=1000,
                                nms=dict(type='nms', iou_threshold=0.7),
                                min_bbox_size=0),
                            rcnn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.5,
                                    neg_iou_thr=0.5,
                                    min_pos_iou=0.5,
                                    match_low_quality=True,
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=512,
                                    pos_fraction=0.25,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=True),
                                mask_size=28,
                                pos_weight=-1,
                                debug=False)),
                 test_cfg = ConfigDict(
                            rpn=dict(
                                nms_pre=1000,
                                max_per_img=1000,
                                nms=dict(type='nms', iou_threshold=0.7),
                                min_bbox_size=0),
                            rcnn=dict(
                                score_thr=0.05,
                                nms=dict(type='nms', iou_threshold=0.5),
                                max_per_img=100,
                                mask_thr_binary=0.5)),
                 neck = dict(
                        type='FPN',
                        in_channels=[256, 512, 1024, 2048],
                        out_channels=256,
                        num_outs=5),
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

    def train_before_roihead(self, x: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        return losses, (x, rpn_results_list, batch_data_samples)

        # roi_losses = self.roi_head.loss(x, rpn_results_list,
        #                                 batch_data_samples)
        # losses.update(roi_losses)

        # return losses

    def test_before_roihead(self,
                x,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        #x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # results_list = self.roi_head.predict(
        #     x, rpn_results_list, batch_data_samples, rescale=rescale)

        # batch_data_samples = self.add_pred_to_datasample(
        #     batch_data_samples, results_list)
        return rpn_results_list, rescale