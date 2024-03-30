# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from torch import Tensor
from mmengine.config import ConfigDict
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from .two_stage import MTP_RD_TwoStageDetector
from mmrotate.models.task_modules.coders import *

angle_version = 'le90'

@MODELS.register_module()
class MTP_RD_OrientedRCNN(MTP_RD_TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 #backbone: ConfigType,
                 rpn_head = ConfigDict(
                        type='MTP_RD_OrientedRPNHead',
                        in_channels=256,
                        feat_channels=256,
                        anchor_generator=dict(
                            type='mmdet.AnchorGenerator',
                            scales=[8],
                            ratios=[0.5, 1.0, 2.0],
                            strides=[4, 8, 16, 32, 64],
                            use_box_type=True),
                        bbox_coder=dict(
                            type='MidpointOffsetCoder',
                            angle_version=angle_version,
                            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
                        loss_cls=dict(
                            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                        loss_bbox=dict(
                            type='mmdet.SmoothL1Loss',
                            beta=0.1111111111111111,
                            loss_weight=1.0)),
                 roi_head = ConfigDict(
                        type='MTP_RD_StandardRoIHead',
                        bbox_roi_extractor=dict(
                            type='RotatedSingleRoIExtractor',
                            roi_layer=dict(
                                type='RoIAlignRotated',
                                out_size=7,
                                sample_num=2,
                                clockwise=True),
                            out_channels=256,
                            featmap_strides=[4, 8, 16, 32]),
                        bbox_head=dict(
                            type='MTP_RD_Shared2FCBBoxHead',
                            predict_box_type='rbox',
                            in_channels=256,
                            fc_out_channels=1024,
                            roi_feat_size=7,
                            #num_classes=15,
                            reg_predictor_cfg=dict(type='mmdet.Linear'),
                            cls_predictor_cfg=dict(type='mmdet.Linear'),
                            bbox_coder=dict(
                                type='mmrotate.DeltaXYWHTRBBoxCoder',
                                angle_version=angle_version,
                                norm_factor=None,
                                edge_swap=True,
                                proj_xy=True,
                                target_means=(.0, .0, .0, .0, .0),
                                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                            reg_class_agnostic=True,
                            loss_cls=dict(
                                type='mmdet.CrossEntropyLoss',
                                use_sigmoid=False,
                                loss_weight=1.0),
                            loss_bbox=dict(
                                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))),
                 train_cfg=ConfigDict(
                            rpn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.7,
                                    neg_iou_thr=0.3,
                                    min_pos_iou=0.3,
                                    match_low_quality=True,
                                    ignore_iof_thr=-1,
                                    iou_calculator=dict(type='mmrotate.RBbox2HBboxOverlaps2D')),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=256,
                                    pos_fraction=0.5,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=False),
                                allowed_border=0,
                                pos_weight=-1,
                                debug=False),
                            rpn_proposal=dict(
                                nms_pre=2000,
                                max_per_img=2000,
                                nms=dict(type='nms', iou_threshold=0.8),
                                min_bbox_size=0),
                            rcnn=dict(
                                assigner=dict(
                                    type='MaxIoUAssigner',
                                    pos_iou_thr=0.5,
                                    neg_iou_thr=0.5,
                                    min_pos_iou=0.5,
                                    match_low_quality=False,
                                    iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
                                    ignore_iof_thr=-1),
                                sampler=dict(
                                    type='RandomSampler',
                                    num=512,
                                    pos_fraction=0.25,
                                    neg_pos_ub=-1,
                                    add_gt_as_proposals=True),
                                pos_weight=-1,
                                debug=False)),
                 test_cfg=ConfigDict(
                            rpn=dict(
                                nms_pre=2000,
                                max_per_img=2000,
                                nms=dict(type='nms', iou_threshold=0.8),
                                min_bbox_size=0),
                            rcnn=dict(
                                nms_pre=2000,
                                min_bbox_size=0,
                                score_thr=0.05,
                                nms=dict(type='nms_rotated', iou_threshold=0.1),
                                max_per_img=2000)),
                 neck = ConfigDict(
                        type='mmdet.FPN',
                        in_channels=[256, 512, 1024, 2048],
                        out_channels=256,
                        num_outs=5),
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            #backbone=backbone,
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
        #x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.r_gt_instances.labels = \
                    torch.zeros_like(data_sample.r_gt_instances.labels)

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

        # roi_losses = self.roi_head.loss(x, rpn_results_list,
        #                                 batch_data_samples)
        # losses.update(roi_losses)

        return losses, (x, rpn_results_list, batch_data_samples)
    
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