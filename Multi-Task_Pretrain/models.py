import torch
import torch.nn as nn
from typing import Optional, Union, List
#from backbone.resnet import Our_ResNet
from backbone.vit_win_rvsa_v3_wsz7 import vit_b_rvsa, vit_l_rvsa
from backbone.intern_image import InternImage

from preprocessing import MTP_DataPreprocessor
from semantic_segmentation.encoder_decoder import MTP_SS_UperNet
from instance_segmentation.mask_rcnn import MTP_IS_MaskRCNN
from rotated_detection.oriented_rcnn import MTP_RD_OrientedRCNN
from mmdet.models.utils import empty_instances


from mmengine.config import ConfigDict

# import instance_segmentation
# import rotated_detection

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MutliTaskPretrnFramework(torch.nn.Module):
    def __init__(self, 
                  args, 
                  classes1: int = 1,
                  classes2: int = 1,
                  classes3: int = 1,
                  batch_augments = None):
        super(MutliTaskPretrnFramework, self).__init__()

        self.data_preprocessor = MTP_DataPreprocessor(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_size_divisor=32,
                pad_mask=True,
                mask_pad_value=0,
                pad_seg=True,
                seg_pad_value=255,
                boxtype2tensor=True,
                batch_augments=batch_augments
        )

        self.data_preprocessor.to('cuda')

        self.args = args

        if args.background == 'True':

            ss_classes1 = classes1
            ss_classes2 = classes2
            ss_classes3 = classes3

            self.is_classes1 = classes1 - 1
            self.is_classes2 = classes2 - 1
            self.is_classes3 = classes3 - 1

            self.rd_classes1 = classes1 - 1
            self.rd_classes2 = classes2 - 1
            self.rd_classes3 = classes3 - 1
        
        else:

            ss_classes1 = classes1 + 1
            ss_classes2 = classes2 + 1
            ss_classes3 = classes3

            self.is_classes1 = classes1
            self.is_classes2 = classes2
            self.is_classes3 = classes3

            self.rd_classes1 = classes1
            self.rd_classes2 = classes2
            self.rd_classes3 = classes3

        # encoder


        if args.backbone == 'vit_b_rvsa':
            self.encoder = vit_b_rvsa(args)
            print('################# Using ViT-B + RVSA as backbone! ###################')
        elif args.backbone == 'vit_l_rvsa':
            self.encoder = vit_l_rvsa(args)
            print('################# Using ViT-L + RVSA as backbone! ###################')
        elif args.backbone == 'internimage_xl':
            self.encoder = InternImage(core_op='DCNv3',
                            channels=192,
                            depths=[5, 5, 24, 5],
                            groups=[12, 24, 48, 96],
                            mlp_ratio=4.,
                            drop_path_rate=0.2,
                            norm_layer='LN',
                            layer_scale=1e-5,
                            offset_scale=2.0,
                            post_norm=True,
                            with_cp=True,
                            out_indices=(0, 1, 2, 3)
                            )
            print('################# Using InternImage-XL as backbone! ###################')
        

        # decoder
            
        print('################# Using UperNet for semseg! ######################')

        self.semsegdecoder = MTP_SS_UperNet(

            decode_head = dict(
                    type='UPerHead',
                    num_classes = 1,
                    in_channels=self.encoder.out_channels,
                    ignore_index=255,
                    in_index=[0, 1, 2, 3],
                    pool_scales=(1, 2, 3, 6),
                    channels=256,
                    dropout_ratio=0.1,
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
            )

        self.semseghead_1 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(256, ss_classes1, kernel_size=1)
            )

        self.semseghead_2 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(256, ss_classes2, kernel_size=1)
            )

        self.semseghead_3 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(256, ss_classes3, kernel_size=1)
            )
        
        # ins

        print('################# Using Mask-RCNN for insseg! ######################')

        self.inssegdecoder = MTP_IS_MaskRCNN(
                neck = ConfigDict(
                        type='FPN',
                        in_channels=self.encoder.out_channels,
                        out_channels=256,
                        num_outs=5),
            )
        
        self.inssegroiboxhead_fc_cls1 = nn.Linear(self.inssegdecoder.roi_head.bbox_head.cls_last_dim, self.is_classes1 + 1)
        self.inssegroiboxhead_fc_cls2 = nn.Linear(self.inssegdecoder.roi_head.bbox_head.cls_last_dim, self.is_classes2 + 1)
        self.inssegroiboxhead_fc_cls3 = nn.Linear(self.inssegdecoder.roi_head.bbox_head.cls_last_dim, self.is_classes3 + 1)

        self.inssegroiboxhead_fc_reg1 = nn.Linear(self.inssegdecoder.roi_head.bbox_head.reg_last_dim, 
                                                  int(self.is_classes1*self.inssegdecoder.roi_head.bbox_head.bbox_coder.encode_size))
        self.inssegroiboxhead_fc_reg2 = nn.Linear(self.inssegdecoder.roi_head.bbox_head.reg_last_dim, 
                                                  int(self.is_classes2*self.inssegdecoder.roi_head.bbox_head.bbox_coder.encode_size))
        self.inssegroiboxhead_fc_reg3 = nn.Linear(self.inssegdecoder.roi_head.bbox_head.reg_last_dim, 
                                                  int(self.is_classes3*self.inssegdecoder.roi_head.bbox_head.bbox_coder.encode_size))
        
        self.inssegroimaskhead_conv1 = nn.Conv2d(self.inssegdecoder.roi_head.mask_head.conv_out_channels, self.is_classes1, 1)
        self.inssegroimaskhead_conv2 = nn.Conv2d(self.inssegdecoder.roi_head.mask_head.conv_out_channels, self.is_classes2, 1)
        self.inssegroimaskhead_conv3 = nn.Conv2d(self.inssegdecoder.roi_head.mask_head.conv_out_channels, self.is_classes3, 1)

        print('################# Using Oriented-RCNN for rotdet! ######################')

        self.rotdetdecoder =MTP_RD_OrientedRCNN(
                neck = ConfigDict(
                        type='mmdet.FPN',
                        in_channels=self.encoder.out_channels,
                        out_channels=256,
                        num_outs=5)
            )
        
        self.rotdetroiboxhead_fc_cls1 = nn.Linear(self.rotdetdecoder.roi_head.bbox_head.cls_last_dim, self.rd_classes1 + 1)
        self.rotdetroiboxhead_fc_cls2 = nn.Linear(self.rotdetdecoder.roi_head.bbox_head.cls_last_dim, self.rd_classes2 + 1)
        self.rotdetroiboxhead_fc_cls3 = nn.Linear(self.rotdetdecoder.roi_head.bbox_head.cls_last_dim, self.rd_classes3 + 1)

        self.rotdetroiboxhead_fc_reg1 = nn.Linear(self.rotdetdecoder.roi_head.bbox_head.reg_last_dim, 
                                                  self.rotdetdecoder.roi_head.bbox_head.bbox_coder.encode_size)
        self.rotdetroiboxhead_fc_reg2 = nn.Linear(self.rotdetdecoder.roi_head.bbox_head.reg_last_dim, 
                                                  self.rotdetdecoder.roi_head.bbox_head.bbox_coder.encode_size)
        self.rotdetroiboxhead_fc_reg3 = nn.Linear(self.rotdetdecoder.roi_head.bbox_head.reg_last_dim, 
                                                  self.rotdetdecoder.roi_head.bbox_head.bbox_coder.encode_size)
        

        print('################# Using UperNet for Pretraining! ######################')

        if args.backbone == 'vit_b_rvsa':
            if args.init_backbone == 'mae':
                self.encoder.init_weights('/diwang22/work_dir/pretrn/vit-b-checkpoint-1599.pth')
                print('################# Initing ViT-B + RVSA pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ViT-B + RVSA SEP Pretraining! ###################')
            else:
                raise NotImplementedError
        elif args.backbone == 'vit_l_rvsa':
            if args.init_backbone == 'mae':
                self.encoder.init_weights('/diwang22/work_dir/mae_pretrain/vit_large_norm_pix/vit-l-mae-checkpoint-1599.pth')
                print('################# Initing ViT-L + RVSA pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ViT-L + RVSA SEP Pretraining! ###################')
            else:
                raise NotImplementedError

        elif args.backbone == 'internimage_xl':
            if args.init_backbone == 'imp':
                self.encoder.init_weights('/diwang22/work_dir/pretrn/internimage_xl_22kto1k_384.pth')
                print('################# Initing InterImage-T pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure InterImage-T SEP Pretraining! ###################')
            else:
                raise NotImplementedError
            

        #self.initialize()

    def train_rotdet_roi_head_box_head_forward(self, r, sampling_results, num_classes, fc_cls=None, fc_reg=None):
                                           
        bbox_feats, rois = self.rotdetdecoder.roi_head.train_bbox_forward(r, sampling_results)

        # box head of roi head
        x_cls, x_reg = self.rotdetdecoder.roi_head.bbox_head(bbox_feats)

        cls_score, bbox_pred = fc_cls(x_cls), fc_reg(x_reg)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        
        bbox_results = self.rotdetdecoder.roi_head.bbox_loss(num_classes, sampling_results, rois, bbox_results)

        return bbox_results
    
    def test_rotdet_roi_head_box_head_forward(self, r, batch_img_metas, proposals, rois, rcnn_test_cfg, num_classes, bbox_rescale, fc_cls=None, fc_reg=None):

        bbox_feats = self.rotdetdecoder.roi_head.test_bbox_roi_feats(r, rois)
        x_cls, x_reg = self.rotdetdecoder.roi_head.bbox_head(bbox_feats)
        cls_score, bbox_pred = fc_cls(x_cls), fc_reg(x_reg)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        result_list = self.rotdetdecoder.roi_head.predict_bbox(
        batch_img_metas, bbox_results, proposals, rois, rcnn_test_cfg, num_classes, rescale = bbox_rescale
        )

        return result_list

    def train_insseg_roi_head_box_head_forward(self, n, sampling_results, num_classes, fc_cls=None, fc_reg=None):

        # roi layer of box head
        bbox_feats, rois = self.inssegdecoder.roi_head.train_bbox_forward(n, sampling_results)

        # box head of roi head
        x_cls, x_reg = self.inssegdecoder.roi_head.bbox_head(bbox_feats)

        cls_score, bbox_pred = fc_cls(x_cls), fc_reg(x_reg)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        bbox_results = self.inssegdecoder.roi_head.bbox_loss(num_classes, sampling_results, rois, bbox_results)
                
        return bbox_results, bbox_feats


    def test_insseg_roi_head_box_head_forward(self, n, batch_img_metas, proposals, rois, rcnn_test_cfg, num_classes, bbox_rescale, fc_cls=None, fc_reg=None):

        bbox_feats = self.inssegdecoder.roi_head.test_bbox_roi_feats(n, rois)
        x_cls, x_reg = self.inssegdecoder.roi_head.bbox_head(bbox_feats)
        cls_score, bbox_pred = fc_cls(x_cls), fc_reg(x_reg)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        result_list = self.inssegdecoder.roi_head.predict_bbox(
        batch_img_metas, bbox_results, proposals, rois, rcnn_test_cfg, num_classes, rescale = bbox_rescale
        )
                
        return result_list
    
    def train_insseg_roi_head_mask_head_forward(self, n, sampling_results, bbox_feats, batch_gt_instance, data_sample, conv_logits=None):

        mask_feats = self.inssegdecoder.roi_head.train_mask_forward(n, sampling_results, bbox_feats)
        # mask head of roi head
        x_mask = self.inssegdecoder.roi_head.mask_head(mask_feats)
        mask_preds = conv_logits(x_mask)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)
        mask_results = self.inssegdecoder.roi_head.mask_loss(mask_results, sampling_results, batch_gt_instance)
                
        return mask_results

    def test_insseg_roi_head_mask_head_forward(self, n, batch_img_metas, mask_rois, results_list, rescale, conv_logits=None):
                                     
        mask_feats = self.inssegdecoder.roi_head.test_mask_roi_feats(n, mask_rois)
        
        x_mask = self.inssegdecoder.roi_head.mask_head(mask_feats)
        
        mask_preds = conv_logits(x_mask)

        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)

        results_list = self.inssegdecoder.roi_head.predict_mask(batch_img_metas, results_list, mask_results, rescale = rescale)
        
        return results_list

    def forward(self, data1, data2, data3):

        if self.training:

            data1 = self.data_preprocessor(data1, True)
            data2 = self.data_preprocessor(data2, True)
            data3 = self.data_preprocessor(data3, True)

        else:

            data1 = self.data_preprocessor(data1, False)
            data2 = self.data_preprocessor(data2, False)
            data3 = self.data_preprocessor(data3, False)

        x1, data_sample1 = data1['inputs'], data1['data_samples']
        x2, data_sample2 = data2['inputs'], data2['data_samples']
        x3, data_sample3 = data3['inputs'], data3['data_samples']

        b = x1.shape[0]

        x = torch.cat((x1, x2, x3), 0)

        x = self.encoder(x)

        e1 = [m[:b] for m in x]
        e2 = [m[b:2*b] for m in x]
        e3 = [m[2*b:] for m in x]

        # e1 = self.encoder(x1)
        # e2 = self.encoder(x2)
        # e3 = self.encoder(x3)

        ######################### sem seg

        if 'ss' in self.args.tasks:

            ss1 = self.semsegdecoder.decode_head._forward_feature(e1)
            ss2 = self.semsegdecoder.decode_head._forward_feature(e2)
            ss3 = self.semsegdecoder.decode_head._forward_feature(e3)

            seg_logits1 = self.semseghead_1(ss1)
            seg_logits2 = self.semseghead_2(ss2)
            seg_logits3 = self.semseghead_3(ss3)

        ######################### ins seg
            
        if 'is' in self.args.tasks:

            n1 = self.inssegdecoder.neck(e1)
            n2 = self.inssegdecoder.neck(e2)
            n3 = self.inssegdecoder.neck(e3)

        ######################### rot det
            
        if 'rd' in self.args.tasks:

            r1 = self.rotdetdecoder.neck(e1)
            r2 = self.rotdetdecoder.neck(e2)
            r3 = self.rotdetdecoder.neck(e3)

        if self.training:

            losses = [0, 0, 0]
            
            ######################### train sem seg
            
            if 'ss' in self.args.tasks:
                loss_ss1 = self.semsegdecoder.decode_head.loss_by_feat(seg_logits1, data_sample1)
                loss_ss2 = self.semsegdecoder.decode_head.loss_by_feat(seg_logits2, data_sample2)
                loss_ss3 = self.semsegdecoder.decode_head.loss_by_feat(seg_logits3, data_sample3)

                loss_ss = [loss_ss1, loss_ss2, loss_ss3]

                losses[0] = loss_ss
            
            ######################### train ins seg

            if 'is' in self.args.tasks:

                # rpn head
                loss_is1, tbr1 = self.inssegdecoder.train_before_roihead(n1, data_sample1)
                loss_is2, tbr2 = self.inssegdecoder.train_before_roihead(n2, data_sample2)
                loss_is3, tbr3 = self.inssegdecoder.train_before_roihead(n3, data_sample3)

                # select propsal
                sampling_results1, batch_gt_instance1 = self.inssegdecoder.roi_head.gen_sampling_results(tbr1)
                sampling_results2, batch_gt_instance2 = self.inssegdecoder.roi_head.gen_sampling_results(tbr2)
                sampling_results3, batch_gt_instance3 = self.inssegdecoder.roi_head.gen_sampling_results(tbr3)

                # roi_head_box_head

                bbox_results1, bbox_feats1 = self.train_insseg_roi_head_box_head_forward(n1, sampling_results1, self.is_classes1, 
                                                                                        fc_cls=self.inssegroiboxhead_fc_cls1,
                                                                                        fc_reg=self.inssegroiboxhead_fc_reg1)
                bbox_results2, bbox_feats2 = self.train_insseg_roi_head_box_head_forward(n2, sampling_results2, self.is_classes2,
                                                                                        fc_cls=self.inssegroiboxhead_fc_cls2,
                                                                                        fc_reg=self.inssegroiboxhead_fc_reg2)
                bbox_results3, bbox_feats3 = self.train_insseg_roi_head_box_head_forward(n3, sampling_results3, self.is_classes3,
                                                                                        fc_cls=self.inssegroiboxhead_fc_cls3,
                                                                                        fc_reg=self.inssegroiboxhead_fc_reg3)

                
                
                loss_is1.update(bbox_results1['loss_bbox'])
                loss_is2.update(bbox_results2['loss_bbox'])
                loss_is3.update(bbox_results3['loss_bbox'])

                # roi_head_mask_head

                mask_results1 = self.train_insseg_roi_head_mask_head_forward(n1, sampling_results1, bbox_feats1, batch_gt_instance1, data_sample1, 
                                                                            conv_logits=self.inssegroimaskhead_conv1)
                mask_results2 = self.train_insseg_roi_head_mask_head_forward(n2, sampling_results2, bbox_feats2, batch_gt_instance2, data_sample2,
                                                                            conv_logits=self.inssegroimaskhead_conv2)
                mask_results3 = self.train_insseg_roi_head_mask_head_forward(n3, sampling_results3, bbox_feats3, batch_gt_instance3, data_sample3,
                                                                            conv_logits=self.inssegroimaskhead_conv3)

                loss_is1.update(mask_results1['loss_mask'])
                loss_is2.update(mask_results2['loss_mask'])
                loss_is3.update(mask_results3['loss_mask'])

                loss_is = [loss_is1, loss_is2, loss_is3]

                losses[1] = loss_is

            ######################### train rot det
                
            if 'rd' in self.args.tasks:

                # rpn head
                loss_rd1, tbr1 = self.rotdetdecoder.train_before_roihead(r1, data_sample1)
                loss_rd2, tbr2 = self.rotdetdecoder.train_before_roihead(r2, data_sample2)
                loss_rd3, tbr3 = self.rotdetdecoder.train_before_roihead(r3, data_sample3)

                # select propsal
                sampling_results1 = self.rotdetdecoder.roi_head.gen_sampling_results(tbr1)
                sampling_results2 = self.rotdetdecoder.roi_head.gen_sampling_results(tbr2)
                sampling_results3 = self.rotdetdecoder.roi_head.gen_sampling_results(tbr3)

                # roi_head_box_head

                bbox_results1 = self.train_rotdet_roi_head_box_head_forward(r1, sampling_results1, self.rd_classes1,
                                                                                        fc_cls = self.rotdetroiboxhead_fc_cls1,
                                                                                        fc_reg = self.rotdetroiboxhead_fc_reg1)
                bbox_results2 = self.train_rotdet_roi_head_box_head_forward(r2, sampling_results2, self.rd_classes2,
                                                                                        fc_cls = self.rotdetroiboxhead_fc_cls2,
                                                                                        fc_reg = self.rotdetroiboxhead_fc_reg2)
                bbox_results3 = self.train_rotdet_roi_head_box_head_forward(r3, sampling_results3, self.rd_classes3,
                                                                                        fc_cls = self.rotdetroiboxhead_fc_cls3,
                                                                                        fc_reg = self.rotdetroiboxhead_fc_reg3)
                
                loss_rd1.update(bbox_results1['loss_bbox'])
                loss_rd2.update(bbox_results2['loss_bbox'])
                loss_rd3.update(bbox_results3['loss_bbox'])

                loss_rd = [loss_rd1, loss_rd2, loss_rd3]

                losses[2] = loss_rd

            return losses


        else:
            outputs = [0, 0, 0]
            
            ######################### test sem seg

            if 'ss' in self.args.tasks:

                if data_sample1 is not None:
                    batch_img_metas1 = [
                        data_sample.metainfo for data_sample in data_sample1
                    ]
                    #print(batch_img_metas)
                else:
                    batch_img_metas1 = [
                        dict(
                            ori_shape=x1.shape[2:],
                            img_shape=x1.shape[2:],
                            pad_shape=x1.shape[2:],
                            padding_size=[0, 0, 0, 0])
                    ] * x1.shape[0]

                if data_sample2 is not None:
                    batch_img_metas2 = [
                        data_sample.metainfo for data_sample in data_sample2
                    ]
                    #print(batch_img_metas)
                else:
                    batch_img_metas2 = [
                        dict(
                            ori_shape=x2.shape[2:],
                            img_shape=x2.shape[2:],
                            pad_shape=x2.shape[2:],
                            padding_size=[0, 0, 0, 0])
                    ] * x2.shape[0]

                if data_sample3 is not None:
                    batch_img_metas3 = [
                        data_sample.metainfo for data_sample in data_sample3
                    ]
                    #print(batch_img_metas)
                else:
                    batch_img_metas3 = [
                        dict(
                            ori_shape=x3.shape[2:],
                            img_shape=x3.shape[2:],
                            pad_shape=x3.shape[2:],
                            padding_size=[0, 0, 0, 0])
                    ] * x3.shape[0]

                seg_logits1 = self.semsegdecoder.decode_head.predict_by_feat(seg_logits1, batch_img_metas1)
                seg_logits2 = self.semsegdecoder.decode_head.predict_by_feat(seg_logits2, batch_img_metas2)
                seg_logits3 = self.semsegdecoder.decode_head.predict_by_feat(seg_logits3, batch_img_metas3)

                output_ss1 = self.semsegdecoder.postprocess_result(seg_logits1, data_sample1)
                output_ss2 = self.semsegdecoder.postprocess_result(seg_logits2, data_sample2)
                output_ss3 = self.semsegdecoder.postprocess_result(seg_logits3, data_sample3)

                output_ss = [output_ss1, output_ss2, output_ss3]

                outputs[0] = output_ss

            ######################### test ins seg

            if 'is' in self.args.tasks:

                rpn_results_list1, rescale = self.inssegdecoder.test_before_roihead(n1, data_sample1)
                rpn_results_list2, _ = self.inssegdecoder.test_before_roihead(n2, data_sample2)
                rpn_results_list3, _ = self.inssegdecoder.test_before_roihead(n3, data_sample3)

                assert self.inssegdecoder.roi_head.with_bbox, 'Bbox head must be implemented.'

                batch_img_metas1 = [
                data_samples.metainfo for data_samples in data_sample1
                ]
                batch_img_metas2 = [
                data_samples.metainfo for data_samples in data_sample2
                ]
                batch_img_metas3 = [
                data_samples.metainfo for data_samples in data_sample3
                ]

                proposals1, rois1 = self.inssegdecoder.roi_head.test_bbox_generate_roi(rpn_results_list1)
                proposals2, rois2 = self.inssegdecoder.roi_head.test_bbox_generate_roi(rpn_results_list2)
                proposals3, rois3 = self.inssegdecoder.roi_head.test_bbox_generate_roi(rpn_results_list3)

                rcnn_test_cfg = self.inssegdecoder.roi_head.test_cfg
                bbox_rescale = rescale if not self.inssegdecoder.roi_head.with_mask else False

                if rois1.shape[0] == 0:
                    results_list1 = empty_instances(
                        batch_img_metas1,
                        rois1.device,
                        task_type='bbox',
                        box_type=self.inssegdecoder.roi_head.bbox_head.predict_box_type,
                        num_classes=self.is_classes1,
                        score_per_cls=rcnn_test_cfg is None)
                    
                else:
                    results_list1 = self.test_insseg_roi_head_box_head_forward(n1, batch_img_metas1, proposals1, rois1, 
                                                                            rcnn_test_cfg, self.is_classes1, bbox_rescale,
                                                                            fc_cls=self.inssegroiboxhead_fc_cls1,
                                                                            fc_reg=self.inssegroiboxhead_fc_reg1)
                    
                if rois2.shape[0] == 0:
                    results_list2 = empty_instances(
                        batch_img_metas2,
                        rois2.device,
                        task_type='bbox',
                        box_type=self.inssegdecoder.roi_head.bbox_head.predict_box_type,
                        num_classes=self.is_classes2,
                        score_per_cls=rcnn_test_cfg is None)
                else:
                    results_list2 = self.test_insseg_roi_head_box_head_forward(n2, batch_img_metas2, proposals2, rois2, 
                                                                            rcnn_test_cfg, self.is_classes2, bbox_rescale,
                                                                            fc_cls=self.inssegroiboxhead_fc_cls2,
                                                                            fc_reg=self.inssegroiboxhead_fc_reg2)

                if rois3.shape[0] == 0:
                    results_list3 = empty_instances(
                        batch_img_metas3,
                        rois3.device,
                        task_type='bbox',
                        box_type=self.inssegdecoder.roi_head.bbox_head.predict_box_type,
                        num_classes=self.is_classes3,
                        score_per_cls=rcnn_test_cfg is None)
                else:
                    results_list3 = self.test_insseg_roi_head_box_head_forward(n3, batch_img_metas3, proposals3, rois3, 
                                                                            rcnn_test_cfg, self.is_classes3, bbox_rescale,
                                                                            fc_cls=self.inssegroiboxhead_fc_cls3,
                                                                            fc_reg=self.inssegroiboxhead_fc_reg3)

                # mask head of roi head
                    
                mask_rois1 = self.inssegdecoder.roi_head.test_mask_generate_rois(results_list1)
                mask_rois2 = self.inssegdecoder.roi_head.test_mask_generate_rois(results_list2)
                mask_rois3 = self.inssegdecoder.roi_head.test_mask_generate_rois(results_list3)

                if mask_rois1.shape[0] == 0:
                    results_list1 = empty_instances(
                        batch_img_metas1,
                        mask_rois1.device,
                        task_type='mask',
                        instance_results=results_list1,
                        mask_thr_binary=self.inssegdecoder.roi_head.test_cfg.mask_thr_binary)
                else:
                    results_list1 = self.test_insseg_roi_head_mask_head_forward(n1, batch_img_metas1, mask_rois1, results_list1, 
                                                                                rescale, conv_logits=self.inssegroimaskhead_conv1)

                if mask_rois2.shape[0] == 0:
                    results_list2 = empty_instances(
                        batch_img_metas2,
                        mask_rois2.device,
                        task_type='mask',
                        instance_results=results_list2,
                        mask_thr_binary=self.inssegdecoder.roi_head.test_cfg.mask_thr_binary)
                else:
                    results_list2 = self.test_insseg_roi_head_mask_head_forward(n2, batch_img_metas2, mask_rois2, results_list2, 
                                                                                rescale, conv_logits=self.inssegroimaskhead_conv2)

                if mask_rois3.shape[0] == 0:
                    results_list3 = empty_instances(
                        batch_img_metas3,
                        mask_rois3.device,
                        task_type='mask',
                        instance_results=results_list3,
                        mask_thr_binary=self.inssegdecoder.roi_head.test_cfg.mask_thr_binary)
                else:
                    results_list3 = self.test_insseg_roi_head_mask_head_forward(n3, batch_img_metas3, mask_rois3, results_list3, 
                                                                                rescale, conv_logits=self.inssegroimaskhead_conv3)


                output_is1 = self.inssegdecoder.add_pred_to_datasample(data_sample1, results_list1)
                output_is2 = self.inssegdecoder.add_pred_to_datasample(data_sample2, results_list2)
                output_is3 = self.inssegdecoder.add_pred_to_datasample(data_sample3, results_list3)

                output_is = [output_is1, output_is2, output_is3]

                outputs[1] = output_is

            ######################### test rot det
                
            if 'rd' in self.args.tasks:

                rpn_results_list1, rescale = self.rotdetdecoder.test_before_roihead(r1, data_sample1)
                rpn_results_list2, _ = self.rotdetdecoder.test_before_roihead(r2, data_sample2)
                rpn_results_list3, _ = self.rotdetdecoder.test_before_roihead(r3, data_sample3)

                assert self.rotdetdecoder.roi_head.with_bbox, 'Bbox head must be implemented.'

                batch_img_metas1 = [
                    data_samples.metainfo for data_samples in data_sample1
                ]
                batch_img_metas2 = [
                    data_samples.metainfo for data_samples in data_sample2
                ]
                batch_img_metas3 = [
                    data_samples.metainfo for data_samples in data_sample3
                ]

                proposals1, rois1 = self.rotdetdecoder.roi_head.test_bbox_generate_roi(rpn_results_list1)
                proposals2, rois2 = self.rotdetdecoder.roi_head.test_bbox_generate_roi(rpn_results_list2)
                proposals3, rois3 = self.rotdetdecoder.roi_head.test_bbox_generate_roi(rpn_results_list3)

                rcnn_test_cfg = self.rotdetdecoder.roi_head.test_cfg
                bbox_rescale = rescale if not self.rotdetdecoder.roi_head.with_mask else False

                if rois1.shape[0] == 0:
                    results_list1 = empty_instances(
                        batch_img_metas1,
                        rois1.device,
                        task_type='bbox',
                        box_type=self.rotdetdecoder.roi_head.bbox_head.predict_box_type,
                        num_classes=self.rd_classes1,
                        score_per_cls=rcnn_test_cfg is None)
                else:
                    results_list1 = self.test_rotdet_roi_head_box_head_forward(r1, batch_img_metas1, proposals1, rois1, 
                                                                            rcnn_test_cfg, self.rd_classes1, bbox_rescale, 
                                                                            fc_cls = self.rotdetroiboxhead_fc_cls1, 
                                                                            fc_reg = self.rotdetroiboxhead_fc_reg1)
                if rois2.shape[0] == 0:
                    results_list2 = empty_instances(
                        batch_img_metas2,
                        rois2.device,
                        task_type='bbox',
                        box_type=self.rotdetdecoder.roi_head.bbox_head.predict_box_type,
                        num_classes=self.rd_classes2,
                        score_per_cls=rcnn_test_cfg is None)
                else:
                    results_list2 = self.test_rotdet_roi_head_box_head_forward(r2, batch_img_metas2, proposals2, rois2, 
                                                                            rcnn_test_cfg, self.rd_classes2, bbox_rescale, 
                                                                            fc_cls = self.rotdetroiboxhead_fc_cls2, 
                                                                            fc_reg = self.rotdetroiboxhead_fc_reg2)

                if rois3.shape[0] == 0:
                    results_list3 = empty_instances(
                        batch_img_metas3,
                        rois3.device,
                        task_type='bbox',
                        box_type=self.rotdetdecoder.roi_head.bbox_head.predict_box_type,
                        num_classes=self.rd_classes3,
                        score_per_cls=rcnn_test_cfg is None)
                else:
                    results_list3 = self.test_rotdet_roi_head_box_head_forward(r3, batch_img_metas3, proposals3, rois3, 
                                                                            rcnn_test_cfg, self.rd_classes3, bbox_rescale, 
                                                                            fc_cls = self.rotdetroiboxhead_fc_cls3, 
                                                                            fc_reg = self.rotdetroiboxhead_fc_reg3)
                    
                output_rd1 = self.rotdetdecoder.add_pred_to_datasample(data_sample1, results_list1)
                output_rd2 = self.rotdetdecoder.add_pred_to_datasample(data_sample2, results_list2)
                output_rd3 = self.rotdetdecoder.add_pred_to_datasample(data_sample3, results_list3)

                output_rd = [output_rd1, output_rd2, output_rd3]

                outputs[2] = output_rd
                
            return outputs
        
    # def initialize(self):

    #     initialize_head(self.semseghead_1)
    #     initialize_head(self.semseghead_2)
    #     initialize_head(self.semseghead_3)

    #     initialize_head(self.inssegroiboxhead_fc_cls1)
    #     initialize_head(self.inssegroiboxhead_fc_cls2)
    #     initialize_head(self.inssegroiboxhead_fc_cls3)
    #     initialize_head(self.inssegroiboxhead_fc_reg1)
    #     initialize_head(self.inssegroiboxhead_fc_reg2)
    #     initialize_head(self.inssegroiboxhead_fc_reg3)

    #     initialize_head(self.inssegroimaskhead_conv1)
    #     initialize_head(self.inssegroimaskhead_conv2)
    #     initialize_head(self.inssegroimaskhead_conv3)

    #     initialize_head(self.rotdetroiboxhead_fc_cls1)
    #     initialize_head(self.rotdetroiboxhead_fc_cls2)
    #     initialize_head(self.rotdetroiboxhead_fc_cls3)
    #     initialize_head(self.rotdetroiboxhead_fc_reg1)
    #     initialize_head(self.rotdetroiboxhead_fc_reg2)
    #     initialize_head(self.rotdetroiboxhead_fc_reg3)

        
        










