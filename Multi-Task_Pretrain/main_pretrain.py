#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import os, random, time
import logging
from tqdm import tqdm

import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import augmentations
from sync_batchnorm import patch_replication_callback
from datasets import SOTAMultiTaskDataset, SIORMultiTaskDataset, FASTMultiTaskDataset
from utils import set_configs, parse_losses, parse_datainfos, data_augs

import cv2
import subprocess

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch MutliTask Pretraining')
parser.add_argument('--backbone', type=str, default=None, choices=['vit_b_rvsa', 'vit_l_rvsa', 'internimage_xl'], help='backbone name')
parser.add_argument('--datasets', type=str, nargs='+',help='used dataset')
parser.add_argument('--tasks', type=str, nargs='+',help='used dataset')
# epoch
parser.add_argument('--start_epoch', type=int, default=0, help='index of start epoch')
parser.add_argument('--start_iter', type=int, default=0, help='index of start iteration')
parser.add_argument('--end_iter', type=int, default=5, help='number of epochs to train')

# batch size
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--batch_size_val', type=int, default=8, help='input batch size for validation')
parser.add_argument('--workers', type=int, default=0, help='workers num')

parser.add_argument('--batch_mode', type=str, default='avg', choices=['ratio','avg'], help='how to assign batch size')

# learning rate
parser.add_argument('--lr', type=float, default=None, help='actual learning rate')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

# distributed
parser.add_argument('--distributed', type=str, default='True', choices=['True', 'False'], help='distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', type=int, default=0)

# ft
parser.add_argument('--ft', type=str, default='False', choices=['True', 'False'], help='finetune model')
parser.add_argument('--resume', type=str, default=None, help='dataset name')

# save
parser.add_argument('--save_path', type=str, default=None, help='path of saving model')

# ignored
parser.add_argument('--ignore_label', type=int, default=255, help='ignore index of loss')

# interval
parser.add_argument('--interval', type=int,  default=2000, help='valid interval')

# init_backbone
parser.add_argument('--init_backbone', type=str, default=None, choices=['imp', 'rsp', 'none', 
                                                                        'mae', 'beit'], help='init model')

# port 
parser.add_argument('--port', type=str, default=None, help='master ports')

# input img size
parser.add_argument('--image_size', type=int, default=None, help='image size')

# background
parser.add_argument('--background', type=str, default='True', choices=['True', 'False'], help='consider background')

# checkpoint mechanism
parser.add_argument('--use_ckpt', type=str, default='False', choices=['True', 'False'], help='consider background')

# mixed presicion
parser.add_argument('--mixed_precision', type=str, default='False', choices=['True', 'False'], help='consider background')


args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

logger_name = "main-logger"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'), mode='a')
log_format = '%(asctime)s %(message)s'
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)

handler = logging.StreamHandler()
fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)

def main_process(args):
    return not args.distributed == 'True' or (args.distributed == 'True' and args.rank % args.world_size == 0)

def set_seeds(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

################################################### setting ###################################################

if args.distributed == 'True':

    if 'SLURM_NTASKS' in os.environ.keys():
        logger.info('#################### srun for DDP! ############################')
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.rank = int(os.environ['SLURM_PROCID']) #if 'RANK' in os.environ else 0
        LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
        torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        dist_url = 'tcp://%s:%s' % (addr, args.port)
        dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size, rank=args.rank)#分布式TCP初始化

    else:
        logger.info('#################### Launch for DDP! ############################')
        args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        args.rank = int(os.environ["RANK"])
        LOCAL_RANK = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)#分布式TCP初始化

if main_process(args):
    logger.info('<<<<<<<<<<<<<<<<< args <<<<<<<<<<<<<<<<<')
    logger.info(args)
    
########################################################## dataset ####################################################

train_pipeline, valid_pipeline, batch_augments =  set_configs(args.image_size)

classes1 = 18
classes2 = 20
classes3 = 37

if args.background == 'True':
    classes1 += 1
    classes2 += 1
    classes3 += 1

    logger.info('Consider background!!!!!!!!!!!!!!!!!!!!!!!!')

################### SOTA

data_root = '/diwang22/dataset/samrs/dotav2_1024_rbb'
data_prefix = dict(img='trainval/images', seg='trainval/rhbox_segs_init/gray/')
ann_file = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/sota_rbb_train_ins_segmentation.json'
rdet_ann_dir = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/rbbtxts/'
seg_map_suffix = '.png'
rdet_post_fix = '.txt'
sota_trn_dataset = SOTAMultiTaskDataset(data_root = data_root, data_prefix = data_prefix, 
                                        ann_file = ann_file, rdet_ann_dir = rdet_ann_dir,
                                        seg_map_suffix = seg_map_suffix, rdet_post_fix = rdet_post_fix, 
                                        pipeline = train_pipeline)

data_root = '/diwang22/dataset/samrs/dotav2_1024_rbb'
data_prefix = dict(img='trainval/images', seg='trainval/rhbox_segs_init/gray/')
ann_file = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/sota_rbb_valid_ins_segmentation.json'
rdet_ann_dir = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/rbbtxts/'
seg_map_suffix = '.png'
rdet_post_fix = '.txt'
sota_val_dataset = SOTAMultiTaskDataset(data_root = data_root, data_prefix = data_prefix, 
                                        ann_file = ann_file, rdet_ann_dir = rdet_ann_dir,
                                        seg_map_suffix = seg_map_suffix, rdet_post_fix = rdet_post_fix, 
                                        pipeline = valid_pipeline)

################### SIOR

data_root = '/diwang22/dataset/samrs/dior'
data_prefix = dict(img='JPEGImages-trainval/', seg='hbox_segs_trainvaltest_init/gray')
ann_file = '/diwang22/dataset/samrs/dior/sior_train_ins_segmentation.json'
rdet_ann_dir = '/diwang22/dataset/samrs/dior/Annotations/Oriented Bounding Boxes/'
seg_map_suffix = '.png'
rdet_post_fix = '.xml'
sior_trn_dataset = SIORMultiTaskDataset(data_root = data_root, data_prefix = data_prefix, 
                                        ann_file = ann_file, rdet_ann_dir = rdet_ann_dir,
                                        seg_map_suffix = seg_map_suffix, rdet_post_fix = rdet_post_fix, 
                                        pipeline = train_pipeline)

data_root = '/diwang22/dataset/samrs/dior'
data_prefix = dict(img='JPEGImages-trainval/', seg='hbox_segs_trainvaltest_init/gray')
ann_file = '/diwang22/dataset/samrs/dior/sior_valid_ins_segmentation.json'
rdet_ann_dir = '/diwang22/dataset/samrs/dior/Annotations/Oriented Bounding Boxes/'
seg_map_suffix = '.png'
rdet_post_fix = '.xml'
sior_val_dataset = SIORMultiTaskDataset(data_root = data_root, data_prefix = data_prefix, 
                                        ann_file = ann_file, rdet_ann_dir = rdet_ann_dir,
                                        seg_map_suffix = seg_map_suffix, rdet_post_fix = rdet_post_fix, 
                                        pipeline = valid_pipeline)

#################### FAST

data_root = '/diwang22/dataset/samrs/fair1m_1024'
data_prefix = dict(img='trainval/images', seg='trainval/rhbox_segs_init/gray')
ann_file = '/diwang22/dataset/samrs/fair1m_1024/trainval/fast_train_ins_segmentation.json'
rdet_ann_dir = '/diwang22/dataset/samrs/fair1m_1024/trainval/rbbtxts/'
seg_map_suffix = '.png'
rdet_post_fix = '.txt'
fast_trn_dataset = FASTMultiTaskDataset(data_root = data_root, data_prefix = data_prefix, 
                                        ann_file = ann_file, rdet_ann_dir = rdet_ann_dir,
                                        seg_map_suffix = seg_map_suffix, rdet_post_fix = rdet_post_fix, 
                                        pipeline = train_pipeline)

data_root = '/diwang22/dataset/samrs/fair1m_1024'
data_prefix = dict(img='trainval/images', seg='trainval/rhbox_segs_init/gray')
ann_file = '/diwang22/dataset/samrs/fair1m_1024/trainval/fast_valid_ins_segmentation.json'
rdet_ann_dir = '/diwang22/dataset/samrs/fair1m_1024/trainval/rbbtxts/'
seg_map_suffix = '.png'
rdet_post_fix = '.txt'
fast_val_dataset = FASTMultiTaskDataset(data_root = data_root, data_prefix = data_prefix, 
                                        ann_file = ann_file, rdet_ann_dir = rdet_ann_dir,
                                        seg_map_suffix = seg_map_suffix, rdet_post_fix = rdet_post_fix, 
                                        pipeline = valid_pipeline)

################################## sampler

if args.distributed=='True':
    train_sampler_sota = D.distributed.DistributedSampler(sota_trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
    train_sampler_sior = D.distributed.DistributedSampler(sior_trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
    train_sampler_fast = D.distributed.DistributedSampler(fast_trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
else:
    train_sampler_sota = None
    train_sampler_sior = None
    train_sampler_fast = None

if args.distributed=='True':
    val_sampler_sota = D.distributed.DistributedSampler(sota_val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
    val_sampler_sior = D.distributed.DistributedSampler(sior_val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
    val_sampler_fast = D.distributed.DistributedSampler(fast_val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
else:
    val_sampler_sota = None
    val_sampler_sior = None
    val_sampler_fast = None

############################### batch size 

batch_size = args.batch_size
batch_size_val = args.batch_size_val
workers = args.workers

if args.batch_mode == 'ratio':
    all_img_num = 0
    if 'sota' in args.datasets:
        all_img_num += 17480
        logger.info('Using SOTA dataset!')
    if 'sior' in args.datasets:
        all_img_num += 11725
        logger.info('Using SIOR dataset!')
    if 'fast' in args.datasets:
        all_img_num += 64147
        logger.info('Using FAST dataset!')

    if 'sota' in args.datasets:
        batch_size_sota = int(batch_size * 17480 * 1.0 / all_img_num)
        batch_size_val_sota = int(batch_size_val * 17480 * 1.0 / all_img_num)
        workers_sota = int(workers * 17480 * 1.0 / all_img_num)
    else:
        batch_size_sota = 1
        batch_size_val_sota = 1
        workers_sota = 1

    if 'sior' in args.datasets:
        batch_size_sior = int(batch_size * 11725 * 1.0 / all_img_num)
        batch_size_val_sior = int(batch_size_val * 11725 * 1.0 / all_img_num)
        workers_sior = int(workers * 11725 * 1.0 / all_img_num)
    else:
        batch_size_sior = 1
        batch_size_val_sior = 1
        workers_sior = 1

    if 'fast' in args.datasets:
        batch_size_fast = int(batch_size * 64147 * 1.0 / all_img_num)
        batch_size_val_fast = int(batch_size_val * 64147 * 1.0 / all_img_num)
        workers_fast = int(workers * 64147 * 1.0 / all_img_num)
    else:
        batch_size_fast = 1
        batch_size_val_fast = 1
        workers_fast = 1
elif args.batch_mode == 'avg':
    dataset_cnt = len(args.datasets)
    avg_batch = int(batch_size / dataset_cnt)
    avg_batch_val = int(batch_size_val / dataset_cnt)
    avg_workers = int(workers / dataset_cnt)

    if 'sota' in args.datasets:
        logger.info('Using SOTA dataset!')
        batch_size_sota = avg_batch
        batch_size_val_sota = avg_batch_val
        workers_sota = avg_workers
    else:
        batch_size_sota = 1
        batch_size_val_sota = 1
        workers_sota = 1

    if 'sior' in args.datasets:
        logger.info('Using SIOR dataset!')
        batch_size_sior = avg_batch
        batch_size_val_sior = avg_batch_val
        workers_sior = avg_workers
    else:
        batch_size_sior = 1
        batch_size_val_sior = 1
        workers_sior = 1

    if 'fast' in args.datasets:
        logger.info('Using FAST dataset!')
        batch_size_fast = avg_batch
        batch_size_val_fast = avg_batch_val
        workers_fast = avg_workers
    else:
        batch_size_fast = 1
        batch_size_val_fast = 1
        workers_fast = 1

else:
    raise NotImplementedError

########## when distributed

# if args.distributed == 'True':
#     batch_size_sota = int(batch_size_sota / args.world_size)#将一个节点的BS按GPU平分
#     batch_size_sior = int(batch_size_sior / args.world_size)
#     batch_size_fast = int(batch_size_fast / args.world_size)

#     batch_size_val_sota = int(batch_size_val_sota / args.world_size)
#     batch_size_val_sior = int(batch_size_val_sior / args.world_size)
#     batch_size_val_fast = int(batch_size_val_fast / args.world_size)

#     workers_sota = int(workers_sota / args.world_size)
#     workers_sior = int(workers_sior / args.world_size)
#     workers_fast = int(workers_fast / args.world_size)

print(sota_trn_dataset.length)

trn_loader_length = np.min([sota_trn_dataset.length / (batch_size_sota * args.world_size),
                          sior_trn_dataset.length / (batch_size_sior * args.world_size),
                          fast_trn_dataset.length / (batch_size_fast * args.world_size)])

val_loader_length = np.min([sota_val_dataset.length / (batch_size_val_sota),
                          sior_val_dataset.length / (batch_size_val_sior),
                          fast_val_dataset.length / (batch_size_val_fast)])

if main_process(args):
    logger.info('train data length: {}, {}, {}'.format(sota_trn_dataset.length, sior_trn_dataset.length, fast_trn_dataset.length))
    logger.info('train batch size: {}, {}, {}'.format(batch_size_sota*args.world_size, batch_size_sior*args.world_size, batch_size_fast*args.world_size))
    logger.info('train loader length: {}'.format(trn_loader_length))

    logger.info('valid data length: {}, {}, {}'.format(sota_val_dataset.length, sior_val_dataset.length, fast_val_dataset.length))
    logger.info('valid batch size: {}, {}, {}'.format(batch_size_val_sota, batch_size_val_sior, batch_size_val_fast))
    logger.info('valid loader length: {}'.format(val_loader_length))

################################### dataloaders

#from utils import mmengine_collate

train_loader_sota = D.DataLoader(
    sota_trn_dataset, batch_size=batch_size_sota, shuffle=(train_sampler_sota is None), 
    num_workers=workers_sota, pin_memory=True, 
    sampler=train_sampler_sota, drop_last=True)
train_loader_sior = D.DataLoader(
    sior_trn_dataset, batch_size=batch_size_sior, shuffle=(train_sampler_sior is None), 
    num_workers=workers_sior, pin_memory=True,
    sampler=train_sampler_sior, drop_last=True)
train_loader_fast = D.DataLoader(
    fast_trn_dataset, batch_size=batch_size_fast, shuffle=(train_sampler_fast is None), 
    num_workers=workers_fast, pin_memory=True,
    sampler=train_sampler_fast, drop_last=True)

valid_loader_sota = D.DataLoader(
    sota_val_dataset, batch_size=batch_size_val_sota, shuffle=False, 
    num_workers=workers_sota, pin_memory=True,
    sampler=val_sampler_sota)

valid_loader_sior = D.DataLoader(
    sior_val_dataset, batch_size=batch_size_val_sior, shuffle=False, 
    num_workers=workers_sior, pin_memory=True,
    sampler=val_sampler_sior)

valid_loader_fast = D.DataLoader(
    fast_val_dataset, batch_size=batch_size_val_fast, shuffle=False, 
    num_workers=workers_fast, pin_memory=True,
    sampler=val_sampler_fast)

##################################################### model #####################################################


from models import MutliTaskPretrnFramework

model = MutliTaskPretrnFramework(args, classes1=classes1, classes2=classes2, 
                                 classes3=classes3, batch_augments = batch_augments)

model.cuda(LOCAL_RANK)

losses = []

if 'ss' in args.tasks:
    logger.info('Implementing Sementic Segmentation Task!')
if 'is' in args.tasks:
    logger.info('Implementing Instance Segmentation Task!')
if 'rd' in args.tasks:
    logger.info('Implementing Rotated Detection Task!')

#####################################################  optimizer #####################################################


if 'vit_' in args.backbone:
    from mmengine.optim import build_optim_wrapper
    from mmcv_custom.layer_decay_optimizer_constructor_vit import *
    # AdamW optimizer, no weight decay for position embedding & layer norm in backbone

    if 'vit_b_' in args.backbone:

        optim_wrapper = dict(
            optimizer=dict(
            type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
            constructor='LayerDecayOptimizerConstructor_ViT', 
            paramwise_cfg=dict(
                num_layers=12, 
                layer_decay_rate=0.9,
                )
                )
        
    elif 'vit_l_' in args.backbone:

        optim_wrapper = dict(
            optimizer=dict(
            type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
            constructor='LayerDecayOptimizerConstructor_ViT', 
            paramwise_cfg=dict(
                num_layers=24, 
                layer_decay_rate=0.9,
                )
                )
        
    else:
        raise NotImplementedError

    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.end_iter, eta_min=0, last_epoch=-1)


elif 'internimage' in args.backbone:
    from mmengine.optim import build_optim_wrapper
    from mmcv_custom.custom_layer_decay_optimizer_constructor import *
    # classification & segmentation
    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05),
        constructor='CustomLayerDecayOptimizerConstructor_InternImage',
        paramwise_cfg=dict(num_layers=39, 
                        layer_decay_rate=0.94,
                        depths=[5, 5, 24, 5]
                        )
                        )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.end_iter, eta_min=0, last_epoch=-1)


## ft
if args.ft=='True':
    if os.path.isfile(args.resume):
        if main_process(args):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        if main_process(args):
            logger.info("=> loading ft model...")
        args.start_epoch = checkpoint['epoch']
        args.start_iter = checkpoint['iteration']
        ckpt_dict = checkpoint['state_dict']
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in ckpt_dict.items():
            if k in state_dict: #and k!='module.GraphReason.conv.0.block1.0.weight':
                model_dict[k] = v
        state_dict.update(model_dict)
        msg = model.load_state_dict(state_dict,strict=False)
        print(msg)
        #model.load_state_dict(checkpoint['state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losses = checkpoint['loss_pretrain'].tolist()
        if main_process(args):
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        if main_process(args):
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

### 分布式
if args.distributed == 'True':
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],find_unused_parameters=True)

    if args.use_ckpt == 'True':
        model._set_static_graph()

    if main_process(args):
        logger.info("Implementing distributed training!")
    seed = 2023 + LOCAL_RANK
    set_seeds(seed)
else:
    model = torch.nn.DataParallel(model)#普通的单机多卡
    patch_replication_callback(model)

    if main_process(args):
        logger.info("Implementing parallel training!")

##################################################### validation #####################################################

from semantic_segmentation.metric import MTP_SS_Metric
from instance_segmentation.metric import MTP_IS_Metric
from rotated_detection.metric import MTP_RD_Metric

ss_metric1 = MTP_SS_Metric(ignore_index=255, iou_metrics=['mIoU'])
ss_metric2 = MTP_SS_Metric(ignore_index=255, iou_metrics=['mIoU'])
ss_metric3 = MTP_SS_Metric(ignore_index=255, iou_metrics=['mIoU'])

ss_metric1.dataset_meta = valid_loader_sota.dataset.metainfo
ss_metric2.dataset_meta = valid_loader_sior.dataset.metainfo
ss_metric3.dataset_meta = valid_loader_fast.dataset.metainfo

is_metric1 = MTP_IS_Metric(metric=['bbox','segm'], ann_file='/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/sota_rbb_valid_ins_segmentation.json')
is_metric2 = MTP_IS_Metric(metric=['bbox','segm'], ann_file='/diwang22/dataset/samrs/dior/sior_valid_ins_segmentation.json')
is_metric3 = MTP_IS_Metric(metric=['bbox','segm'], ann_file='/diwang22/dataset/samrs/fair1m_1024/trainval/fast_valid_ins_segmentation.json')

is_metric1.dataset_meta = valid_loader_sota.dataset.metainfo
is_metric2.dataset_meta = valid_loader_sior.dataset.metainfo
is_metric3.dataset_meta = valid_loader_fast.dataset.metainfo

rd_metric1 = MTP_RD_Metric(metric = 'mAP', predict_box_type='rbox')
rd_metric2 = MTP_RD_Metric(metric = 'mAP', predict_box_type='rbox')
rd_metric3 = MTP_RD_Metric(metric = 'mAP', predict_box_type='rbox')

rd_metric1.dataset_meta = valid_loader_sota.dataset.metainfo
rd_metric2.dataset_meta = valid_loader_sior.dataset.metainfo
rd_metric3.dataset_meta = valid_loader_fast.dataset.metainfo



@torch.no_grad()
def validation(args, logger, epoch, model, valid_loader_sota, valid_loader_sior, valid_loader_fast):

    model.eval()

    eval_length_sota = eval_length_sior = eval_length_fast = 0

    for (i, (id1, id2, id3)) in enumerate(zip(valid_loader_sota, valid_loader_sior, valid_loader_fast)):


        datainfo_list1 = parse_datainfos(sota_val_dataset, id1)
        datainfo_list2 = parse_datainfos(sior_val_dataset, id2)
        datainfo_list3 = parse_datainfos(fast_val_dataset, id3)

        # datainfo_list1 = sota_val_dataset.filter_data(datainfo_list1)
        # datainfo_list2 = sior_val_dataset.filter_data(datainfo_list2)
        # datainfo_list3 = fast_val_dataset.filter_data(datainfo_list3)

        x1 = data_augs(sota_val_dataset, datainfo_list1)
        x2 = data_augs(sior_val_dataset, datainfo_list2)
        x3 = data_augs(fast_val_dataset, datainfo_list3)

        outputs = model.forward(x1, x2, x3)

        output1, output2, output3 = outputs

        if 'ss' in args.tasks:

            ss_metric1.process(data_batch=x1, data_samples = output1[0])
            ss_metric2.process(data_batch=x2, data_samples = output1[1])
            ss_metric3.process(data_batch=x3, data_samples = output1[2])

        if 'is' in args.tasks:

            is_metric1.process(data_batch=x1, data_samples = output2[0])
            is_metric2.process(data_batch=x2, data_samples = output2[1])
            is_metric3.process(data_batch=x3, data_samples = output2[2])

        if 'rd' in args.tasks:

            rd_metric1.process(data_batch=x1, data_samples = output3[0])
            rd_metric2.process(data_batch=x2, data_samples = output3[1])
            rd_metric3.process(data_batch=x3, data_samples = output3[2])

        eval_length_sota += len(x1['inputs'])
        eval_length_sior += len(x2['inputs'])
        eval_length_fast += len(x3['inputs'])

        if main_process(args):

            logger.info('Valid epoch {}, sample {}/{}'.format(epoch, i, val_loader_length))

    ss_res1 = ss_res2 = ss_res3 = -1
    is_res1 = is_res2 = is_res3 = -1
    rd_res1 = rd_res2 = rd_res3 = -1

    if 'ss' in args.tasks:

        ss_res1 = ss_metric1.evaluate(eval_length_sota)
        ss_res2 = ss_metric2.evaluate(eval_length_sior)
        ss_res3 = ss_metric3.evaluate(eval_length_fast)

        if main_process(args):

            logger.info('SS mIOU: SOTA {:.2f}, SIOR {:.2f}, FAST: {:.2f}'.format(ss_res1['mIoU'], ss_res2['mIoU'], ss_res3['mIoU']))

    if 'is' in args.tasks:

        is_res1 = is_metric1.evaluate(eval_length_sota)
        is_res2 = is_metric2.evaluate(eval_length_sior)
        is_res3 = is_metric3.evaluate(eval_length_fast)

        if main_process(args):

            logger.info('IS bbox mAP: SOTA {:.2f}, SIOR {:.2f}, FAST: {:.2f}'.format(is_res1['coco/bbox_mAP'], is_res2['coco/bbox_mAP'], is_res3['coco/bbox_mAP']))
            logger.info('IS segm mAP: SOTA {:.2f}, SIOR {:.2f}, FAST: {:.2f}'.format(is_res1['coco/segm_mAP'], is_res2['coco/segm_mAP'], is_res3['coco/segm_mAP']))

    if 'rd' in args.tasks:

        rd_res1 = rd_metric1.evaluate(eval_length_sota)
        rd_res2 = rd_metric2.evaluate(eval_length_sior)
        rd_res3 = rd_metric3.evaluate(eval_length_fast)

        if main_process(args):

            logger.info('RD rbox mAP: SOTA {:.2f}, SIOR {:.2f}, FAST: {:.2f}'.format(rd_res1['dota/mAP'], rd_res2['dota/mAP'], rd_res3['dota/mAP']))

    acc = 0
    task_cnt = 0

    if 'ss' in args.tasks:

        acc += ss_res1['mIoU'] + ss_res2['mIoU'] + ss_res3['mIoU']
        task_cnt += 3
    
    if 'is' in args.tasks:

        acc += is_res1['coco/segm_mAP'] + is_res2['coco/segm_mAP'] + is_res3['coco/segm_mAP']
        task_cnt += 3

    if 'rd' in args.tasks:

        acc += rd_res1['dota/mAP'] + rd_res2['dota/mAP'] + rd_res3['dota/mAP']
        task_cnt += 3

    acc = acc / (task_cnt * 1.0)

    if main_process(args):

        logger.info('Valid epoch {}, MTP Average accuracy: {:.4f}'.format(epoch, acc))

    model.train()

    return acc


##################################################### training #####################################################

best_acc = 0

iter = args.start_iter
epoch = args.start_epoch
tasks_str = ''

for task in args.tasks:
    tasks_str += str(task)+'_'


if args.mixed_precision == 'True':
    scaler = GradScaler()

while True:

    if args.distributed == 'True':
        train_sampler_sota.set_epoch(epoch)
        train_sampler_sior.set_epoch(epoch)
        train_sampler_fast.set_epoch(epoch)

    start_time = time.time()
    model.train()

    optimizer.zero_grad()

    for (id1, id2, id3) in zip(train_loader_sota, train_loader_sior, train_loader_fast):

        datainfo_list1 = parse_datainfos(sota_trn_dataset, id1)
        datainfo_list2 = parse_datainfos(sior_trn_dataset, id2)
        datainfo_list3 = parse_datainfos(fast_trn_dataset, id3)

        # datainfo_list1 = sota_trn_dataset.filter_data(datainfo_list1)
        # datainfo_list2 = sior_trn_dataset.filter_data(datainfo_list2)
        # datainfo_list3 = fast_trn_dataset.filter_data(datainfo_list3)

        x1 = data_augs(sota_trn_dataset, datainfo_list1)
        x2 = data_augs(sior_trn_dataset, datainfo_list2)
        x3 = data_augs(fast_trn_dataset, datainfo_list3)

        loss = 0

        loss_ss_datasets = loss_is_datasets = loss_rd_datasets = [-1, -1, -1]

        loss_ = ''

        loss_datasets = model.forward(x1, x2, x3)

        loss1, loss2, loss3 = loss_datasets

        if 'ss' in args.tasks:

            loss_ss, loss_ss_datasets = parse_losses(loss1)

        else:
            loss_ss = loss1

        loss += loss_ss
        loss_ += '|SS_SOTA: {:.2f}, SS_SIOR: {:.2f}, SS_FAST: {:.2f}|'.format(loss_ss_datasets[0], loss_ss_datasets[1], loss_ss_datasets[2]) 

        if 'is' in args.tasks:

            loss_is, loss_is_datasets = parse_losses(loss2)

        else:
            loss_is = loss2

        loss +=  loss_is
        loss_ += '|IS_SOTA: {:.2f}, IS_SIOR: {:.2f}, IS_FAST: {:.2f}|'.format(loss_is_datasets[0], loss_is_datasets[1], loss_is_datasets[2]) 

        if 'rd' in args.tasks:

            loss_rd, loss_rd_datasets  = parse_losses(loss3)

        else:
            loss_rd = loss3

        loss +=  loss_rd
        loss_ += '|RD_SOTA: {:.2f}, RD_SIOR: {:.2f}, RD_FAST: {:.2f}|'.format(loss_rd_datasets[0], loss_rd_datasets[1], loss_rd_datasets[2]) 

        iter +=1

        if args.mixed_precision == 'True':

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            old_scaler = scaler.get_scale()
            # Updates the scale for next iteration.
            scaler.update()
            new_scaler = scaler.get_scale()

            skip_lr_sched = (old_scaler > new_scaler)

            torch.cuda.synchronize()

        else:
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            torch.cuda.synchronize()
        
        losses.append(loss.item())

        if main_process(args):
            logger.info('Train epoch {} iter {}/{}, lr: {:.7f} loss: {} sum: {:.2f}'.format(epoch+1, iter, args.end_iter, optimizer.param_groups[0]["lr"], loss_, loss.item()))

        
        if iter % args.interval == 0:

            # logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

            # start_time = time.time()
            # vacc = validation(args, logger, epoch, model, valid_loader_sota, valid_loader_sior, valid_loader_fast)
            # end_time = time.time()

            # logger.info('Validation epoch {}, iter [{}/{}]: Average mIoU {:.2f}. Cost {:.2f} secs'.format(epoch+1, iter, args.end_iter, vacc, end_time-start_time))

            # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

            # if vacc > best_acc:
            #     best_acc = vacc
            #     if main_process(args):
            #         filename = args.save_path + '/best_{}_{}pretrn_model.pth'.format(args.backbone, tasks_str)
            #         logger.info('Saving epoch {} checkpoint to: {}'.format(epoch,filename))
            #         torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
            #                     filename)
            #         filename = args.save_path + '/best_{}_{}pretrn_model_encoder.pth'.format(args.backbone, tasks_str)
            #         torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
            #                     filename)            
            #     if main_process(args):
            #         print("best acc is {}".format(best_acc))

            logger.info('>>>>>>>>>>>>>>>> Save model trained on iter {} >>>>>>>>>>>>>>>>'.format(iter))

            if main_process(args):
                filename = args.save_path + '/Iter_{}_{}_{}pretrn_model.pth'.format(iter, args.backbone, tasks_str)
                torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                                    filename)
                filename = args.save_path + '/Iter_{}_{}_{}pretrn_model_encoder.pth'.format(iter, args.backbone, tasks_str)
                torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                                        filename)

        
        scheduler.step()

        if iter >= args.end_iter:
            break

    if iter >= args.end_iter:
        break

    epoch +=1

########### last validation

logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

start_time = time.time()
vacc = validation(args, logger, epoch, model, valid_loader_sota, valid_loader_sior, valid_loader_fast)
end_time = time.time()

logger.info('Last: validation epoch {}, iter [{}/{}]: Average Acc {:.2f}. Cost {:.2f} secs'.format(epoch+1, iter, args.end_iter, vacc, end_time-start_time))

logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

if main_process(args):
    print("last acc is {}".format(vacc))

if main_process(args):            
    filename = args.save_path + '/last_{}_{}pretrn_model.pth'.format(args.backbone, tasks_str)
    torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                        filename)
    filename = args.save_path + '/last_{}_{}pretrn_model_encoder.pth'.format(args.backbone, tasks_str)
    torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                            filename)

logger.info('################# Pretrain model save finished! ######################')