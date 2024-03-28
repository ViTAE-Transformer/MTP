<h1 align="center"> MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining </h1>

<p align="center">
<a href="https://arxiv.org/abs/2403.13430"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<h5 align="center"><em>Di Wang, Jing Zhang, Minqiang Xu, Lin Liu, Dongsheng Wang, Erzhong Gao, Chengxi Han, Haonan Guo,  Bo Du, 

Dacheng Tao and Liangpei Zhang</em></h5>

<p align="center">
  <a href="#🔥-update">Update</a> |
  <a href="#🌞-overview">Overview</a> |
  <a href="#📖-datasets-and-models">Datasets and Models</a> |
  <a href="#🛠️-usage">Usage</a> |
  <a href="#🎺-statement">Statement</a>
</p >

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/image-classification-on-eurosat)](https://paperswithcode.com/sota/image-classification-on-eurosat?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/aerial-scene-classification-on-nwpu-20-as)](https://paperswithcode.com/sota/aerial-scene-classification-on-nwpu-20-as?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/object-detection-in-aerial-images-on-xview)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-xview?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/object-detection-in-aerial-images-on-dior)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dior?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/object-detection-in-aerial-images-on-dior-r)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dior-r?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/object-detection-in-aerial-images-on-fair1m-2)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-fair1m-2?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/oriented-object-detction-on-dota-2-0)](https://paperswithcode.com/sota/oriented-object-detction-on-dota-2-0?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/semantic-segmentation-on-spacenet-1)](https://paperswithcode.com/sota/semantic-segmentation-on-spacenet-1?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/semantic-segmentation-on-loveda)](https://paperswithcode.com/sota/semantic-segmentation-on-loveda?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/change-detection-on-oscd-3ch)](https://paperswithcode.com/sota/change-detection-on-oscd-3ch?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/change-detection-on-whu-building-dataset)](https://paperswithcode.com/sota/change-detection-on-whu-building-dataset?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/change-detection-on-levir-cd)](https://paperswithcode.com/sota/change-detection-on-levir-cd?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/building-change-detection-for-remote-sensing)](https://paperswithcode.com/sota/building-change-detection-for-remote-sensing?p=mtp-advancing-remote-sensing-foundation-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mtp-advancing-remote-sensing-foundation-model/change-detection-for-remote-sensing-images-on)](https://paperswithcode.com/sota/change-detection-for-remote-sensing-images-on?p=mtp-advancing-remote-sensing-foundation-model)

## 🚩 Current applications

> **Remote Sensing Related Works: Please see [Remote Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)**;

> **Remote Sensing Supervised Pretraining Foundation Model: Please see [RSP](https://github.com/ViTAE-Transformer/RSP)**;

> **100M-parameter Remote Sensing Unsupervised Pretraining Foundation Model: Please see [RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)**;

> **Large-Scale RS Segmentation Pretraining Dataset: Please see [SAMRS](https://github.com/ViTAE-Transformer/SAMRS)**;

> Other applications: [ViTAE](https://github.com/ViTAE-Transformer/ViTAE-Transformer) | [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA) | [QFormer](https://github.com/ViTAE-Transformer/QFormer) | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | [Matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting) | [Scene Text Spotting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Scene-Text-Detection)  | [Video Object Segmentation](https://github.com/ViTAE-Transformer/VOS-LLB)

# 🔥 Update

**2024.03.28**

- The horizontal detection finetuned models are released!

**2024.03.27**

- The classification finetuned models are released!

**2024.03.26**

- The pretrained models are released!

**2024.03.25**

- The SOTA-RBB set of the pretraining dataset is uploaded to [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgckJ0Xip2yD0y9HD_Q?e=ErZXPS) and [Baidu](https://pan.baidu.com/s/15N7DoZj53cIXEDKw6hzD4A?pwd=q9u6)!

**2024.03.21**

- The paper is post on arxiv!

# 🌞 Overview

This is the official repository of the paper: <a href="https://arxiv.org/abs/2403.13430">  MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining </a>

<figure>
<img src="Figs/pipeline.png">
<figcaption align = "center"><b>Figure 1: The overall pipeline of MTP. 
 </b></figcaption>
</figure>


In this study, we explore the Multi-Task Pretraining (MTP) paradigm for RS foundation models. Using a shared encoder and task-specific decoder architecture, we conduct multi-task supervised pretraining on the SAMRS dataset, encompassing semantic segmentation, instance segmentation, and rotated object detection. MTP supports both convolutional neural networks and vision transformer foundation models with over 300 million parameters. The pretrained models are finetuned on various RS downstream tasks, such as scene classification, horizontal and rotated object detection, semantic segmentation, and change detection. We hope this research encourages further exploration of RS foundation models and anticipate the widespread application of these models across diverse fields of RS image interpretation.

# 📖 Datasets and Models

## Pretraining Dataset

We clip the DOTA-2.0 rotated bounding box version and produce the segmentation label by SAM, obtaining **SOTA-RBB**.
(original SAMRS uses DOTA-2.0 horizontal bounding box version)

**SOTA-RBB and the SIOR and FAST of original SAMRS** is together used for implementing MTP.

We have uploaded SOTA-RBB to [OneDive](https://1drv.ms/f/s!AimBgYV7JjTlgckJ0Xip2yD0y9HD_Q?e=ErZXPS) and [Baidu](https://pan.baidu.com/s/15N7DoZj53cIXEDKw6hzD4A?pwd=q9u6).

## Pretrained Models

| Pretrain | Pretraining Dataset | Backbone | Backbone Weights | Model Weights |
| :------- | :------: | :------ | :-----: | :-----: |
| MAE | Million-AID | ViT-L | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) | - |
| MAE + MTP | [SAMRS](https://github.com/ViTAE-Transformer/SAMRS) | ViT-B + RVSA | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) |
| MAE + MTP | [SAMRS](https://github.com/ViTAE-Transformer/SAMRS) | ViT-L + RVSA | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) |
| IMP + MTP | [SAMRS](https://github.com/ViTAE-Transformer/SAMRS)| InternImage-XL | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) | [Baidu](https://pan.baidu.com/s/1Zh6yv2AouboGEP4phyR7xA?pwd=yqv9) & [OneDrive](https://1drv.ms/f/s!AimBgYV7JjTlgcpa7t2sywuWOm3HQA?e=LAh8WN) |

## Finetuned Models

### Classification

| Pretrain | Dataset | Backbone | OA | Config | Log | Weights |
| :------- | :------ | :------ | :-----: | :-----: |:-----: | :-----: |
| MAE + MTP | EuroSAT| ViT-B ＋ RVSA | 98.76 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1w4ORVO4Q4XGzOJY0NMBy8Q?pwd=jsoc) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gTwi2qBsPuiSrgrL?e=utqMdY) |
| MAE + MTP | EuroSAT| ViT-L ＋ RVSA | 98.78 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1w4ORVO4Q4XGzOJY0NMBy8Q?pwd=jsoc) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gTwi2qBsPuiSrgrL?e=utqMdY) |
| IMP + MTP | EuroSAT| InternImage-XL | 99.24 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1w4ORVO4Q4XGzOJY0NMBy8Q?pwd=jsoc) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gTwi2qBsPuiSrgrL?e=utqMdY) |
| MAE + MTP | RESISC-45| ViT-B ＋ RVSA | 95.57 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1w4ORVO4Q4XGzOJY0NMBy8Q?pwd=jsoc) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gTwi2qBsPuiSrgrL?e=utqMdY) |
| MAE + MTP | RESISC-45| ViT-L ＋ RVSA | 95.88 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1w4ORVO4Q4XGzOJY0NMBy8Q?pwd=jsoc) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gTwi2qBsPuiSrgrL?e=utqMdY)  |
| IMP + MTP | RESISC-45| InternImage-XL | 96.27 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1w4ORVO4Q4XGzOJY0NMBy8Q?pwd=jsoc) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gTwi2qBsPuiSrgrL?e=utqMdY)  |

### Horizontal Object Detection

| Pretrain | Dataset | Backbone | Method | AP50 | Config | Log | Weights |
| :------- | :------ | :------ | :----- | :-----: | :-----: |:-----: | :-----: |
| MAE + MTP | Xview | ViT-B ＋ RVSA | RetinaNet | 16.40 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw) |
| MAE + MTP | Xview | ViT-L ＋ RVSA | RetinaNet| 19.40 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw) |
| IMP + MTP | Xview | InternImage-XL| RetinaNet | 18.20 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw) |
| MAE + MTP | DIOR | ViT-B ＋ RVSA | Faster-RCNN | 79.00 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw)|
| MAE + MTP | DIOR | ViT-L ＋ RVSA | Faster-RCNN | 81.60 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw)  |
| IMP + MTP | DIOR | InternImage-XL | Faster-RCNN | 77.50 | Coming Soon | Coming Soon | [Baidu](https://pan.baidu.com/s/1yiJISQYg0Xl84PvZr_r84w?pwd=ag0x) & [OneDrive](https://1drv.ms/f/s!AiSncQLqo7V6gUNIOKO-VtlKyT4d?e=NXA4Nw)  |


# 🛠️ Usage

## Environment

#### This environment adopts new version OpenMMLab series to support multi-task pretraining and finetuning on various RS tasks.

| Package | Version | Package | Version | Package | Version |Package | Version |
| ----- | :-----: | ----- | :-----: | ----- | :-----: | ----- | :-----: |
| Python | 3.8.17 | timm | 0.9.5 | MMEngine | 0.8.4 | MMDetection | 3.1.0 |
| Pytorch | 1.10.0 | OpenCV | 4.8.0 | MMPretrain | 1.2.0| MMRotate | 1.0.0rc1
| Torchvision | 0.10.0 | MMCV | 2.0.0 | MMSegmentation |1.0.0 | Open-CD | 1.1.0 |

#### ❗❗❗ We also configure an environment for MMRotate 0.3.4

| Package | Version | Package | Version | Package | Version |
| ----- | :-----: | ----- | :-----: | ----- | :-----: |
| Python | 3.8.0 | timm | 0.9.2 | MMEngine | 0.10.3 | 
| Pytorch | 1.10.0 | OpenCV | 4.7.0 | MMDetection | 2.28.2 |
| Torchvision | 0.10.0 | MMCV-full | 1.6.1 | MMRotate | 0.3.4 |

This environment is used for multi-scale prediction of FAIR1M-2.0 and DOTA-V1.0.

## Preparing Pretraining Dataset

1. Download SOTA-RBB and the SIOR and FAST sets from SAMRS dataset.

2. Transform the *.pkl in SAMRS dataset to COCO *.json.

    ```
    python scripts/convert_pkl_json.py
    ```

## Performing Multi-Task Pretraining

We conduct the MTP with SLURM. This is an example of pretraining ViT-L + RVSA:

```
srun -J mtp -p gpu --gres=dcu:4 --ntasks=32 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain.py \
    --backbone 'vit_l_rvsa' --tasks 'ss' 'is' 'rd' \
    --datasets 'sota' 'sior' 'fast' \
    --batch_size 3 --batch_size_val 3 --workers 8 \
    --save_path [folder path of saved model] \
    --distributed 'True' --end_iter 80000 \
    --image_size 448 --init_backbone 'mae' --port '16003' --batch_mode 'avg' --background 'True' --use_ckpt 'True' --interval 5000
```
The training can be recovered by setting `--ft` and `--resume`
```
--ft 'True' --resume [path of saved multi-task pretrained model]
```

## Preparing Finetuning Dataset

**For Xview**: using `scripts/prepare_xview_dataset.py`, it contains the following functions:

* Transform geojson to labels in yolo format
* Divide training and testing sets
* Clip images and yolo format labels
* Transform yolo format labels to COCO format *.json

**For DIOR**: transform *.xml to COCO *.json format for feeding into MMDetection

```
python scripts/dior_h_2_coco.py
```

**For FAIR1M**: transform *.txt in DOTA format to required *.xml for submitting

```
python scripts/dota_submit_txt_to_fair1m_xml.py --txt_dir [path of *.txt]
```

**For SpaceNetv1**: extracting segmentation mask from geojson

```
python scripts/process_spacenet.py
```


## Finetuning on Various RS tasks

**Except for the rotated detection, we perform the finetuning on the SLURM. Here are examples:**

### Scene Classification (using MMPretrain)

Training and Validation on EuroSAT using MAE + MTP pretrained ViT-L + RVSA:

```
srun -J mmpretrn -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/vit-rvsa-l-224-mae-mtp_eurosat.py \
--work-dir=/diwang/work_dir/multitask_pretrain/finetune/classification/eurosat/vit-rvsa-l-224-mae-mtp_eurosat \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

### Horizontal Object Detection (using MMDetection)

Training on DIOR using Faster-RCNN with a backbone network of MAE + MTP pretrained ViT-L + RVSA:

```
srun -J mmdet -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py \
--work-dir=/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

Then testing and generating dection results:

```
srun -J mmdet -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py \
/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior/epoch_12.pth \
--work-dir=/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior/predict \
--show-dir=/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

### Rotated Object Detection (using MMRotate, running on both SLURM and GPU server)

**1. Running on SLURM:**


**(Using MMRotate 1.0.0rc1)** Training on DIOR-R using Oriented-RCNN with a backbone network of MAE + MTP pretrained ViT-L + RVSA:

```
srun -J mmrot -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr \
--launcher="slurm"
```

**(Using MMRotate 1.0.0rc1)** Testing on DIOR-R for evaluation and visualizing detection maps.

```
srun -J mmrot -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/oriented_rcnn_rvsa_l_800_mae_mtp_diorr.py \
/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr/epoch_12.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

**(Using MMRotate 0.3.4)** If the dataset is evaluated online, we use `--format-only`, here is an example of testing on FAIR1M-2.0 for submitting results and visualizing detection maps.

```
srun -J mmrot -p gpu --gres=dcu:4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/fair1m/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20.py \
/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/fair1mv2/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20/epoch_12.pth --format-only \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/fair1mv2/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20/predict/show \
--eval-options submission_dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/fair1mv2/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20/predict/submit \
--launcher="slurm"
```

**2. Running on GPU server:** 


**(Using MMRotate 1.0.0rc1)** Training on DOTA-2.0 using Oriented-RCNN with a backbone network of MAE + MTP pretrained ViT-L + RVSA:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=40002 --master_addr=1.2.3.4 \
tools/train.py configs/mtp/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20.py \
--work-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20
```
**(Using MMRotate 1.0.0rc1)** Single-scale testing on DOTA-2.0 for submitting online evaluation results and visualizing detection maps.
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mtp/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20.py \
/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20/epoch_40.pth \
--work-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20/test \
--show-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20/test/vis
```
**(Using MMRotate 0.3.4)** Multi-scale testing on DOTA-V1.0 for submitting online evaluation results and visualizing detection maps.
```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mtp/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10.py \
/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/epoch_12.pth --format-only \
--show-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/predict/show \
--eval-options submission_dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/predict/submit
```

### Semantic Segmentation (using MMSegmentation)

Training on SpaceNetv1 using UperNet with a backbone network of MAE + MTP pretrained ViT-L + RVSA:

```
srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1 \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

Testing on SpaceNetv1 for accuracy evaluation and generating prediction maps:

```
srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1.py \
/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/iter_80000.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```
**Online Evaluation**: Testing on LoveDA for submittting online evaluation results and generating prediction maps:

```
srun -J mmseg -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/loveda/rvsa-l-upernet-512-mae-mtp-loveda.py \
/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/iter_80000.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/predict \
--out=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/predict/submit \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

*Note: after inferencing, the predictions of LoveDA needs to manually reduce 1 to meet the requirement of evaluation site*

```
python scripts/change_loveda_label.py
```

### Change Detection (using Open-CD)

Training on WHU using UperNet with a backbone network of MAE + MTP pretrained ViT-L + RVSA:

```
srun -J opencd -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/whu/rvsa-l-unet-256-mae-mtp_whu.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

Testing for accuracy evaluation and generating prediction maps:

```
srun -J opencd -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/whu/rvsa-l-unet-256-mae-mtp_whu.py \
/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu/epoch_200.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

### Decoder Parameter Reusing

Take an example of reusing segmentation decoder in finetuning:

1. Change the keys of MTP saved weights:

    ```
    python scripts/change_ckpt.py
    ```

2. Then training with the revised weights
    ```
    srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
    python -u tools/train.py configs/mtp/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1.py \
    --work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1_reuse_decoder \
    --launcher="slurm" \
    --cfg-options 'find_unused_parameters'=True load_from=[path of the revised weights]
    ```
The remaining steps are the same as regular testing.

## 🎵 Citation

If you find MTP helpful, please consider giving this repo a ⭐ and citing:

```
@article{MTP,
  title={{MTP}: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining},
  author={Di Wang, Jing Zhang, Minqiang Xu, Lin Liu, Dongsheng Wang, Erzhong Gao, Chengxi Han, Haonan Guo, Bo Du, Dacheng Tao, Liangpei Zhang},
  journal={arXiv preprint arXiv:2403.13430},
  year={2024}
}
```

## 🎺 Statement

This project is for research purpose only. For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn).

## 💖 Thanks

* [segment-anything](https://github.com/facebookresearch/segment-anything), [BBoxToolkit](https://github.com/jbwang1997/BboxToolkit)
* [RSPrompter](https://github.com/KyanChen/RSPrompter), [InternImage](https://github.com/OpenGVLab/InternImage)
* [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine), [MMPretrain](https://github.com/open-mmlab/mmpretrain), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMRotate](https://github.com/open-mmlab/mmrotate), [Open-CD](https://github.com/likyoo/open-cd)

## 💡 Relevant Projects

[1] <strong>An Empirical Study of Remote Sensing Pretraining, IEEE TGRS, 2022</strong> | [Paper](https://ieeexplore.ieee.org/document/9782149) | [Github](https://github.com/ViTAE-Transformer/RSP)
<br><em>&ensp; &ensp; &ensp;Di Wang<sup>&#8727;</sup>, Jing Zhang<sup>&#8727;</sup>, Bo Du, Gui-Song Xia and Dacheng Tao</em>

[2] <strong>Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model, IEEE TGRS, 2022</strong> | [Paper](https://ieeexplore.ieee.org/document/9956816/) | [Github](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)
<br><em>&ensp; &ensp; &ensp;Di Wang<sup>&#8727;</sup>, Qiming Zhang<sup>&#8727;</sup>, Yufei Xu<sup>&#8727;</sup>, Jing Zhang, Bo Du, Dacheng Tao and Liangpei Zhang</em>

[3] <strong>SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model, NeurIPS Datasets and Benchmarks Track, 2023</strong> | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1be3843e534ee06d3a70c7f62b983b31-Abstract-Datasets_and_Benchmarks.html) | [Github](https://github.com/ViTAE-Transformer/SAMRS)
<br><em>&ensp; &ensp; &ensp;Di Wang<sup>&#8727;</sup>, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao and Liangpei Zhang</em>