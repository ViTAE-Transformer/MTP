import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
from mmengine.utils import is_list_of

def mmengine_collate(input_batch):
    images = []
    data_samples = []
    for sample in input_batch:
        images.append(sample['inputs'])
        data_samples.append(sample['data_samples'])

    return {'inputs':images, 'data_samples':data_samples}

def set_configs(size):

    # ins
    crop_size = (size, size)

    batch_augments = [
        dict(
            type='BatchFixedSizePad',
            size=crop_size,
            img_pad_value=0,
            pad_mask=True,
            mask_pad_value=0,
            pad_seg=False,
            seg_pad_value=255)
    ]

    train_pipeline = [
        dict(type='MTP_LoadImageFromFile', to_float32=True),
        dict(type='MTP_LoadAnnotations', with_bbox=True, with_rbox=True, 
            with_mask=True, with_seg=True, box_type = 'hbox', rbox_type='qbox',
            reduce_zero_label=False, ignore_index = 255),
        dict(type='ROD_ConvertBoxType', box_type_mapping=dict(gt_rboxes='rbox')),
        dict(
            type='MTP_RandomFlip',
            prob=0.75,
            direction=['horizontal', 'vertical', 'diagonal']),
        dict(
            type='MTP_RandomResize',
            scale=crop_size,
            ratio_range=(0.5, 2.0),
            resize_type='MTP_Resize',
            keep_ratio=True),
        dict(
            type='MTP_RandomCrop',
            crop_size=crop_size,
            crop_type='absolute',
            recompute_bbox=True,
            allow_negative_crop=True),
        dict(type='INS_FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
        dict(type='MTP_PhotoMetricDistortion'),
        dict(type='MTP_PackInputs')
    ]

    valid_pipeline = [
        dict(type='MTP_LoadImageFromFile', to_float32=True),
        dict(type='MTP_Resize', scale=crop_size, keep_ratio=True),
        # avoid bboxes being resized
        dict(type='MTP_LoadAnnotations', with_bbox=True, with_rbox=True, 
            with_mask=True, with_seg=True, box_type = 'hbox', rbox_type='qbox',
            reduce_zero_label=False, ignore_index = 255),
        dict(type='ROD_ConvertBoxType', box_type_mapping=dict(gt_rboxes='rbox')),
        dict(
            type='MTP_PackInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]

    return train_pipeline, valid_pipeline, batch_augments


def parse_datainfos(dataset, img_ids):

    data_list = []
    total_ann_ids = []
    
    for img_id in img_ids:
        img_id = int(img_id)
        raw_img_info = dataset.coco.load_imgs([img_id])[0] # 图片信息
        raw_img_info['img_id'] = img_id

        ann_ids = dataset.coco.get_ann_ids(img_ids=[img_id]) # 得到每张图包含的ann id
        raw_ann_info = dataset.coco.load_anns(ann_ids) # 这些ann id 对应的ann信息
        total_ann_ids.extend(ann_ids)

        parsed_data_info = dataset.parse_data_info({
            'raw_ann_info':
            raw_ann_info,
            'raw_img_info':
            raw_img_info
        })
        data_list.append(parsed_data_info)
    if dataset.ANN_ID_UNIQUE: # ann id必须是独一无二的
        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{dataset.ann_file}' are not unique!"

    #del self.coco
    
    return data_list

def data_augs(dataset, datainfos):

    data_samples = []

    for i in range(len(datainfos)):

        data_info = datainfos[i]

        data_sample = dataset.pipeline(data_info)

        while data_sample is None:

            idx = np.random.randint(0, dataset.length)

            raw_img_info = dataset.coco.load_imgs([dataset.img_ids[idx]])[0] # 图片信息
            raw_img_info['img_id'] = dataset.img_ids[idx]

            ann_ids = dataset.coco.get_ann_ids(img_ids=[dataset.img_ids[idx]]) # 得到每张图包含的ann id
            raw_ann_info = dataset.coco.load_anns(ann_ids) # 这些ann id 对应的ann信息

            parsed_data_info = dataset.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })

            data_sample = dataset.pipeline(parsed_data_info)

        data_samples.append(data_sample)

    data_samples = mmengine_collate(data_samples)

    return data_samples


def parse_losses(losses: List[Dict[str, torch.Tensor]]):
    """Parses the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: There are two elements. The first is the
        loss tensor passed to optim_wrapper which may be a weighted sum
        of all losses, and the second is log_vars which will be sent to
        the logger.
    """

    loss_sum = 0
    loss_datasets = []

    for loss_dataset in losses:
        log_vars = []
        for loss_name, loss_value in loss_dataset.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                        sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)

        loss_sum += loss

        loss_datasets.append(loss)

    return loss_sum, loss_datasets

