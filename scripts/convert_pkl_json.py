
# convert to instance label for SAMRS
# borrow from https://github.com/KyanChen/RSPrompter/blob/release/tools/rsprompter/whu2coco.py

import argparse
import glob
import os
import os.path as osp
import cv2
import mmcv
import numpy as np
import pickle
import pycocotools.mask as maskUtils
from PIL import Image
from mmengine.fileio import dump
from mmengine.utils import (Timer, mkdir_or_exist, track_parallel_progress,
                            track_progress)


DOTA2_0 = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane', 'airport', 'helipad')

DIOR = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'expressway-service-area', 'expressway-toll-station',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
        'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'windmill')

FAIR1M = ('A220','A321','A330','A350','ARJ21','Baseball-Field','Basketball-Court',
'Boeing737','Boeing747','Boeing777','Boeing787','Bridge','Bus','C919','Cargo-Truck',
'Dry-Cargo-Ship','Dump-Truck','Engineering-Ship','Excavator','Fishing-Boat',
'Football-Field','Intersection','Liquid-Cargo-Ship','Motorboat','other-airplane',
'other-ship','other-vehicle','Passenger-Ship','Roundabout','Small-Car','Tennis-Court',
'Tractor','Trailer','Truck-Tractor','Tugboat','Van','Warship')

def collect_categories(dataset):

    categories = []

    dataset_list = ['sota', 'sior', 'fast']
    idx = dataset_list.index(dataset)
    category_list = [DOTA2_0, DIOR, FAIR1M]

    for label, category in enumerate(category_list[idx]):
        categories.append(dict(id=label, name=category))
    
    return categories 


def collect_files(root, image_path, label_path, split, ext_img, ext_lbl):
    with open(os.path.join(root, '{}.txt'.format(split)), mode='r') as f:
        image_infos = f.readlines()
    f.close()

    files = []

    for item in image_infos:
        fname = item.strip()
        #print(fname)
        files.append(
                    (os.path.join(image_path, fname + ext_img), 
                     os.path.join(label_path, fname + ext_lbl))
                     )

    assert len(files), f'No images found in {image_path}'
    print(f'Loaded {len(files)} images from {image_path}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = track_parallel_progress(load_img_info, files, nproc=nproc)
    else:
        images = track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, segm_file = files # 都是路径+文件名
    segm_list = pickle.load(open(segm_file,'rb'))

    image = np.array(Image.open(img_file))

    anno_info = []
    for ins_info in segm_list:
        category_id = ins_info['label']
        mask_rle = ins_info['mask']
        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        anno = dict(
            iscrowd=0,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area,
            segmentation=mask_rle)
        anno_info.append(anno)

    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.basename(img_file),
        height=image.shape[0],
        width=image.shape[1],
        anno_info=anno_info,
        segm_file=osp.basename(segm_file))

    return img_info


def cvt_annotations(image_infos, out_json_name, dataset):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id # unique for each instance
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1

    out_json['categories'] = collect_categories(dataset)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    dump(out_json, out_json_name)
    return out_json


def main():

    dataset = 'sota'
    root = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval'
    image_path = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/images/'
    label_path = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval/rhbox_segs_init/ins/'
    out_dir = '/diwang22/dataset/samrs/dotav2_1024_rbb/trainval'
    nproc = 1
    ext_img = '.png'
    ext_lbl = '.pkl'

    mkdir_or_exist(out_dir)

    set_name = dict(
        train='{}_rbb_train_ins_segmentation.json'.format(dataset),
        valid='{}_rbb_valid_ins_segmentation.json'.format(dataset)
    )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with Timer(print_tmpl='It took {}s to convert annotation'):
            files = collect_files(root, image_path, label_path, split, ext_img, ext_lbl)
            image_infos = collect_annotations(files, nproc=nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name), dataset)


if __name__ == '__main__':
    main()