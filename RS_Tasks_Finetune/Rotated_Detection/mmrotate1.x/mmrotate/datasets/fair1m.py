# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS

@DATASETS.register_module()
class FAIR1Mv2Dataset(BaseDataset):

    METAINFO = {
        'classes':
        ('A220','A321','A330','A350','ARJ21','Baseball-Field','Basketball-Court',
         'Boeing737','Boeing747','Boeing777','Boeing787','Bridge','Bus','C919','Cargo-Truck',
         'Dry-Cargo-Ship','Dump-Truck','Engineering-Ship','Excavator','Fishing-Boat',
         'Football-Field','Intersection','Liquid-Cargo-Ship','Motorboat','other-airplane',
         'other-ship','other-vehicle','Passenger-Ship','Roundabout','Small-Car','Tennis-Court',
         'Tractor','Trailer','Truck-Tractor','Tugboat','Van','Warship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(151, 0, 95),
         (9, 80, 61), (84, 105, 51), (74, 65, 105), (166, 196, 102), 
         (208, 195, 210), (255, 109, 65), (0, 143, 149), (179, 0, 194), 
         (209, 99, 106), (5, 121, 0), (227, 255, 205), (147, 186, 208), 
         (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170),
         (0, 182, 199), (0, 165, 120), (183, 130, 88), (95, 32, 0), 
         (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), 
         (79, 210, 114), (178, 90, 62), (65, 70, 15), (127, 167, 115), 
         (59, 105, 106), (142, 108, 45), (196, 172, 0), (95, 54, 80), 
         (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208)
         ]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]

                        if (instance['bbox'][0:2] == instance['bbox'][2:4]) or  (instance['bbox'][2:4] == instance['bbox'][4:6]) \
                            or (instance['bbox'][4:6] == instance['bbox'][6:]) or (instance['bbox'][6:] == instance['bbox'][0:2]):
                            continue

                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]