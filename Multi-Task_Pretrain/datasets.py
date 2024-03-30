# Copyright (c) OpenMMLab. All rights reserved.
import copy
import functools
import gc
import logging
import pickle
import torch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

from mmengine.fileio import join_path, list_from_file, load
from mmengine.logging import print_log
from mmengine.registry import TRANSFORMS
from mmengine.utils import is_abs
from mmdet.registry import DATASETS


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def force_full_init(old_func: Callable) -> Any:
    """Those methods decorated by ``force_full_init`` will be forced to call
    ``full_init`` if the instance has not been fully initiated.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``full_init`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} does not have full_init '
                                 'method.')
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, '_fully_initialized', False):
            print_log(
                f'Attribute `_fully_initialized` is not defined in '
                f'{type(obj)} or `type(obj)._fully_initialized is '
                'False, `full_init` will be called and '
                f'{type(obj)}._fully_initialized will be set to True',
                logger='current',
                level=logging.WARNING)
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper

####################################### mm数据集基类

class MultiTaskBaseDataset(Dataset):
    r"""BaseDataset for open source projects in OpenMMLab.

    The annotation format is shown as follows.

    .. code-block:: none

        {
            "metainfo":
            {
              "dataset_type": "test_dataset",
              "task_name": "test_task"
            },
            "data_list":
            [
              {
                "img_path": "test_img.jpg",
                "height": 604,
                "width": 640,
                "instances":
                [
                  {
                    "bbox": [0, 0, 10, 20],
                    "bbox_label": 1,
                    "mask": [[0,0],[0,10],[10,20],[20,0]],
                    "extra_anns": [1,2,3]
                  },
                  {
                    "bbox": [10, 10, 110, 120],
                    "bbox_label": 2,
                    "mask": [[10,10],[10,110],[110,120],[120,10]],
                    "extra_anns": [4,5,6]
                  }
                ]
              },
            ]
        }

    Args:
        ann_file (str, optional): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(img_path='').
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.

    Note:
        BaseDataset collects meta information from ``annotation file`` (the
        lowest priority), ``BaseDataset.METAINFO``(medium) and ``metainfo
        parameter`` (highest) passed to constructors. The lower priority meta
        information will be overwritten by higher one.

    Note:
        Dataset wrapper such as ``ConcatDataset``, ``RepeatDataset`` .etc.
        should not inherit from ``BaseDataset`` since ``get_subset`` and
        ``get_subset_`` could produce ambiguous meaning sub-dataset which
        conflicts with original dataset.

    Examples:
        >>> # Assume the annotation file is given above.
        >>> class CustomDataset(BaseDataset):
        >>>     METAINFO: dict = dict(task_name='custom_task',
        >>>                           dataset_type='custom_type')
        >>> metainfo=dict(task_name='custom_task_name')
        >>> custom_dataset = CustomDataset(
        >>>                      'path/to/ann_file',
        >>>                      metainfo=metainfo)
        >>> # meta information of annotation file will be overwritten by
        >>> # `CustomDataset.METAINFO`. The merged meta information will
        >>> # further be overwritten by argument `metainfo`.
        >>> custom_dataset.metainfo
        {'task_name': custom_task_name, dataset_type: custom_type}
    """

    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 reduce_zero_label: bool = False,):
        self.ann_file = ann_file
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        ####### for sem segmentation ##########
        self.reduce_zero_label = reduce_zero_label

        # Get label map for custom classes
        new_classes = self._metainfo.get('ss_classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))
        ####### for sem segmentation ##########

        # Join paths.
        self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        # if not lazy_init:
        #     self.full_init() # 完全初始化数据集类

    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get('ss_classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('ss_palette', [])
        classes = self._metainfo.get('ss_classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette
    
    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        # 采样数据，并赋一个sample_idx，用于后续pipeline
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info

    def full_init(self):
        # full init：完全初始化数据集类，该步骤主要包含以下操作：

        # load data list：读取与解析满足 OpenMMLab 2.0 数据集格式规范的标注文件，该步骤中会调用 parse_data_info() 方法，该方法负责解析标注文件里的每个原始数据；

        # filter data (可选)：根据 filter_cfg 过滤无用数据，比如不包含标注的样本等；默认不做过滤操作，下游子类可以按自身所需对其进行重写；

        # get subset (可选)：根据给定的索引或整数值采样数据，比如只取前 10 个样本参与训练/测试；默认不采样数据，即使用全部数据样本；

        # serialize data (可选)：序列化全部样本，以达到节省内存的效果，详情请参考节省内存；默认操作为序列化全部样本。
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        #self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    @property
    def metainfo(self) -> dict:
        # 获取数据集元信息
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        # 从标注文件中提取路径并连接
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix,
                                                  raw_data_info[prefix_key])
        return raw_data_info

    # def filter_data(self) -> List[dict]:
    #     """Filter annotations according to filter_cfg. Defaults return all
    #     ``data_list``.

    #     If some ``data_list`` could be filtered according to specific logic,
    #     the subclass should override this method.

    #     Returns:
    #         list[int]: Filtered results.
    #     """
    #     return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``ClassBalancedDataset`` requires a subclass which implements this
        method.

        Args:
            idx (int): The index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        raise NotImplementedError(f'{type(self)} must implement `get_cat_ids` '
                                  'method')

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    # def load_data_list(self) -> List[dict]:
    #     # 加载标注文件self.ann_file
    #     """Load annotations from an annotation file named as ``self.ann_file``

    #     If the annotation file does not follow `OpenMMLab 2.0 format dataset
    #     <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
    #     The subclass must override this method for load annotations. The meta
    #     information of annotation file will be overwritten :attr:`METAINFO`
    #     and ``metainfo`` argument of constructor.

    #     Returns:
    #         list[dict]: A list of annotation.
    #     """  # noqa: E501
    #     # `self.ann_file` denotes the absolute annotation file path if
    #     # `self.root=None` or relative path if `self.root=/path/to/data/`.
    #     annotations = load(self.ann_file)
    #     if not isinstance(annotations, dict):
    #         raise TypeError(f'The annotations loaded from annotation file '
    #                         f'should be a dict, but got {type(annotations)}!')
    #     if 'data_list' not in annotations or 'metainfo' not in annotations:
    #         raise ValueError('Annotation must have data_list and metainfo '
    #                          'keys')
    #     metainfo = annotations['metainfo']
    #     raw_data_list = annotations['data_list']

    #     # Meta information load from annotation file will not influence the
    #     # existed meta information load from `BaseDataset.METAINFO` and
    #     # `metainfo` arguments defined in constructor.
    #     for k, v in metainfo.items():
    #         self._metainfo.setdefault(k, v)

    #     # load and parse data_infos.
    #     data_list = []
    #     for raw_data_info in raw_data_list:
    #         # parse raw data information to target format
    #         data_info = self.parse_data_info(raw_data_info)
    #         if isinstance(data_info, dict):
    #             # For image tasks, `data_info` should information if single
    #             # image, such as dict(img_path='xxx', width=360, ...)
    #             data_list.append(data_info)
    #         elif isinstance(data_info, list):
    #             # For video tasks, `data_info` could contain image
    #             # information of multiple frames, such as
    #             # [dict(video_path='xxx', timestamps=...),
    #             #  dict(video_path='xxx', timestamps=...)]
    #             for item in data_info:
    #                 if not isinstance(item, dict):
    #                     raise TypeError('data_info must be list of dict, but '
    #                                     f'got {type(item)}')
    #             data_list.extend(data_info)
    #         else:
    #             raise TypeError('data_info should be a dict or list of dict, '
    #                             f'but got {type(data_info)}')

    #     return data_list

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:

        # 获取数据元信息
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Meta information dict. If ``metainfo``
                contains existed filename, it will be parsed by
                ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def _join_prefix(self):
        # 拼接生成数据和标记的路径
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    @force_full_init
    def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
        # 和get_subset的区别在于有没有copy数据
        """The in-place version of ``get_subset`` to convert dataset to a
        subset of original dataset.

        This method will convert the original dataset to a subset of dataset.
        If type of indices is int, ``get_subset_`` will return a subdataset
        which contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the data information according to
        the index given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> dataset.get_subset_(90)
              >>> len(dataset)
              90
              >>> # if type of indices is sequence, extract the corresponding
              >>> # index data information
              >>> dataset.get_subset_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              >>> len(dataset)
              10
              >>> dataset.get_subset_(-3)
              >>> len(dataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.
        """
        # Get subset of data from serialized data or data information sequence
        # according to `self.serialize_data`.
        if self.serialize_data:
            self.data_bytes, self.data_address = \
                self._get_serialized_subset(indices)
        else:
            self.data_list = self._get_unserialized_subset(indices)

    @force_full_init
    def get_subset(self, indices: Union[Sequence[int], int]) -> 'MultiTaskBaseDataset':
        """Return a subset of dataset.

        This method will return a subset of original dataset. If type of
        indices is int, ``get_subset_`` will return a subdataset which
        contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the information according to the index
        given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> subdataset = dataset.get_subset(90)
              >>> len(sub_dataset)
              90
              >>> # if type of indices is list, extract the corresponding
              >>> # index data information
              >>> subdataset = dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7,
              >>>                                  8, 9])
              >>> len(sub_dataset)
              10
              >>> subdataset = dataset.get_subset(-3)
              >>> len(subdataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.

        Returns:
            BaseDataset: A subset of dataset.
        """
        # Get subset of data from serialized data or data information list
        # according to `self.serialize_data`. Since `_get_serialized_subset`
        # will recalculate the subset data information,
        # `_copy_without_annotation` will copy all attributes except data
        # information.
        sub_dataset = self._copy_without_annotation()
        # Get subset of dataset with serialize and unserialized data.
        if self.serialize_data:
            data_bytes, data_address = \
                self._get_serialized_subset(indices)
            sub_dataset.data_bytes = data_bytes.copy()
            sub_dataset.data_address = data_address.copy()
        else:
            data_list = self._get_unserialized_subset(indices)
            sub_dataset.data_list = copy.deepcopy(data_list)
        return sub_dataset

    def _get_serialized_subset(self, indices: Union[Sequence[int], int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        # 根据索引抽取从序列化数据中抽取
        """Get subset of serialized data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of serialized
                data information list. If type of indices is Sequence, indices
                represents the target data information index which consist of
                subset data information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of serialized data
            information.
        """
        sub_data_bytes: Union[List, np.ndarray]
        sub_data_address: Union[List, np.ndarray]
        if isinstance(indices, int):
            if indices >= 0:
                assert indices < len(self.data_address), \
                    f'{indices} is out of dataset length({len(self)}'
                # Return the first few data information.
                end_addr = self.data_address[indices - 1].item() \
                    if indices > 0 else 0
                # Slicing operation of `np.ndarray` does not trigger a memory
                # copy.
                sub_data_bytes = self.data_bytes[:end_addr]
                # Since the buffer size of first few data information is not
                # changed,
                sub_data_address = self.data_address[:indices]
            else:
                assert -indices <= len(self.data_address), \
                    f'{indices} is out of dataset length({len(self)}'
                # Return the last few data information.
                ignored_bytes_size = self.data_address[indices - 1]
                start_addr = self.data_address[indices - 1].item()
                sub_data_bytes = self.data_bytes[start_addr:]
                sub_data_address = self.data_address[indices:]
                sub_data_address = sub_data_address - ignored_bytes_size
        elif isinstance(indices, Sequence):
            sub_data_bytes = []
            sub_data_address = []
            for idx in indices:
                assert len(self) > idx >= -len(self)
                start_addr = 0 if idx == 0 else \
                    self.data_address[idx - 1].item()
                end_addr = self.data_address[idx].item()
                # Get data information by address.
                sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
                # Get data information size.
                sub_data_address.append(end_addr - start_addr)
            # Handle indices is an empty list.
            if sub_data_bytes:
                sub_data_bytes = np.concatenate(sub_data_bytes)
                sub_data_address = np.cumsum(sub_data_address)
            else:
                sub_data_bytes = np.array([])
                sub_data_address = np.array([])
        else:
            raise TypeError('indices should be a int or sequence of int, '
                            f'but got {type(indices)}')
        return sub_data_bytes, sub_data_address  # type: ignore

    def _get_unserialized_subset(self, indices: Union[Sequence[int],
                                                      int]) -> list:
        # 根据索引抽取部分子数据
        """Get subset of data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of data
                information. If type of indices is Sequence, indices represents
                the target data information index which consist of subset data
                information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of data information.
        """
        if isinstance(indices, int):
            if indices >= 0:
                # Return the first few data information.
                sub_data_list = self.data_list[:indices]
            else:
                # Return the last few data information.
                sub_data_list = self.data_list[indices:]
        elif isinstance(indices, Sequence):
            # Return the data information according to given indices.
            sub_data_list = []
            for idx in indices:
                sub_data_list.append(self.data_list[idx])
        else:
            raise TypeError('indices should be a int or sequence of int, '
                            f'but got {type(indices)}')
        return sub_data_list

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:

        # 对数据进行序列化来加速
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def _rand_another(self) -> int:
        # 随便给个index，以防pipeline后返回数据为空
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def prepare_data(self, idx) -> Any:
        
        # 采样数据

        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    #@force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def _copy_without_annotation(self, memo=dict()) -> 'MultiTaskBaseDataset':
        # 拷贝一些数据集属性
        """Deepcopy for all attributes other than ``data_list``,
        ``data_address`` and ``data_bytes``.

        Args:
            memo: Memory dict which used to reconstruct complex object
                correctly.
        """
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            if key in ['data_list', 'data_address', 'data_bytes']:
                continue
            super(MultiTaskBaseDataset, other).__setattr__(key,
                                                  copy.deepcopy(value, memo))

        return other
    

####################################### 检测数据集基类

#@DATASETS.register_module()
class MultiTaskBaseDetDataset(MultiTaskBaseDataset):
    """Base dataset for detection.

    Args:
        proposal_file (str, optional): Proposals file path. Defaults to None.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        return_classes (bool): Whether to return class information
            for open vocabulary-based algorithms. Defaults to False.
    """

    def __init__(self,
                 *args,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False,
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(*args, **kwargs)

    def full_init(self) -> None:
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - load_proposals: Load proposals from proposal file, if
              `self.proposal_file` is not None.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        self.length = len(self.data_list)

        # get proposals from file
        if self.proposal_file is not None:
            self.load_proposals() # 相比数据集基类，多了个加载proposal
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def load_proposals(self) -> None:
        
        # 如果有proposal的话可以加载，并赋给每个样本的data info

        """Load proposals from proposals file.

        The `proposals_list` should be a dict[img_path: proposals]
        with the same length as `data_list`. And the `proposals` should be
        a `dict` or :obj:`InstanceData` usually contains following keys.

            - bboxes (np.ndarry): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
            - scores (np.ndarry): Classification scores, has a shape
              (num_instance, ).
        """
        # TODO: Add Unit Test after fully support Dump-Proposal Metric
        if not is_abs(self.proposal_file):
            self.proposal_file = osp.join(self.data_root, self.proposal_file)
        proposals_list = load(
            self.proposal_file, backend_args=self.backend_args)
        assert len(self.data_list) == len(proposals_list)
        for data_info in self.data_list:
            img_path = data_info['img_path']
            # `file_name` is the key to obtain the proposals from the
            # `proposals_list`.
            file_name = osp.join(
                osp.split(osp.split(img_path)[0])[-1],
                osp.split(img_path)[-1])
            proposals = proposals_list[file_name]
            data_info['proposals'] = proposals

    def get_cat_ids(self, idx: int) -> List[int]:
        # 每个图像中实例的类别标记
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        r_instances = self.get_data_info(idx)['r_instances']
        return [instance['bbox_label'] for instance in instances] | [instance['rbox_label'] for instance in r_instances]

####################################### COCO数据集

from mmdet.datasets.api_wrappers.coco_api import COCO
from mmengine.fileio import get_local_path
import xml.etree.ElementTree as ET

#@DATASETS.register_module()
class MultiTaskCocoDataset(MultiTaskBaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        'r_classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self,
                 *args,
                 diff_thr: int = 100,
                 rdet_ann_dir: Optional[str] = '',
                 rdet_post_fix: str = '.txt',
                 **kwargs)-> None:
        self.diff_thr = diff_thr
        self.rdet_ann_dir = rdet_ann_dir
        self.rdet_post_fix = rdet_post_fix
        
        super().__init__(*args, **kwargs)

    #def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes']) # 类别标记
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)} # 类别标记和训练标记映射
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map) # 某类有哪些图片？

        # rotated detection
        self.r_cls_map = {c: i
            for i, c in enumerate(self.metainfo['classes'])
            }  # in mmdet v2.0 label is 0-based

        self.img_ids = self.coco.get_img_ids()

        self.length = len(self.img_ids)

        #return img_ids
    
        # data_list = []
        # total_ann_ids = []
        # for img_id in img_ids:
        #     raw_img_info = self.coco.load_imgs([img_id])[0] # 图片信息
        #     raw_img_info['img_id'] = img_id

        #     ann_ids = self.coco.get_ann_ids(img_ids=[img_id]) # 得到每张图包含的ann id
        #     raw_ann_info = self.coco.load_anns(ann_ids) # 这些ann id 对应的ann信息
        #     total_ann_ids.extend(ann_ids)

        #     parsed_data_info = self.parse_data_info({
        #         'raw_ann_info':
        #         raw_ann_info,
        #         'raw_img_info':
        #         raw_img_info
        #     })
        #     data_list.append(parsed_data_info)
        # if self.ANN_ID_UNIQUE: # ann id必须是独一无二的
        #     assert len(set(total_ann_ids)) == len(
        #         total_ann_ids
        #     ), f"Annotation ids in '{self.ann_file}' are not unique!"

        # #del self.coco

        # return data_list

    def __len__(self):


        return len(self.img_ids)

    def __getitem__(self, i):

        return self.img_ids[i]

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None
        
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:

        # 将提取出来的粗信息进行精挑
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])  # 每个样本的图像路径
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path # 语义分割图？
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        data_info['reduce_zero_label'] = self.reduce_zero_label

        # borrow from mmrotate
        img_name = osp.split(img_path)[1] 
        data_info['file_name'] = img_name

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h] # 水平框信息

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation'] # 实例的mask

            instances.append(instance)
        data_info['instances'] = instances

        ## rotated detection

        if self.rdet_ann_dir != '':

            if self.rdet_post_fix == '.txt':

                txt_file = osp.join(self.rdet_ann_dir, 
                                    img_info['file_name'].rsplit('.', 1)[0] + '.txt')

                r_instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        r_instance = {}
                        rbox_info = si.split() # 每个旋转框实例
                        r_instance['rbox'] = [float(i) for i in rbox_info[:8]]

                        if (r_instance['rbox'][0:2] == r_instance['rbox'][2:4]) or  (r_instance['rbox'][2:4] == r_instance['rbox'][4:6]) \
                            or (r_instance['rbox'][4:6] == r_instance['rbox'][6:]) or (r_instance['rbox'][6:] == r_instance['rbox'][0:2]):
                            continue

                        r_cls_name = rbox_info[8]
                        r_instance['rbox_label'] = self.r_cls_map[r_cls_name]
                        difficulty = int(rbox_info[9])
                        if difficulty > self.diff_thr:
                            r_instance['r_ignore_flag'] = 1
                        else:
                            r_instance['r_ignore_flag'] = 0
                        r_instances.append(r_instance)

            elif self.rdet_post_fix == '.xml':

                xml_file = osp.join(self.rdet_ann_dir, 
                                    img_info['file_name'].rsplit('.', 1)[0] + '.xml')

                # deal with xml file
                with get_local_path(
                        xml_file,
                        backend_args=self.backend_args) as local_path:
                    raw_ann_info = ET.parse(local_path)
                root = raw_ann_info.getroot()

                r_instances = []
                for obj in root.findall('object'):
                    r_instance = {}
                    r_cls = obj.find('name').text.lower()
                    r_label = self.r_cls_map[r_cls]
                    if r_label is None:
                        continue

                    r_bnd_box = obj.find('robndbox')
                    polygon = np.array([
                        float(r_bnd_box.find('x_left_top').text),
                        float(r_bnd_box.find('y_left_top').text),
                        float(r_bnd_box.find('x_right_top').text),
                        float(r_bnd_box.find('y_right_top').text),
                        float(r_bnd_box.find('x_right_bottom').text),
                        float(r_bnd_box.find('y_right_bottom').text),
                        float(r_bnd_box.find('x_left_bottom').text),
                        float(r_bnd_box.find('y_left_bottom').text),
                    ]).astype(np.float32)

                    if (polygon[0:2] == polygon[2:4]).all() or  (polygon[2:4] == polygon[4:6]).all() or \
                        (polygon[4:6] == polygon[6:]).all() or (polygon[6:] == polygon[0:2]).all():
                        continue

                    ignore = False
                    if self.bbox_min_size is not None:
                        assert not self.test_mode
                        if img_info['width'] < self.bbox_min_size or img_info['height'] < self.bbox_min_size:
                            ignore = True
                    if ignore:
                        r_instance['r_ignore_flag'] = 1
                    else:
                        r_instance['r_ignore_flag'] = 0
                    r_instance['rbox'] = polygon
                    r_instance['rbox_label'] = r_label
                    r_instances.append(r_instance)

            else:

                raise NotImplementedError
            
            data_info['r_instances'] = r_instances
                
        else:

            raise NotImplementedError

        return data_info

    def filter_data(self, data_list) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return data_list

        if self.filter_cfg is None:
            return data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation # 有标记的图片
        ids_with_ann = set(data_info['img_id'] for data_info in data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set() # 有所需类别的图片
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id]) 
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            # 图片需同时满足上述两条件以及旋转框不为空的要求
            if filter_empty_gt and ((img_id not in ids_in_cat) or (len(data_info['r_instances']==0))):
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

            # 最后得到的是拥有所需类别及相应标注且尺寸符合要求的图片

        return valid_data_infos
    

#@DATASETS.register_module()
class SOTAMultiTaskDataset(MultiTaskCocoDataset):
    METAINFO = {
        'ss_classes': ['background', 'large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane', 'airport', 'helipad'],
                
        'classes': ['large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane', 'airport', 'helipad'],

        'ss_palette': [(255,255,255), (0, 127, 255), (0, 63, 0), (0, 127, 63), (0, 63, 255),
                    (0, 0, 127), (0, 127, 127), (0, 0, 63), (0, 63, 127),
                    (0, 63, 191), (0, 191, 127), (0, 127, 191), (0, 63, 63),
                    (0, 100, 155), (0, 0, 255), (0, 0, 191), (64, 191, 127),
                    (64, 0, 191), (128, 63, 63)],

        'palette': [(0, 127, 255), (0, 63, 0), (0, 127, 63), (0, 63, 255),
                    (0, 0, 127), (0, 127, 127), (0, 0, 63), (0, 63, 127),
                    (0, 63, 191), (0, 191, 127), (0, 127, 191), (0, 63, 63),
                    (0, 100, 155), (0, 0, 255), (0, 0, 191), (64, 191, 127),
                    (64, 0, 191), (128, 63, 63)]
    }
                 


#@DATASETS.register_module()
class SIORMultiTaskDataset(MultiTaskCocoDataset):
    METAINFO = {
        'ss_classes': ['background', 'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
                    'chimney', 'expressway-service-area', 'expressway-toll-station',
                    'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
                    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                    'windmill'],

        'classes': ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
                    'chimney', 'expressway-service-area', 'expressway-toll-station',
                    'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
                    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
                    'windmill'],

        'ss_palette': [(255,255,255),(0, 127, 255), (0, 63, 0), (0, 127, 63), (0, 63, 255),
                    (0, 0, 127), (0, 127, 127), (0, 0, 63), (0, 63, 127),
                    (0, 63, 191), (0, 191, 127), (0, 127, 191), (0, 63, 63),
                    (0, 100, 155), (0, 0, 255), (0, 0, 191), (64, 191, 127),
                    (64, 0, 191), (128, 63, 63), (128, 0, 63), (191, 63, 0)],

        'palette': [(0, 127, 255), (0, 63, 0), (0, 127, 63), (0, 63, 255),
                    (0, 0, 127), (0, 127, 127), (0, 0, 63), (0, 63, 127),
                    (0, 63, 191), (0, 191, 127), (0, 127, 191), (0, 63, 63),
                    (0, 100, 155), (0, 0, 255), (0, 0, 191), (64, 191, 127),
                    (64, 0, 191), (128, 63, 63), (128, 0, 63), (191, 63, 0)]
    }

#@DATASETS.register_module()
class FASTMultiTaskDataset(MultiTaskCocoDataset):
    METAINFO = {
        'ss_classes': ['background','A220','A321','A330','A350','ARJ21','Baseball-Field','Basketball-Court',
                    'Boeing737','Boeing747','Boeing777','Boeing787','Bridge','Bus','C919','Cargo-Truck',
                    'Dry-Cargo-Ship','Dump-Truck','Engineering-Ship','Excavator','Fishing-Boat',
                    'Football-Field','Intersection','Liquid-Cargo-Ship','Motorboat','other-airplane',
                    'other-ship','other-vehicle','Passenger-Ship','Roundabout','Small-Car','Tennis-Court',
                    'Tractor','Trailer','Truck-Tractor','Tugboat','Van','Warship'],

        'classes': ['A220','A321','A330','A350','ARJ21','Baseball-Field','Basketball-Court',
                    'Boeing737','Boeing747','Boeing777','Boeing787','Bridge','Bus','C919','Cargo-Truck',
                    'Dry-Cargo-Ship','Dump-Truck','Engineering-Ship','Excavator','Fishing-Boat',
                    'Football-Field','Intersection','Liquid-Cargo-Ship','Motorboat','other-airplane',
                    'other-ship','other-vehicle','Passenger-Ship','Roundabout','Small-Car','Tennis-Court',
                    'Tractor','Trailer','Truck-Tractor','Tugboat','Van','Warship'],

        'ss_palette': [(255,255,255), (0, 127, 255), (0, 63, 0), (0, 127, 63), (0, 63, 255),
                    (0, 0, 127), (0, 127, 127), (0, 0, 63), (0, 63, 127),
                    (0, 63, 191), (0, 191, 127), (0, 127, 191), (0, 63, 63),
                    (0, 100, 155), (0, 0, 255), (0, 0, 191), (64, 191, 127),
                    (64, 0, 191), (128, 63, 63), (128, 0, 63), (191, 63, 0),
                    (255, 127, 0), (63, 0, 0), (127, 63, 0), (63, 255, 0),
                    (0, 127, 0), (127, 127, 0), (63, 0, 63), (63, 127, 0),
                    (63, 191, 0), (191, 127, 0), (127, 191, 0), (63, 63, 0), 
                    (100, 155, 0),  (0, 255, 0), (0, 191, 0), (191, 127, 64),
                    (0, 191, 64)],

        'palette': [(0, 127, 255), (0, 63, 0), (0, 127, 63), (0, 63, 255),
                    (0, 0, 127), (0, 127, 127), (0, 0, 63), (0, 63, 127),
                    (0, 63, 191), (0, 191, 127), (0, 127, 191), (0, 63, 63),
                    (0, 100, 155), (0, 0, 255), (0, 0, 191), (64, 191, 127),
                    (64, 0, 191), (128, 63, 63), (128, 0, 63), (191, 63, 0),
                    (255, 127, 0), (63, 0, 0), (127, 63, 0), (63, 255, 0),
                    (0, 127, 0), (127, 127, 0), (63, 0, 63), (63, 127, 0),
                    (63, 191, 0), (191, 127, 0), (127, 191, 0), (63, 63, 0), 
                    (100, 155, 0),  (0, 255, 0), (0, 191, 0), (191, 127, 64),
                    (0, 191, 64)]
    }