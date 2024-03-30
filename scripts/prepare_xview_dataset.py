import json
import os
import random
import torch
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from glob import glob

HELP_URL = "See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance."
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # image suffixes

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml
##################################################### 把geojson转成yolo格式的标签 ####################################
                
# 保证框在图像内
def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

# 左上右下格式变归一化xywh（yolo合适）
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def convert_labels(fname=Path('xView/xView_train.geojson')):
    # Convert xView geoJSON labels to YOLO format
    path = fname.parent
    with open(fname) as f:
        print(f'Loading {fname}...')
        data = json.load(f)

    # Make dirs
    labels = Path(path / 'train_txts')
    # os.system(f'rm -rf {labels}')
    labels.mkdir(exist_ok=True)

    # xView classes 11-94 to 0-59
    xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
                        12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
                        29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
                        47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]

    shapes = {}
    for feature in tqdm(data['features'], desc=f'Converting {fname}'):
        p = feature['properties']
        if p['bounds_imcoords']:
            id = p['image_id'] # 文件名
            file = path / 'train_images' / id
            if file.exists():  # 1395.tif missing
                try:
                    box = np.array([int(num) for num in p['bounds_imcoords'].split(",")]) # 框坐标
                    assert box.shape[0] == 4, f'incorrect box shape {box.shape[0]}'
                    cls = p['type_id'] # 初始类别号
                    cls = xview_class2index[int(cls)]  # xView class to 0-60， 获得目标类别
                    assert 59 >= cls >= 0, f'incorrect class index {cls}'

                    # Write YOLO label
                    if id not in shapes:
                        shapes[id] = Image.open(file).size # 图像尺寸
                    box = xyxy2xywhn(box[None].astype(np.float64), w=shapes[id][0], h=shapes[id][1], clip=True)
                    with open((labels / id).with_suffix('.txt'), 'a') as f:
                        f.write(f"{cls} {' '.join(f'{x:.6f}' for x in box[0])}\n")  # write label.txt
                except Exception as e:
                    print(f'WARNING: skipping one label for {file}: {e}')


# Download manually from https://challenge.xviewdataset.org
dir = Path('/diwang22/dataset/Xview-dataset')  # dataset root dir

# Convert labels
#convert_labels(fname = (dir / 'xView_train.geojson'))

# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml
##################################################### 随机划分训练集和验证集 ####################################

# import shutil

# def copy_file(src_filepath, dest_dir):
#     address, _ =  os.path.split(dest_dir)
#     os.makedirs(address, exist_ok=True)
#     shutil.copy(src_filepath, dest_dir)

# txt_addr = '/diwang22/dataset/Xview-dataset/train_txts/'
# img_addr = '/diwang22/dataset/Xview-dataset/train_images/'

# txt_files = glob(os.path.join(txt_addr,'*.txt'))
# img_files = glob(os.path.join(img_addr, '*.tif'))

# idxs = np.arange(len(txt_files))

# np.random.shuffle(idxs)

# trn_num = 700
# val_num = len(txt_files) - trn_num

############ train set

# dest_txt_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/train/txts/'
# dest_img_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/train/images/'

# for i in tqdm(range(trn_num)):
#     _, txt_name = os.path.split(txt_files[idxs[i]])
#     src_txt_path = os.path.join(txt_addr, txt_name)
#     src_img_path = os.path.join(img_addr, txt_name[:-4]+'.tif')
#     dest_txt_path = os.path.join(dest_txt_addr, txt_name)
#     dest_img_path = os.path.join(dest_img_addr, txt_name[:-4]+'.tif')
    # copy_file(src_txt_path, dest_txt_path)
    # copy_file(src_img_path, dest_img_path)

############ val set

# dest_txt_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/val/txts/'
# dest_img_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/val/images/'

# for i in tqdm(range(trn_num, trn_num + val_num)):
#     _, txt_name = os.path.split(txt_files[idxs[i]])
#     src_txt_path = os.path.join(txt_addr, txt_name)
#     src_img_path = os.path.join(img_addr, txt_name[:-4]+'.tif')
#     dest_txt_path = os.path.join(dest_txt_addr, txt_name)
#     dest_img_path = os.path.join(dest_img_addr, txt_name[:-4]+'.tif')
    # copy_file(src_txt_path, dest_txt_path)
    # copy_file(src_img_path, dest_img_path)

################################################ 裁剪yolo格式的 #################################################
import cv2
from skimage import io

def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    if imgPath.endswith('.tif'):
        img = io.imread(imgPath)
        #img = tif.read_image()
        # img = tifffile.imread(imgPath)
    else:
        raise ValueError('Install pillow and uncomment line in load_img')
    return img

BLOCK_SZ = (416, 416)
BLOCK_MIN_OVERLAP = 0

def clip_xview_yolo(txt_addr, img_addr, patch_txt_addr, patch_img_addr):

    os.makedirs(patch_txt_addr, exist_ok=True)
    os.makedirs(patch_img_addr, exist_ok=True)

    txt_files = glob(os.path.join(txt_addr, '*.txt'))

    for i in tqdm(range(len(txt_files))):

        _, fname = os.path.split(txt_files[i])
        img_name = os.path.join(img_addr, fname[:-4] + '.tif')
        img = load_img(img_name)

        padding_h = max(0, BLOCK_SZ[0]-img.shape[0])
        padding_w = max(0, BLOCK_SZ[1]-img.shape[1])

        img = cv2.copyMakeBorder(img, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=(128,128,128))#top,bottom,left,right

        IMG_SZ = img.shape[:2]

        h, w = IMG_SZ

        yEnd, xEnd = np.subtract(IMG_SZ, BLOCK_SZ)
        x = np.linspace(0, xEnd, int(np.ceil(xEnd / (BLOCK_SZ[1] - BLOCK_MIN_OVERLAP))) + 1, endpoint=True).astype('int')
        y = np.linspace(0, yEnd, int(np.ceil(yEnd / (BLOCK_SZ[0] - BLOCK_MIN_OVERLAP)))+ 1, endpoint=True).astype('int')

        partInd = 0

        # 每张大图的所有box信息（归一化后的）
        with open(os.path.join(txt_files[i]), mode='r') as f:
            box_infos_norm = f.readlines()
            f.close()

        box_data_norm = []

        for box_item_norm in box_infos_norm:
            lab, cx, cy, bw, bh = box_item_norm.strip().split()
            box_data_norm.append([int(lab), float(cx), float(cy), float(bw), float(bh)])
        
        box_data_norm = np.array(box_data_norm)
        box_data = np.zeros(box_data_norm.shape)

        # 获得该图片中所有box的绝对位置坐标（cx, cy, bw, bh)

        box_data[:,0] = box_data_norm[:, 0]
        box_data[:,1] = box_data_norm[:, 1] * w
        box_data[:,2] = box_data_norm[:, 2] * h
        box_data[:,3] = box_data_norm[:, 3] * w
        box_data[:,4] = box_data_norm[:, 4] * h

        for j in range(len(y)):
            for k in range(len(x)):
                rStart, cStart = (y[j], x[k])
                rEnd, cEnd = (rStart + BLOCK_SZ[0], cStart + BLOCK_SZ[1])

                # 获得中心点在子图内部的box，此时还是绝对坐标

                valid_ids = list(set(np.where(box_data[:,1] >= cStart)[0]) & set(np.where(box_data[:,1] < cEnd)[0]) & \
                                set(np.where(box_data[:,2] >= rStart)[0]) & set(np.where(box_data[:,2] < rEnd)[0]))
                
                # 子图内有框的话就处理
                if len(valid_ids) > 0:

                    # 保存子图
                    curr_Img = img[rStart:rEnd, cStart:cEnd, :]
                    imgpartname = os.path.join(patch_img_addr, fname[:-4] + '_' + str(partInd) + '.png')
                    curr_Img = Image.fromarray(curr_Img)
                    curr_Img.save(imgpartname)

                    # 取出那些中心点在子图内的框
                    box_data_vaild = box_data[valid_ids]
                    txtpartname = os.path.join(patch_txt_addr, fname[:-4] + '_' + str(partInd) + '.txt')

                    f = open(txtpartname, mode='w')

                    for ii in range(box_data_vaild.shape[0]):

                        lab, cx, cy, bw, bh = box_data_vaild[ii]

                        # box的左上角和右下角

                        x1 = cx - 0.5*bw
                        x2 = cx + 0.5*bw
                        y1 = cy - 0.5*bh
                        y2 = cy + 0.5*bh

                        # 根据框和子图的关系调整框的位置

                        if x1 < cStart:
                            x1 = cStart
                        
                        if x2 >= cEnd:
                            x2 = cEnd-1
                        
                        if y1 < rStart:
                            y1 = rStart

                        if y2 >= rEnd:
                            y2 = rEnd-1

                        # 调整后新框的中心坐标

                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        assert cStart <= cx < cEnd and rStart <= cy < rEnd

                        # 对新框根据子图大小进行归一化

                        bw = (x2 - x1) / (cEnd - cStart)
                        bh = (y2 - y1) / (rEnd - rStart)

                        # 根据子图的位置转换框中心坐标系，并归一化
                        cx = (cx - cStart) / (cEnd - cStart)
                        cy = (cy - rStart) / (rEnd - rStart)

                        new_box_norm = [cx, cy, bw, bh]

                        f.write(f"{lab} {' '.join(f'{x:.6f}' for x in new_box_norm)}\n")

                    f.close()

                    partInd += 1

                else:
                    continue

############ train set

txt_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/train/txts'
img_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/train/images'

patch_txt_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format_patch/train/txts'
patch_img_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format_patch/train/images'

#clip_xview_yolo(txt_addr, img_addr, patch_txt_addr, patch_img_addr)

############ val set

txt_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/val/txts'
img_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format/val/images'

patch_txt_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format_patch/val/txts'
patch_img_addr = '/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format_patch/val/images'
            
#clip_xview_yolo(txt_addr, img_addr, patch_txt_addr, patch_img_addr)


############################################ 最后，把yolo格式转成COCO格式 ########################################

import os
import json
from PIL import Image
from mmengine.fileio import dump

categories={
  0:'Fixed-wing Aircraft',
  1:'Small Aircraft',
  2:'Cargo Plane',
  3:'Helicopter',
  4:'Passenger Vehicle',
  5:'Small Car',
  6:'Bus',
  7:'Pickup Truck',
  8:'Utility Truck',
  9:'Truck',
  10:'Cargo Truck',
  11:'Truck w/Box',
  12:'Truck Tractor',
  13:'Trailer',
  14:'Truck w/Flatbed',
  15:'Truck w/Liquid',
  16:'Crane Truck',
  17:'Railway Vehicle',
  18:'Passenger Car',
  19:'Cargo Car',
  20:'Flat Car',
  21:'Tank car',
  22:'Locomotive',
  23:'Maritime Vessel',
  24:'Motorboat',
  25:'Sailboat',
  26:'Tugboat',
  27:'Barge',
  28:'Fishing Vessel',
  29:'Ferry',
  30:'Yacht',
  31:'Container Ship',
  32:'Oil Tanker',
  33:'Engineering Vehicle',
  34:'Tower crane',
  35:'Container Crane',
  36:'Reach Stacker',
  37:'Straddle Carrier',
  38:'Mobile Crane',
  39:'Dump Truck',
  40:'Haul Truck',
  41:'Scraper/Tractor',
  42:'Front loader/Bulldozer',
  43:'Excavator',
  44:'Cement Mixer',
  45:'Ground Grader',
  46:'Hut/Tent',
  47:'Shed',
  48:'Building',
  49:'Aircraft Hangar',
  50:'Damaged Building',
  51:'Facility',
  52:'Construction Site',
  53:'Vehicle Lot',
  54:'Helipad',
  55:'Storage Tank',
  56:'Shipping container lot',
  57:'Shipping Container',
  58:'Pylon',
  59:'Tower'
}

def collect_categories():

    categories_list = []

    for i in range(60):
        categories_list.append(dict(id=i, name=categories[i]))
    
    return categories_list


 
# 设置数据集路径
dataset_path = "/diwang22/dataset/Xview-dataset/xview_dataset/yolo_format_patch"
# images_path = os.path.join(dataset_path, "images")
# labels_path = os.path.join(dataset_path, "labels")
 
# # 类别映射
# categories = [
#     {"id": 1, "name": "category1"},
#     {"id": 2, "name": "category2"},
#     # 添加更多类别
# ]


# YOLO格式转COCO格式的函数
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]
 
# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": collect_categories()
    }
 
# 处理每个数据集分区
for split in ['train', 'val']:
    coco_format = init_coco_format()
    img_id = 0
    annotation_id = 0
 
    for img_name in tqdm(os.listdir(os.path.join(dataset_path, split, 'images'))):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, split, 'images', img_name)
            label_path = os.path.join(dataset_path, split, 'txts', img_name.replace("png", "txt"))
 
            img = Image.open(img_path)
            img_width, img_height = img.size
            image_info = {
                "file_name": img_name,
                "id": img_id,
                "width": img_width,
                "height": img_height
            }
            coco_format["images"].append(image_info)
 
            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        category_id, x_center, y_center, width, height = map(float, line.split())
                        bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_info["id"],
                            "category_id": int(category_id),
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        }
                        coco_format["annotations"].append(annotation)
                        annotation_id += 1
            
            img_id += 1
 
    # 为每个分区保存JSON文件
    #with open(os.path.join(dataset_path, "{}_coco_format.json".format(split)), "w") as json_file:
    #json.dump(coco_format, json_file)
    json_name = os.path.join(dataset_path, "xview_{}_coco_format.json".format(split))
    dump(coco_format, json_name)






