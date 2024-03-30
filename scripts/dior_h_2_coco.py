# codes borrow from　https://blog.csdn.net/qq_20793791/article/details/110881417, something changed

import os
import cv2
from tqdm import tqdm
import json
import xml.dom.minidom

DIOR = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'expressway-service-area', 'expressway-toll-station',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
        'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'windmill')

dior_cls2lbl = {k:v for v,k in enumerate(DIOR)}

def convert_to_cocodetection(dir, output_dir):
    """
    input:
        dir:the path to DIOR dataset
        output_dir:the path write the coco form json file
    """
    annotations_path = dir
    namelist_path = os.path.join(dir, "ImageSets", "Main")
    trainval_images_path = os.path.join(dir,  "JPEGImages-trainval")
    test_images_path = os.path.join(dir, "JPEGImages-test")
    categories = [{"id": 0, "name": "airplane"},
                  {"id": 1, "name": "airport"},
                  {"id": 2, "name": "baseballfield"},
                  {"id": 3, "name": "basketballcourt"},
                  {"id": 4, "name": "bridge"},
                  {"id": 5, "name": "chimney"},
                  {"id": 6, "name": "expressway-service-area"},
                  {"id": 7, "name": "expressway-toll-station"},
                  {"id": 8, "name": "dam"},
                  {"id": 9, "name": "golffield"},
                  {"id": 10, "name": "groundtrackfield"},
                  {"id": 11, "name": "harbor"},
                  {"id": 12, "name": "overpass"},
                  {"id": 13, "name": "ship"},
                  {"id": 14, "name": "stadium"},
                  {"id": 15, "name": "storagetank"},
                  {"id": 16, "name": "tenniscourt"},
                  {"id": 17, "name": "trainstation"},
                  {"id": 18, "name": "vehicle"},
                  {"id": 19, "name": "windmill"},
                  ]
    for mode in ["trainval", "test"]:
        images = []
        annotations = []
        img_id = 0
        annotation_id = 0
        print(f"start loading {mode} data...")
        # if mode == "train":
        #     f = open(namelist_path + "/" + "train.txt", "r")
        #     images_path = trainval_images_path
        # elif mode == "val":
        #     f = open(namelist_path + "/" + "val.txt", "r")
        #     images_path = trainval_images_path
        if mode == "trainval":
            f = open(namelist_path + "/" + "trainval.txt", "r")
            images_path = trainval_images_path
        else:
            f = open(namelist_path + "/" + "test.txt", "r")
            images_path = test_images_path
        for name in tqdm(f.readlines()):
            # image part
            image = {}
 
            name = name.replace("\n", "")
            image_name = name + ".jpg"
            annotation_name = name + ".xml"
            height, width = cv2.imread(images_path + "/" + image_name).shape[:2]
            image["file_name"] = image_name
            image["height"] = height
            image["width"] = width
            image["id"] = img_id
            images.append(image)
            # anno part
            dom = xml.dom.minidom.parse(dir + "/Annotations/Horizontal Bounding Boxes/" + annotation_name)
            root_data = dom.documentElement
            for i in range(len(root_data.getElementsByTagName('name'))):
                annotation = {}
                category = root_data.getElementsByTagName('name')[i].firstChild.data
                top_left_x = root_data.getElementsByTagName('xmin')[i].firstChild.data
                top_left_y = root_data.getElementsByTagName('ymin')[i].firstChild.data
                right_bottom_x = root_data.getElementsByTagName('xmax')[i].firstChild.data
                right_bottom_y = root_data.getElementsByTagName('ymax')[i].firstChild.data
                bbox = [top_left_x, top_left_y, right_bottom_x, right_bottom_y]
                bbox = [int(i) for i in bbox]
                bbox = xyxy_to_xywh(bbox)
                annotation["image_id"] = img_id
                annotation["bbox"] = bbox
                annotation["category_id"] = int(dior_cls2lbl[category.lower()])
                annotation["id"] = annotation_id
                annotation["iscrowd"] = 0
                annotation["segmentation"] = []
                annotation["area"] = bbox[2] * bbox[3]
                annotations.append(annotation)
                annotation_id += 1
            img_id += 1

        dataset_dict = {}
        dataset_dict["images"] = images
        dataset_dict["annotations"] = annotations
        dataset_dict["categories"] = categories
        json_str = json.dumps(dataset_dict)
        with open(f'{output_dir}/DIOR_{mode}_coco.json', 'w') as json_file:
            json_file.write(json_str)
    print("json file write done...")
 
 
def get_test_namelist(dir, out_dir):
    full_path = out_dir + "/" + "test.txt"
    file = open(full_path, 'w')
    for name in tqdm(os.listdir(dir)):
        name = name.replace(".txt", "")
        file.write(name + "\n")
    file.close()
    return None
 
 
def centerxywh_to_xyxy(boxes):
    """
    args:
        boxes:list of center_x,center_y,width,height,
    return:
        boxes:list of x,y,x,y,cooresponding to top left and bottom right
    """
    x_top_left = boxes[0] - boxes[2] / 2
    y_top_left = boxes[1] - boxes[3] / 2
    x_bottom_right = boxes[0] + boxes[2] / 2
    y_bottom_right = boxes[1] + boxes[3] / 2
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
 
 
def centerxywh_to_topleftxywh(boxes):
    """
    args:
        boxes:list of center_x,center_y,width,height,
    return:
        boxes:list of x,y,x,y,cooresponding to top left and bottom right
    """
    x_top_left = boxes[0] - boxes[2] / 2
    y_top_left = boxes[1] - boxes[3] / 2
    width = boxes[2]
    height = boxes[3]
    return [x_top_left, y_top_left, width, height]
 
 
def xyxy_to_xywh(boxes):
    width = boxes[2] - boxes[0]
    height = boxes[3] - boxes[1]
    return [boxes[0], boxes[1], width, height]
 
 
def clamp(coord, width, height):
    if coord[0] < 0:
        coord[0] = 0
    if coord[1] < 0:
        coord[1] = 0
    if coord[2] > width:
        coord[2] = width
    if coord[3] > height:
        coord[3] = height
    return coord
 
 
if __name__ == '__main__':
    convert_to_cocodetection("/diwang22/dataset/samrs/dior", "/diwang22/dataset/samrs/dior/ImageSets/Main")