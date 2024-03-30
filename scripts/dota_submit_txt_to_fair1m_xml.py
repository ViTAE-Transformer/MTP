import os
from tqdm import tqdm
from glob import glob
from skimage import io
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    if imgPath.endswith(".tif"):
        img = io.imread(imgPath)
        #img = tif.read_image()
        # img = tifffile.imread(imgPath)
    else:
        raise ValueError("Install pillow and uncomment line in load_img")
    return img

img_dir = "/diwang22/dataset/samrs/fair1m20_full/test/images/"

parser = argparse.ArgumentParser(description="dota txt 2 fair1m xml")
parser.add_argument("--txt_dir", type=str, default="fair1m", help="address of predicted txt in dota format")
args = parser.parse_args()

img_files = glob(os.path.join(img_dir, "*.tif"))
save_xml_dir = os.path.join(args.txt_dir, "init_xml")
os.makedirs(save_xml_dir, exist_ok=True)

for img_file in tqdm(img_files):

    _, fname = os.path.split(img_file)

    img_name = fname[:-4]

    annotation = ET.Element("annotation")

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "filename").text =  img_name + ".tif"
    ET.SubElement(source, "origin").text =  "GF2/GF3"

    research = ET.SubElement(annotation, "research")
    ET.SubElement(research, "version").text =  "2.0"
    ET.SubElement(research, "provider").text =  "WHU"
    ET.SubElement(research, "author").text =  "SIGMA_dw"
    ET.SubElement(research, "pluginname").text =  "FAIR1M"
    ET.SubElement(research, "pluginclass").text =  "object detection"
    ET.SubElement(research, "time").text =  "2024-02"

    img = load_img(os.path.join(img_dir, img_name+".tif"))
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img.shape[1])
    ET.SubElement(size, "height").text = str(img.shape[0])
    ET.SubElement(size, "depth").text = str(img.shape[2])

    objs = ET.SubElement(annotation, "objects")

    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(save_xml_dir, "{}.xml".format(img_name)), xml_declaration=True, encoding="utf-8", method="xml")

print('Finish init xml generalization!')

txt_files = glob(os.path.join(args.txt_dir, "*.txt"))

cnt = 1

for txt_file in txt_files:
    with open(txt_file, mode="r") as f:
        infos = f.readlines()
    f.close()

    _, fname = os.path.split(txt_file)
    fname = fname[:-4] # remove ".txt"
    category_name = fname.split("_")[-1] # obtain ["Task1", "xxxx"], xxxx is the category name defined in BBoxToolit/SAMRS/mmrotate
    fair1m_category_name = category_name.replace("-", " ") # truth category name in fair1m

    for box_info in tqdm(infos):
        img_name, prob, x1, y1, x2, y2, x3, y3, x4, y4  = box_info.strip().split()

        x1, y1, x2, y2, x3, y3, x4, y4 = float(x1),float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)

        #if os.path.exists(os.path.join(save_xml_dir, img_name+".xml")):
        # xml存在，则加载现有的xml
        tree = ET.parse(os.path.join(save_xml_dir, img_name+".xml"))
        annotation = tree.getroot()
        objs = annotation.find("objects")
        obj = ET.SubElement(objs, "object")

        ET.SubElement(obj, "coordinate").text = "pixel"
        ET.SubElement(obj, "type").text = "rectangle"
        ET.SubElement(obj, "description").text = "None"
        possibleresult = ET.SubElement(obj, "possibleresult")
        ET.SubElement(possibleresult, "name").text = fair1m_category_name
        ET.SubElement(possibleresult, "probability").text = str(prob)
        points = ET.SubElement(obj, "points")
        ET.SubElement(points, "point").text = "{:.6f},{:.6f}".format(x1,y1)
        ET.SubElement(points, "point").text = "{:.6f},{:.6f}".format(x2,y2)
        ET.SubElement(points, "point").text = "{:.6f},{:.6f}".format(x3,y3)
        ET.SubElement(points, "point").text = "{:.6f},{:.6f}".format(x4,y4)
        ET.SubElement(points, "point").text = "{:.6f},{:.6f}".format(x1,y1)

        tree.write(os.path.join(save_xml_dir, "{}.xml".format(img_name)), xml_declaration=True, encoding="utf-8", method="xml")


    print('Finish txt to xml for class {}: {}'.format(cnt, fair1m_category_name))
    cnt += 1

# 规整xml格式

save_new_xml_dir = os.path.join(args.txt_dir, "test")
os.makedirs(save_new_xml_dir, exist_ok=True)
xml_files = glob(os.path.join(save_xml_dir, "*.xml"))

for xml_file in tqdm(xml_files):

    _, filename = os.path.split(xml_file)

    # tree = ET.parse(xml_file)
    # annotation = tree.getroot()

    rawxml = minidom.parse(xml_file)

    #借用dom，添加缩进
    #rawtext = ET.tostring(annotation)
    #dom = minidom.parseString(rawtext)
    xml_pretty_str = rawxml.toprettyxml(encoding='utf-8')
    with open(os.path.join(save_new_xml_dir, filename), "wb") as f:
        f.write(xml_pretty_str)
    f.close()

print('Finish xml transformation!')













            





