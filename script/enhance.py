# -*- coding: utf-8 -*-
# @Author: lushujie

# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np
import os

pic_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor_extend/"
xml_path = "/home/cc/ros_ws/sim_ws/rolling_ws/src/local_ssh/data/Floor_extend/"

classes = ["gateway", "corner", "unknown"]
pic_type = "png"

ia.seed(1)

# Sometimes(0.3, ...) applies the given augmenter in 50% of all cases,
sometimes = lambda aug: iaa.Sometimes(0.3, aug)


def enhance_seq():
    oneof_seq = []
    seq_list = []
    
    seq_list.append(sometimes(iaa.Affine(
        scale={"x": (1.0, 1.0), "y": (1.0, 1.0)},
        translate_percent={"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
        rotate=(-180, 180),
        shear=(-0, 0)
    )))

    seq = iaa.Sequential(seq_list, random_order=True)
    return seq

def load_box(imge_name):
    # more than one bbox
    name = imge_name.split('.')[0] + "." + imge_name.split('.')[1]
    xml_name = name + '.xml'
    xml_localpath = os.path.join(xml_path, xml_name)
    # print(xml_localpath+"  "+imge_name)
    in_file = open(xml_localpath,'r')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    boxs = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        # cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        onebox = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                  float(xmlbox.find('ymax').text)]
        # print(str(cls_id)+" "+str(onebox))
        obj = [cls, onebox]
        boxs.append(obj)
    return boxs


def load_batch():
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    final_result = []
    filelist = os.listdir(pic_path)
    readlist = []
    readbox = []
    for f in filelist:
        if f.split('.')[-1] == pic_type:
            cur_pic_path = os.path.join(pic_path, f) #图片地址
            rgbimg = imageio.imread(cur_pic_path)
            # print(cur_pic_path)
            # rgbimg = cv2.imread(cur_pic_path)
            rgbimg = np.asarray(rgbimg, dtype=np.uint8)
            # readlist.append(rgbimg)
            boxs = load_box(f)
            ibox = [cur_pic_path, boxs] # 图和标注框
            # readbox.append(ibox)
            current_name_img_box = [f.split('.')[0] + "." + f.split('.')[1],rgbimg,ibox]
            final_result.append(current_name_img_box)

    return final_result


def make_xml(newname, h, w, c, bbs_aug):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Floor2'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(newname) + '.' + pic_type

    node_filepath = SubElement(node_root, 'path')
    node_filepath.text = pic_path + str(newname) + '.' + pic_type

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = "Unknown"


    # node_object_num = SubElement(node_root, 'object_num')
    # node_object_num.text = str(len(xmin_tuple))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(w)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(h)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(c)

    for i in range(len(bbs_aug.bounding_boxes)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(bbs_aug.bounding_boxes[i].label)
        
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = "Unspecified"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbs_aug.bounding_boxes[i].x1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbs_aug.bounding_boxes[i].y1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(bbs_aug.bounding_boxes[i].x2)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbs_aug.bounding_boxes[i].y2)

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    return dom


def wirte_img_xml(newname, img, bbs_aug):
    # (R, G, B) = cv2.split(img)
    # get = cv2.merge([B, G, R])
    img_path = os.path.join(pic_path, newname + '.'+ pic_type)
    cv2.imwrite(img_path, img)

    h, w = img.shape
    dom = make_xml(newname, h, w, 1, bbs_aug)
    xml_name = os.path.join(xml_path, newname + '.xml')
    with open(xml_name, 'wb') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


def start_img_enhance():
    print('start_img_enhance,please wait a few seconds')
    name_img_box = load_batch()
    for times in range(3):
        for i in name_img_box:
            ori_name = i[0]
            current_image = i[1]
            img_boxs_id = i[2]
            all_box = []
            for j in range(len(img_boxs_id[1])):
                current_box = BoundingBox(img_boxs_id[1][j][1][0], img_boxs_id[1][j][1][1],
                                          img_boxs_id[1][j][1][2], img_boxs_id[1][j][1][3],
                                          label=img_boxs_id[1][j][0])
                all_box.append(current_box)
            bbs = BoundingBoxesOnImage(all_box, shape=current_image.shape)
            seq = enhance_seq()
            image_aug, bbs_aug = seq(image=current_image, bounding_boxes=bbs)
            newname = 'en_' + str(times) + '_' + str(ori_name)
            wirte_img_xml(newname, image_aug, bbs_aug)

    print("end img_enhance")


if __name__ == '__main__':
    start_img_enhance()