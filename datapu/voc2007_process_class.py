import os
import csv
import copy
import numpy as np
import xml.etree.ElementTree as ET
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict
import platform


class VOC2007Processor(torch.utils.data.Dataset):
    # Func:
    #   Constructor
    def __init__(self, img_set_root, img_set_type, use_diff=False, img_label=True, obj_label=True):
        self.img_set_root_ = img_set_root
        self.img_set_type_ = img_set_type
        self.img_names_    = self.load_img_names()
        self.img_num_      = len(self.img_names_)
        self.transformer_  = None
        self.img_label_    = img_label
        self.obj_label_    = obj_label
        self.use_diff_     = use_diff
        self.obj_categories_ = ['aeroplane', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat', 'chair',
                                'cow', 'diningtable', 'dog', 'horse',
                                'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor']
        if self.img_label_:
            # load image-level label as one-hot
            self.img_labels_ = self.load_img_level_label()
            # load image-level label
            self.img_labels_onehot_ = self.load_img_level_label_onehot()
        if self.obj_label_:
            # load object-level label
            self.obj_labels_ = self.load_obj_level_label3(use_diff=self.use_diff_)

    # Func:
    #   Load image names.
    def load_img_names(self):
        img_names = []
        img_set_file = os.path.join(self.img_set_root_, 'ImageSets', 'Main', self.img_set_type_ + '.txt')
        assert os.path.exists(img_set_file), 'log: does not exist: {}'.format(img_set_file)
        # open image set file
        with open(img_set_file) as f:
            img_names = [line.strip() for line in f.readlines()]
        # return
        print('log: image num is %d' % (len(img_names)))
        return img_names

    # Func:
    #   Load image-level label, including image path and objec categories.
    #   Format: key - img_name, value - img_label
    def load_img_level_label(self):
        img_labels = OrderedDict()
        # init dict
        for i in range(self.img_num_):
            img_labels[self.img_names_[i]] = np.zeros(len(self.obj_categories_))
        # load label
        for i in range(len(self.obj_categories_)):
            category_file = os.path.join(self.img_set_root_, 'ImageSets', 'Main', self.obj_categories_[i] + '_' + self.img_set_type_ + '.txt')
            # open category file
            with open(category_file) as f:
                for line in f.readlines():
                    tmp_arr = line.split(' ')
                    img_name  = tmp_arr[0]
                    img_label = int(tmp_arr[-1])
                    img_labels[img_name][i] = img_label
        # return
        print('log: image-label num is %d' % (len(img_labels)))
        return img_labels


    def load_img_level_label_onehot(self):
        img_labels_onehot = []
        for key, value in self.img_labels_.items():
            img_label = copy.deepcopy(value)
            img_label = img_label.ravel()
            # transform label
            # : consider difficult in classifiation
            if self.use_diff_:
                img_label[img_label == 0] = 1
            # : transfer format for loss-func
            img_label[img_label == -1] = 0
            img_labels_onehot.append(img_label)
        return img_labels_onehot

    # Func:
    #   Load object-level label, including objec category, bbox.
    #   Type: dict, Format: key - img_name, value - [cls_name, bbox]
    def load_obj_level_label(self, use_diff=False):
        obj_labels = OrderedDict()
        obj_num = 0
        # init dict
        for i in range(self.img_num_):
            obj_labels[self.img_names_[i]] = []
        # load label        
        for img_idx, img_name in enumerate(self.img_names_):
            xml_name = os.path.join(self.img_set_root_, 'Annotations', img_name + '.xml')
            # : check
            if not os.path.exists(xml_name):
                print('log: miss %s' % xml_name)
                continue
            # load xml-tree
            xml_tree = ET.parse(xml_name)
            # read image size
            img_size = (xml_tree.find('size').find('height').text, xml_tree.find('size').find('width').text, xml_tree.find('size').find('depth').text)
            # read object info
            cls_infos = xml_tree.findall('object')
            # : check diff
            if not use_diff:
                cls_infos = [cls_info for cls_info in cls_infos if int(cls_info.find('difficult').text) == 0]
            # : read bbox 
            for cls_idx, cls_info in enumerate(cls_infos):
                cls_name = cls_info.find('name').text.lower().strip()
                obj_bbox = cls_info.find('bndbox')
                xmin = float(obj_bbox.find('xmin').text)
                ymin = float(obj_bbox.find('ymin').text)
                xmax = float(obj_bbox.find('xmax').text)
                ymax = float(obj_bbox.find('ymax').text)
                obj_labels[img_name].append((cls_name, (xmin, ymin, xmax, ymax)))
                obj_num += 1
        # return
        print('log: object-label num is %d' % obj_num)
        return obj_labels

    # Func:
    #   Load object-level label
    # Return:
    #   type is list, format is [img_name, cls_name, (xmin, ymin, xmax, ymax)]
    def load_obj_level_label2(self, use_diff=False):
        obj_labels = []
        obj_num = 0
        # load label        
        for img_idx, img_name in enumerate(self.img_names_):
            xml_name = os.path.join(self.img_set_root_, 'Annotations', img_name + '.xml')
            # : check
            if not os.path.exists(xml_name):
                print('log: miss %s' % xml_name)
                continue
            # load xml-tree
            xml_tree = ET.parse(xml_name)
            # read image size
            img_size = (xml_tree.find('size').find('height').text, xml_tree.find('size').find('width').text, xml_tree.find('size').find('depth').text)
            # read object info
            cls_infos = xml_tree.findall('object')
            # : check diff
            if not use_diff:
                cls_infos = [cls_info for cls_info in cls_infos if int(cls_info.find('difficult').text) == 0]
            # : read bbox 
            for cls_idx, cls_info in enumerate(cls_infos):
                cls_name = cls_info.find('name').text.lower().strip()
                obj_bbox = cls_info.find('bndbox')
                xmin = float(obj_bbox.find('xmin').text)
                ymin = float(obj_bbox.find('ymin').text)
                xmax = float(obj_bbox.find('xmax').text)
                ymax = float(obj_bbox.find('ymax').text)
                obj_labels.append((img_name, cls_name, (xmin, ymin, xmax, ymax)))
                obj_num += 1
        # return
        print('log: object-label num is %d' % obj_num)
        return obj_labels

    # Func:
    #   Load object-level label
    # Return:
    #   type is list, format is [iid, cid, xmin, ymin, xmax, ymax]
    def load_obj_level_label3(self, use_diff=False):
        obj_labels = []
        obj_num = 0
        # load label        
        for img_idx, img_name in enumerate(self.img_names_):
            xml_name = os.path.join(self.img_set_root_, 'Annotations', img_name + '.xml')
            # : check
            if not os.path.exists(xml_name):
                print('log: miss %s' % xml_name)
                continue
            # load xml-tree
            xml_tree = ET.parse(xml_name)
            # read image size
            img_size = (xml_tree.find('size').find('height').text, xml_tree.find('size').find('width').text, xml_tree.find('size').find('depth').text)
            # read object info
            cls_infos = xml_tree.findall('object')
            # : check diff
            if not use_diff:
                cls_infos = [cls_info for cls_info in cls_infos if int(cls_info.find('difficult').text) == 0]
            # : read bbox 
            for cls_idx, cls_info in enumerate(cls_infos):
                obj_category_name = cls_info.find('name').text.lower().strip()
                obj_category_id = self.obj_categories_.index(obj_category_name)
                obj_bbox = cls_info.find('bndbox')
                xmin = float(obj_bbox.find('xmin').text)
                ymin = float(obj_bbox.find('ymin').text)
                xmax = float(obj_bbox.find('xmax').text)
                ymax = float(obj_bbox.find('ymax').text)
                obj_labels.append((img_idx, obj_category_id, xmin, ymin, xmax, ymax))
                obj_num += 1
        # return
        print('log: object-label num is %d' % obj_num)
        return obj_labels

    # Func:
    #   Create groundtruth files
    def create_groundtruth_files(self, grth_root):
        # check
        assert len(self.obj_labels_) > 0, 'log-err: no object labels'
        if not os.path.exists(grth_root): os.makedirs(grth_root)
        # create
        for img_name, obj_labels in self.obj_labels_.items():
            # Type: dict, Format: key - img_name, value - [cls_name, bbox]
            grth_file = os.path.join(grth_root, img_name + '.txt')
            with open(grth_file, 'w') as f:
                for cls_name, obj_bbox in obj_labels:
                    left  = str(int(obj_bbox[0]))
                    top   = str(int(obj_bbox[1]))
                    right = str(int(obj_bbox[2]))
                    bottom= str(int(obj_bbox[3]))
                    f.write(cls_name + ' ' + left + ' ' + top + ' ' + right + ' ' + bottom + '\n')

    # Func:
    #   Create detection files.
    # Input:
    #   grth_bboxes : groundtruth bboxes,   type is numpy.ndarray, format is [iid, cid, xmin, ymin, xmax, ymax, score]
    def create_groundtruth_files2(self, grth_bboxes, det_root):
        # check
        assert len(grth_bboxes) > 0, 'log-err: no object'
        if not os.path.exists(det_root): os.makedirs(det_root)
        # create
        for pred_bbox in grth_bboxes:
            img_name = self.img_names_[int(pred_bbox[0])]
            cls_name = self.obj_categories_[int(pred_bbox[1])]
            det_file = os.path.join(det_root, img_name + '.txt')
            with open(det_file, 'a') as f:
                score = str(pred_bbox[6])
                left  = str(int(pred_bbox[2]))
                top   = str(int(pred_bbox[3]))
                right = str(int(pred_bbox[4]))
                bottom= str(int(pred_bbox[5]))               
                f.write(cls_name + ' ' + score + ' ' + left + ' ' + top + ' ' + right + ' ' + bottom + '\n')    

    # Func:
    #   Create detection files.
    # Input:
    #   pred_bboxes : predicted bboxes,   type is numpy.ndarray, format is [iid, cid, xmin, ymin, xmax, ymax, score]
    def create_detection_files(self, pred_bboxes, det_root):
        # check
        assert len(pred_bboxes) > 0, 'log-err: no object'
        if not os.path.exists(det_root): os.makedirs(det_root)
        # create
        for pred_bbox in pred_bboxes:
            img_name = self.img_names_[int(pred_bbox[0])]
            cls_name = self.obj_categories_[int(pred_bbox[1])]
            det_file = os.path.join(det_root, img_name + '.txt')
            with open(det_file, 'a') as f:
                score = str(pred_bbox[6])
                left  = str(int(pred_bbox[2]))
                top   = str(int(pred_bbox[3]))
                right = str(int(pred_bbox[4]))
                bottom= str(int(pred_bbox[5]))               
                f.write(cls_name + ' ' + score + ' ' + left + ' ' + top + ' ' + right + ' ' + bottom + '\n')    

    # Func:
    #   Get item for classification.
    def __getitem__(self, index):
        img_name = self.img_names_[index]
        img_path = os.path.join(self.img_set_root_, 'JPEGImages', img_name + '.jpg')


        # img_name = '000026'
        # if(platform.system() =="Windows"): img_path = 'c:/dataset_research/VOC2007/VOCdevkit/VOC2007/JPEGImages/' + img_name + '.jpg'
        # else: img_path = '/home/bowang/ds_research/VOC2007/VOCdevkit/VOC2007/JPEGImages/' + img_name + '.jpg'



        # open image
        img = Image.open(img_path).convert('RGB')
        img_size = img.size[::-1]   # [h, w]
        # transform image
        if self.transformer_ is not None:
            img = self.transformer_(img)
        # transform label
        if self.img_label_:
            img_label = copy.deepcopy(self.img_labels_[img_name])
            # consider difficult in classifiation
            if self.use_diff_:
                img_label[img_label == 0] = 1
            # transfer format for loss-func
            img_label[img_label == -1] = 0
            img_label = torch.from_numpy(img_label).float()
            # return
            return img, img_label, img_name, img_size
        else:
            return img, img_name, img_size

    def __len__(self):
        return len(self.img_names_)