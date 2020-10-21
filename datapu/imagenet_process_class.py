import torch.utils.data as data
import json
import copy
import os
import subprocess
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import torch
import pickle
from PIL import ImageFile
from collections import OrderedDict
import torchvision.datasets as datasets


class ImageNetProcessor(data.Dataset):
    def __init__(self, dataset_root, dataset_type, objxml_root=None, use_diff=False, transform=None):
        self.dataset_root_ = os.path.join(dataset_root, dataset_type) 
        self.dataset_type_ = dataset_type
        self.objxml_root_  = objxml_root
        self.use_diff_     = use_diff

        # create dataset
        self.dataset_ = datasets.ImageFolder(self.dataset_root_, transform=transform)
        # load category info
        self.obj_categories_ = self.dataset_.classes
        self.obj_cname2cid_ = self.dataset_.class_to_idx
        # load image-level annos
        self.img_names_ = []
        self.img_paths_ = []
        self.img_labels_= []
        self.img_labels_onehot_ = []
        self.load_img_annos()        
        self.img_num_ = len(self.img_names_)
        # create iname-iid mapping
        self.img_iname2iid_ = OrderedDict()
        for i in range(len(self.img_names_)):
            img_name = os.path.basename(self.img_names_[i]).replace('.JPEG', '').replace('.jpg', '')
            self.img_iname2iid_[img_name] = i
        # load object-level annos
        if self.objxml_root_ is not None: 
            self.obj_labels_ = self.load_obj_annos()

        # # test
        # black_list_file = 'C:/dataset_research/ImageNet_ILSVRC2014/ILSVRC2014_devkit/data_backup/ILSVRC2014_clsloc_validation_blacklist.txt'
        # black_list = []
        # with open(black_list_file) as f:
        #     black_list = [int(line.strip()) for line in f.readlines()]

        # from PIL import Image, ImageDraw, ImageFont
        # grth_bboxes = np.array(self.obj_labels_)
        # for img_idx, img_name in enumerate(self.img_names_):
        #     if img_idx not in black_list: continue
        #     img_path   = self.img_paths_[img_idx]
        #     img_pil  = Image.open(img_path)
        #     img_bboxes = grth_bboxes[grth_bboxes[:,0] == img_idx, :]
        #     for img_bbox in img_bboxes:
        #         iid, cid, xmin, ymin, xmax, ymax = img_bbox
        #         # draw rectangle
        #         ImageDraw.Draw(img_pil).rectangle((xmin, ymin, xmax, ymax), outline='blue')
        #     # save
        #     img_pil.save(img_name + '.jpg') 

    # Func:
    #   Load image-level label.
    def load_img_annos(self):
        # load image label
        for img_path, img_cid in self.dataset_.imgs:
            self.img_names_.append(os.path.basename(img_path))
            self.img_paths_.append(img_path)
            self.img_labels_.append(img_cid)        
        # load image label as ont-hot
        for img_idx in range(len(self.img_labels_)):
            img_labels = self.img_labels_[img_idx]
            img_labels_onehot = np.zeros(len(self.obj_cname2cid_), np.float32)
            img_labels_onehot[img_labels] = 1
            self.img_labels_onehot_.append(img_labels_onehot)
        print('log: image-label num is %d' % len(self.img_labels_))

    # Func:
    #   Load object-level label, including objec category, bbox.
    # Return:
    #   type is list, format is [iid, cid, xmin, ymin, xmax, ymax]    
    def load_obj_annos(self):
        obj_labels = []
        obj_num = 0
        # load label        
        for img_idx, img_path in enumerate(self.img_names_):
            img_name = os.path.basename(img_path)
            _, file_ext = os.path.splitext(img_path)
            xml_name = os.path.join(self.objxml_root_, img_name.replace(file_ext, '') + '.xml')
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
            if not self.use_diff_:
                cls_infos = [cls_info for cls_info in cls_infos if int(cls_info.find('difficult').text) == 0]
            # : read bbox 
            for cls_idx, cls_info in enumerate(cls_infos):
                obj_category_name = cls_info.find('name').text.lower().strip()
                obj_category_id = self.obj_categories_.index(obj_category_name)
                obj_bbox = cls_info.find('bndbox')
                xmin = int(obj_bbox.find('xmin').text)
                ymin = int(obj_bbox.find('ymin').text)
                xmax = int(obj_bbox.find('xmax').text)
                ymax = int(obj_bbox.find('ymax').text)
                obj_labels.append((img_idx, obj_category_id, xmin, ymin, xmax, ymax))
                obj_num += 1
        # return
        print('log: object-label num is %d' % obj_num)
        return obj_labels

    # Func:
    #   Load object-level label, including objec category, bbox.
    #   Type: dict, Format: key - img_name, value - [cls_name, bbox]
    def load_obj_annos2(self, use_diff=False):
        obj_labels = OrderedDict()
        obj_num = 0
        # init dict
        for i in range(self.img_num_):
            img_name = os.path.basename(self.img_names_[i])
            obj_labels[img_name] = []
        # load label        
        for img_idx, img_path in enumerate(self.img_names_):
            img_name = os.path.basename(img_path)
            _, file_ext = os.path.splitext(img_path)
            xml_name = os.path.join(self.objxml_root_, img_name.replace(file_ext, '') + '.xml')
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
                cls_id   = self.obj_categories_.index(cls_name)
                obj_bbox = cls_info.find('bndbox')
                xmin = int(obj_bbox.find('xmin').text)
                ymin = int(obj_bbox.find('ymin').text)
                xmax = int(obj_bbox.find('xmax').text)
                ymax = int(obj_bbox.find('ymax').text)
                obj_labels[img_name].append((cls_id, xmin, ymin, xmax, ymax))
                obj_num += 1
        # return
        print('log: object-label num is %d' % obj_num)
        return obj_labels

    # Func:
    #   Load object-level label of trainset, including objec category, bbox.
    #   Type: dict, Format: key - img_name, value - [cls_name, bbox]
    def load_obj_annos3_train(self, use_diff=False):
        obj_labels = OrderedDict()
        obj_num = 0
        # load xml
        xml_names = []
        self.traverse_file_pathes_recursively(self.objxml_root_, xml_names)
        # init dict
        for i in range(len(xml_names)):
            img_name = os.path.basename(xml_names[i])
            obj_labels[img_name] = []
        # load label        
        for img_idx, xml_name in enumerate(xml_names):
            img_name = os.path.basename(xml_name)
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
                obj_labels[img_name].append((cls_name, xmin, ymin, xmax, ymax))
                obj_num += 1
        # return
        print('log: object-label num is %d' % obj_num)
        return obj_labels

    # Func:
    #   Traverse all file path in specialed directory recursively.
    def traverse_file_pathes_recursively(self, fileroot, filepathes, extension='.*'):
        items = os.listdir(fileroot)
        for item in items:
            if os.path.isfile(os.path.join(fileroot, item)):
                filepath = os.path.join(fileroot, item)
                fileext = self.get_file_ext(filepath)
                if extension == '.*':
                    filepathes.append(filepath)
                elif fileext == extension:
                    filepathes.append(filepath)
                else:
                    pass                    
            elif os.path.isdir(os.path.join(fileroot, item)):
                self.traverse_file_pathes_recursively(os.path.join(fileroot, item), filepathes, extension)
            else:
                pass

    def get_file_name(self, filepath):
        return os.path.basename(filepath)

    def get_file_ext(self, filepath):
        return self.get_file_name_ext(filepath)[1]

    def get_file_name_ext(self, filepath):
        # analyze
        file_name, file_ext = os.path.splitext(filepath)
        # return
        return file_name, file_ext


    # Func:
    #   Get item for classification.
    def __getitem__(self, index):
        # # load
        # img_name = self.img_names_[index]
        # img_path = os.path.join(self.dataset_root_, 'images', self.img_paths_[index])
        # # open image
        # img = Image.open(img_path).convert('RGB')
        # img_size = img.size[::-1]   # [h, w]
        # # transform image
        # if self.transformer_ is not None:
        #     img = self.transformer_(img)
        # # transform label
        # img_label = copy.deepcopy(self.img_labels_[index])
        # # return
        # return img, img_label, img_path, img_size






        # # --------------------------------------------
        # # Localization
        # # --------------------------------------------
        # # load
        # img_name = self.img_names_[index]
        # img_path = os.path.join(self.dataset_root_, 'images', self.img_paths_[index])
        # # open image
        # img = Image.open(img_path).convert('RGB')
        # img_size = img.size[::-1]   # [h, w]        
        # # transform image
        # img = torch.zeros(1,1)
        # # transform label
        # img_label = copy.deepcopy(self.img_labels_[index])
        # # return
        # return img, img_label, img_path, img_size       




        # --------------------------------------------
        # Visualize
        # --------------------------------------------

        # # load
        # img_name = self.img_names_[index]
        # img_path = os.path.join(self.dataset_root_, 'images', self.img_paths_[index])

        # load
        img_path = '/home/penglu/Downloads/ImageNet/val/n02113799/ILSVRC2012_val_00024272.JPEG'
        img_path = r'C:\dataset_research\ImageNet_ILSVRC2012\val\n01796340\ILSVRC2012_val_00024630.JPEG'
        img_name = os.path.basename(img_path).replace('.JPEG', '').replace('.jpg', '')
        index = self.img_iname2iid_[img_name]

        # open image
        img = Image.open(img_path).convert('RGB')
        img_size = img.size[::-1]   # [h, w]
        # transform image
        if self.transformer_ is not None:
            img = self.transformer_(img)
        # transform label
        img_label = copy.deepcopy(self.img_labels_[index])
        # return
        return img, img_label, img_path, img_size, index


    def __len__(self):
        return len(self.img_names_)