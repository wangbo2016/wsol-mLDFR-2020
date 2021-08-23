import os
import copy
import torch
import numpy as np
import xml.etree.ElementTree as ET
import torchvision.datasets as datasets
from PIL import Image
from collections import OrderedDict

class ImageNetProcessor(torch.utils.data.Dataset):
    def __init__(self, dataset_root, dataset_type, load_obj_label=False, transform=None):
        self.dataset_root_ = os.path.join(dataset_root, dataset_type) 
        self.dataset_type_ = dataset_type
        self.transformer_  = transform
        # create dataset
        self.dataset_ = datasets.ImageFolder(self.dataset_root_, transform=transform)
        # load category info
        self.categories_ = self.dataset_.classes
        self.cname2cid_  = self.dataset_.class_to_idx
        # load image-level annos
        self.img_names_ = []
        self.img_paths_ = []
        self.img_labels_= []
        self.load_img_annos()        
        # load object-level annos
        if load_obj_label:
            self.obj_labels_ = []
            self.load_obj_annos()
        # create iname-iid mapping
        self.img_iname2iid_ = OrderedDict()
        for i in range(len(self.img_names_)):
            img_name = os.path.basename(self.img_names_[i]).replace('.JPEG', '').replace('.jpg', '')
            self.img_iname2iid_[img_name] = i

    # Func:
    #   Load image-level label.
    def load_img_annos(self):
        # load image label
        for img_path, img_cid in self.dataset_.imgs:
            self.img_names_.append(os.path.basename(img_path))
            self.img_paths_.append(img_path)
            self.img_labels_.append(img_cid)        
        # log
        if self.dataset_type_ == 'train': print('log: train image-label num is %d' % len(self.img_labels_))
        else: print('log: val image-label num is %d' % len(self.img_labels_))

    # Func:
    #   Load object-level label, including objec category, bbox.
    # Return:
    #   type is list, format is [iid, cid, xmin, ymin, xmax, ymax]    
    def load_obj_annos(self):
        objxml_root = self.dataset_root_ + '_xml'
        assert os.path.exists(objxml_root), 'log: does not exist: {}'.format(objxml_root)
        # load label        
        for img_idx, img_path in enumerate(self.img_names_):
            img_name = os.path.basename(img_path)
            _, file_ext = os.path.splitext(img_path)
            xml_name = os.path.join(objxml_root, img_name.replace(file_ext, '') + '.xml')
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
            for cls_idx, cls_info in enumerate(cls_infos):
                obj_category_name = cls_info.find('name').text.lower().strip()
                obj_category_id = self.categories_.index(obj_category_name)
                obj_bbox = cls_info.find('bndbox')
                xmin = int(obj_bbox.find('xmin').text)
                ymin = int(obj_bbox.find('ymin').text)
                xmax = int(obj_bbox.find('xmax').text)
                ymax = int(obj_bbox.find('ymax').text)
                self.obj_labels_.append((img_idx, obj_category_id, xmin, ymin, xmax, ymax))
        # log
        if self.dataset_type_ == 'train': print('log: train object-label num is %d' % len(self.img_labels_))
        else: print('log: val object-label num is %d' % len(self.obj_labels_))        

    # Func:
    #   Get item for classification.
    def __getitem__(self, index):
        # load
        img_name = self.img_names_[index]
        img_path = os.path.join(self.dataset_root_, 'images', self.img_paths_[index])
        # open image
        img_data = Image.open(img_path).convert('RGB')
        img_size = img_data.size[::-1]   # [h, w]
        # transform image
        if self.transformer_ is not None:
            img_data = self.transformer_(img_data)
        # transform label
        img_label = copy.deepcopy(self.img_labels_[index])
        # return
        return img_data, img_label, img_path, img_size


    def __len__(self):
        return len(self.img_names_)

