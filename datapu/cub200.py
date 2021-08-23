import os
import copy
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from collections import OrderedDict

class CUBProcessor(torch.utils.data.Dataset):
    # Func:
    #   Constructor
    def __init__(self, dataset_root, dataset_type, load_obj_label=False, transform=None):
        self.dataset_root_ = dataset_root
        self.dataset_type_ = dataset_type
        self.transformer_  = transform
        # load category info
        self.categories_ = []
        self.cid2cidx_   = OrderedDict()
        self.load_category_info()
        # load train/test split
        self.train_iid2iidx_ = OrderedDict()
        self.test_iid2iidx_  = OrderedDict()
        self.train_iid_ = []
        self.test_iid_  = []        
        self.load_split()
        # load image-level annos
        self.img_names_ = []    # iidx2iname
        self.img_paths_ = []    # iidx2ipath
        self.img_labels_ = []
        self.load_img_names()
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
    #   Load category infomations.
    def load_category_info(self):
        # load category info
        obj_category_file = os.path.join(self.dataset_root_, 'classes.txt')
        assert os.path.exists(obj_category_file), 'log: does not exist: {}'.format(obj_category_file)
        # open image set file
        obj_category_infos = []
        with open(obj_category_file) as f:
            obj_category_infos = [line.strip() for line in f.readlines()]
        # load cid & cname
        for cinfo in obj_category_infos:
            tmp   = cinfo.split(' ')
            cid   = int(tmp[0]) - 1
            cname = tmp[1]
            self.cid2cidx_[cid] = len(self.cid2cidx_)
            self.categories_.append(cname)
        # return
        print('log: category num is %d' % (len(obj_category_infos)))

    def load_split(self):
        split_file = os.path.join(self.dataset_root_, 'train_test_split.txt')
        assert os.path.exists(split_file), 'log: does not exist: {}'.format(split_file)
        # open image anno file
        split_infos = []
        with open(split_file) as f:
            split_infos = [line.strip() for line in f.readlines()]
        # load image path & name
        for sinfo in split_infos:
            tmp = sinfo.split(' ')
            iid = int(tmp[0]) - 1
            typ = int(tmp[1])
            if typ == 1: 
                self.train_iid2iidx_[iid] = len(self.train_iid2iidx_)
                self.train_iid_.append(iid)
            else: 
                self.test_iid2iidx_[iid] = len(self.test_iid2iidx_)
                self.test_iid_.append(iid)
        # log
        if self.dataset_type_ == 'train': print('log: train image num is %d' % (len(self.train_iid2iidx_)))
        else: print('log: test image num is %d' % (len(self.test_iid2iidx_)))

    # Func:
    #   Load image names.
    def load_img_names(self):
        img_path_file = os.path.join(self.dataset_root_, 'images.txt')
        assert os.path.exists(img_path_file), 'log: does not exist: {}'.format(img_path_file)
        # open image path file
        img_path_infos = []
        with open(img_path_file) as f:
            img_path_infos = [line.strip() for line in f.readlines()]
        # load image path & name
        for iinfo in img_path_infos:
            tmp   = iinfo.split(' ')
            iid   = int(tmp[0]) - 1
            ipath = tmp[1]
            iname = ipath.split('/')[-1]
            if self.dataset_type_ == 'train' and iid in self.train_iid_:
                self.img_names_.append(iname)
                self.img_paths_.append(ipath)
            if self.dataset_type_ == 'test'  and iid in self.test_iid_:
                self.img_names_.append(iname)
                self.img_paths_.append(ipath)                

    # Func:
    #   Load image-level label, including image path and object categories.
    #   Format: list. We assume that the index is the image id.
    def load_img_annos(self):
        img_anno_file = os.path.join(self.dataset_root_, 'image_class_labels.txt')
        assert os.path.exists(img_anno_file), 'log: does not exist: {}'.format(img_anno_file)
        # open image anno file
        img_anno_infos = []
        with open(img_anno_file) as f:
            img_anno_infos = [line.strip() for line in f.readlines()]
        # load image path & name
        for ianno in img_anno_infos:
            tmp = ianno.split(' ')
            iid = int(tmp[0]) - 1
            cid = int(tmp[1]) - 1
            cidx = self.cid2cidx_[cid]
            if self.dataset_type_ == 'train' and iid in self.train_iid_:
                self.img_labels_.append(cidx)
            if self.dataset_type_ == 'test'  and iid in self.test_iid_:
                self.img_labels_.append(cidx)
    
    # Func:
    #   Load object-level label, including objec category, bbox.
    # Return:
    #   type is list, format is [iidx, cidx, xmin, ymin, xmax, ymax]
    def load_obj_annos(self, use_diff=False):
        obj_anno_file = os.path.join(self.dataset_root_, 'bounding_boxes.txt')
        assert os.path.exists(obj_anno_file), 'log: does not exist: {}'.format(obj_anno_file)
        # open object anno file
        obj_anno_infos = []
        with open(obj_anno_file) as f:
            obj_anno_infos = [line.strip() for line in f.readlines()]
        # load object infos
        for oanno in obj_anno_infos:
            tmp  = oanno.split(' ')
            iid  = int(tmp[0]) - 1
            xmin = float(tmp[1])
            ymin = float(tmp[2])
            xmax = xmin + float(tmp[3])
            ymax = ymin + float(tmp[4])
            if self.dataset_type_ == 'train' and iid in self.train_iid_:
                iidx = self.train_iid2iidx_[iid]
                cidx = self.img_labels_[iidx]
                self.obj_labels_.append((iidx, cidx, xmin, ymin, xmax, ymax))                
            if self.dataset_type_ == 'test'  and iid in self.test_iid_:
                iidx = self.test_iid2iidx_[iid]
                cidx = self.img_labels_[iidx]
                self.obj_labels_.append((iidx, cidx, xmin, ymin, xmax, ymax))
    
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
        if self.dataset_type_ == 'train': return len(self.train_iid2iidx_)
        else: return len(self.test_iid2iidx_)
