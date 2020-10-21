import os
import sys
import time
import datetime
import numpy as np
import platform
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from engine.TestEngine import TestEngine
from datapu.imagenet_process_class import ImageNetProcessor

# ----------------------------------
# Set experimental params
# ----------------------------------
batch_size = 1
img_resize = 224    # 112, 224, 336, 448, 560
img_type   ='val'
use_diff   = False
cls_num    = 1000
gpus       = [0]
num_workers= 0
dataset_name = 'ilsvrc2014'
network_name = 'VGG16'
# set dataset path
img_root = '/home/penglu/Downloads/ImageNet/' 
objxml_root = '/home/bowang/ds_research/ILSVRC2014/ILSVRC_XML/val'
print('dataset: %s, network: %s, input_size: %d, cls_num: %d' % (dataset_name, network_name, img_resize, cls_num))
# set model path
checkpoint_path = 'zlog/loc-vgg16ybn-cub200-model-gap-f14-sgd-max-b128-100.0-100.0-81.02-95.09/20200422-09-08-46.pth'

# ----------------------------------
# Create data env
# ----------------------------------
# : create val dataloader
val_dataset = ImageNetProcessor(dataset_root=img_root, dataset_type=img_type, objxml_root=None, transform=None)
val_dataset.transformer_  = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

# ----------------------------------
# Create experimental env
# ----------------------------------
test_engine = TestEngine(ds_name=dataset_name, img_names=val_dataset.img_names_, img_labels=val_dataset.img_labels_, obj_bboxes=None, obj_categories=val_dataset.obj_categories_)
test_engine.create_env(checkpoint_path=checkpoint_path, net_name=network_name, task_type='multi-class', cls_num=cls_num, parallel=True, gpus=gpus)

# ----------------------------------
# Val & Test
# ----------------------------------
# classification 
test_engine.val_single_scale_for_classification_ilsvrc(val_loader=val_loader, res_file='loc-vgg16y-ilsvrc-model-gap-csatt-f14-sgd-max-b256-70.33088.09-73.79-91.62-44-loc.preds_224')

