import time
import datetime
import numpy as np
import platform
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from engine.TrainEngineLoc import TrainEngine
from datapu.imagenet_process_class import ImageNetProcessor

# ----------------------------------
# Set experimental params
# ----------------------------------
dataset_name = 'ilsvrc2014'
network_name = 'VGG16'
# train params
batch_size   = 196
img_resize   = 224
cls_num        = 1000
task_type      = 'multi-class'
gpus             = [0,1,2]
optimizer_type = 'sgd'
scheduler_type = 'max'
num_workers    = 6
# optimizer params
lr = 1e-2 * 3
wd = 1e-4
mt = 0.9
ss = 7
log_root = 'zlog/' + dataset_name + '-' + network_name + '-'  + time.strftime('%Y%m%d%H%M', time.localtime()) + '-' + optimizer_type + '-' + scheduler_type+ '-bs' + str(batch_size) + '-lrstep' + str(ss) + '-ims' + str(img_resize) + '-diff' + str(int(use_diff))
writer = SummaryWriter(log_root)

# ----------------------------------
# Create data env
# ----------------------------------
if(platform.system() =="Windows"): img_root = 'C:/dataset_research/ImageNet_ILSVRC2012'
else: img_root = '/home/penglu/Downloads/ImageNet/' # /mnt/storage/liuyufan/data/ImageNet_ILSVRC2012/ILSVRC2012_img_train/
print('dataset: %s, network: %s, input_size: %d, cls_num: %d' % (dataset_name, network_name, img_resize, cls_num))
# : create train dataloader
train_transformer = transforms.Compose([
       transforms.RandomResizedCrop(img_resize),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])   
train_dataset = ImageNetProcessor(dataset_root=img_root, dataset_type='train', objxml_root=None, transform=train_transformer)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset.dataset_, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
# : create val dataloader
val_transformer = transforms.Compose([
        transforms.Resize(img_resize+32),
        transforms.CenterCrop(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
val_dataset = ImageNetProcessor(dataset_root=img_root, dataset_type='val', objxml_root=None, transform=val_transformer)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset.dataset_, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)


# ----------------------------------
# Create experimental env
# ----------------------------------
train_engine = TrainEngine(lr=lr, mt=mt, wd=wd, ss=ss)
train_engine.create_env(net_name=network_name, task_type=task_type, cls_num=cls_num, pretrained=True, optimizer_type=optimizer_type, scheduler_type=scheduler_type, gpus=gpus)



# ----------------------------------
# Train & Val & Test
# ----------------------------------
epoch_s = 0
epoch_e = 60
best_train_mAP = 0.0
best_test_mAP = 0.0
for epoch_idx in range(epoch_s, epoch_e):
    # train
    train_top1, train_top5, train_mLoss, train_lr = train_engine.train_single_scale_ilsvrc(train_loader=train_loader, epoch_idx=epoch_idx)
    # test
    test_top1, test_top5, test_mLoss = train_engine.val_single_scale_ilsvrc(val_loader=val_loader, epoch_idx=epoch_idx)
    # check mAP and save
    train_engine.save_checkpoint(log_root, epoch_idx, best_train_mAP, best_test_mAP)            

    # curve all mAP & mLoss
    writer.add_scalars('top1', {'train': train_top1,  'valid': test_top1},  epoch_idx)
    writer.add_scalars('top5', {'train': train_top5,  'valid': test_top5},  epoch_idx)
    writer.add_scalars('loss', {'train': train_mLoss, 'valid': test_mLoss}, epoch_idx)
    # curve lr
    writer.add_scalar('train_lr', train_lr, epoch_idx)    
