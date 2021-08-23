import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from tools.cmetric import MultiClassificationMetric, MultilabelClassificationMetric, accuracy
from models.vgg16bnlocalizerhma import VGG16BNLocalizer
from models.googlenetlocalizerhma import GoogLeNetBNLocalizer
from models.resnet50localizerhma import ResNet50Localizer
from models.inceptionv3localizerhma import InceptionV3Localizer
from models.densenet161localizerhma import DenseNet161Localizer

class TrainEngine(object):
    # Func:
    #   Constructor.
    def __init__(self, cfg):
        # init setting
        self.device_ = ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg_    = cfg
        # create tool
        self.cls_meter_  = MultilabelClassificationMetric()
        self.los_meter_  = MultiClassificationMetric()
        self.top1_meter_ = MultiClassificationMetric()
        self.top5_meter_ = MultiClassificationMetric()

    # Func:
    #   Create experimental environment 
    def create_env(self):
        # create network 
        if   self.cfg_.network.name == 'vgg16bn':
            self.netloc_ = VGG16BNLocalizer(cls_num=self.cfg_.network.out_size)  
        elif self.cfg_.network.name == 'googlenetbn':
            self.netloc_ = GoogLeNetBNLocalizer(cls_num=self.cfg_.network.out_size)                   
        elif self.cfg_.network.name == 'inceptionv3':
            self.netloc_ = InceptionV3Localizer(cls_num=self.cfg_.network.out_size)
        elif self.cfg_.network.name == 'resnet50':
            self.netloc_ = ResNet50Localizer(cls_num=self.cfg_.network.out_size)  
        elif self.cfg_.network.name == 'densenet161':
            self.netloc_ = DenseNet161Localizer(cls_num=self.cfg_.network.out_size)   

        if self.cfg_.train.resume_path is not None:
            # load state_dict(with dataparallel)
            checkpoint = torch.load(self.cfg_.train.resume_path)
            state_dict = checkpoint['state_dict']
            # create new OrderedDict that does not contain `module.`
            state_dict_new = OrderedDict()
            # remove `module.`
            for k, v in state_dict.items():
                state_dict_new[k[7:]] = v
            # load params
            self.netloc_.load_state_dict(state_dict_new)
        else:
            # load pretrained params
            self.netloc_.init(params_path=self.cfg_.train.params_path)
        print(self.netloc_)
        # set parallel & cuda
        self.netloc_ = torch.nn.DataParallel(module=self.netloc_, device_ids=self.cfg_.train.device_ids)
        self.netloc_.cuda(device=self.cfg_.train.device_ids[0])
        # create loss function
        if   self.cfg_.train.type == 'multi-label':
            self.criterion_ = nn.MultiLabelSoftMarginLoss().cuda()
        elif self.cfg_.train.type == 'multi-class':
            self.criterion_ = nn.CrossEntropyLoss().cuda()
        # create optimizer
        if   self.cfg_.train.optimizer.name == 'sgd':
            self.optimizer_ = torch.optim.SGD(self.netloc_.parameters(), lr=self.cfg_.train.optimizer.lr, momentum=self.cfg_.train.optimizer.momentum, weight_decay=self.cfg_.train.optimizer.weight_decay)
        elif self.cfg_.train.optimizer.name == 'adam':
            self.optimizer_ = torch.optim.Adam(self.netloc_.parameters(), lr=self.cfg_.train.optimizer.lr, weight_decay=self.cfg_.train.optimizer.weight_decay)
        # create scheduler
        if   self.cfg_.train.scheduler.name == 'step':
            self.scheduler_ = torch.optim.lr_scheduler.StepLR(self.optimizer_, step_size=self.ss_, gamma=0.1)
        elif self.cfg_.train.scheduler.name == 'max':
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='max', factor=0.1, patience=2)
        elif self.cfg_.train.scheduler.name == 'min':
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='min', factor=0.1, patience=2)  

        if self.cfg_.train.resume_path is not None:
            self.optimizer_.load_state_dict(checkpoint['optimizer'])
            self.scheduler_.load_state_dict(checkpoint['scheduler'])
            self.lr_ = checkpoint['lr']

    # Func:
    #   Train multi-label model.
    def train_multi_label(self, train_loader, epoch_idx):
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.train()
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # train
        train_loader = tqdm(train_loader, desc='train', ascii=True)
        for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(train_loader):
            # set cuda
            imgs_tensor = imgs_tensor.cuda()
            imgs_label = imgs_label.cuda()
            # clear gradients 
            self.optimizer_.zero_grad()            
            # calc forward
            preds, _ = self.netloc_(imgs_tensor)
            # check 
            if preds.dim() == 1: preds = preds.unsqueeze(dim=0)            
            # calc acc & loss
            loss = self.criterion_(preds, imgs_label)
            # backpropagation
            loss.backward()
            # update parameters
            self.optimizer_.step()
            # accumulate loss & acc
            self.cls_meter_.add(preds.data, imgs_label.data)
            self.los_meter_.add(loss.data.item())
        # return 
        aAP   = self.cls_meter_.calc_avg_precision()
        mAP   = self.cls_meter_.calc_avg_precision().mean()
        loss  = self.los_meter_.value()[0]
        endtime = datetime.datetime.now()
        self.lr_ = self.optimizer_.param_groups[0]['lr']
        print('log: epoch-%d, train_map is %f, train_loss is %f, lr is %f, time is %d' % (epoch_idx, mAP, loss, self.lr_, (endtime - starttime).seconds))
        # return 
        return aAP, mAP, loss, self.lr_

    def train_multi_class(self, train_loader, epoch_idx):
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.train()
        self.los_meter_.reset()
        self.top1_meter_.reset()
        self.top5_meter_.reset()
        # train
        train_loader = tqdm(train_loader, desc='train', ascii=True)
        for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(train_loader):
            # set cuda
            imgs_tensor = imgs_tensor.cuda(device=self.cfg_.train.device_ids[0]) # [256, 3, 224, 224]
            imgs_label = imgs_label.cuda(device=self.cfg_.train.device_ids[0])     
            # clear gradients(zero the parameter gradients)
            self.optimizer_.zero_grad()
            # calc forward
            preds, _ = self.netloc_(imgs_tensor)
            # calc acc & loss
            loss = self.criterion_(preds, imgs_label)
            # backpropagation
            loss.backward()
            # update parameters
            self.optimizer_.step()
            # accumulate loss & acc
            acc1, acc5 = accuracy(preds, imgs_label, topk=(1, 5))
            self.los_meter_.update(loss.data.item())
            self.top1_meter_.update(acc1[0])
            self.top5_meter_.update(acc5[0])
        # eval 
        top1   = self.top1_meter_.mean
        top5   = self.top5_meter_.mean
        loss   = self.los_meter_.mean
        endtime= datetime.datetime.now()
        self.lr_ = self.optimizer_.param_groups[0]['lr']
        print('log: epoch-%d, train_top1 is %f, train_top5 is %f, train_loss is %f, lr is %f, time is %d' % (epoch_idx, top1, top5, loss, self.lr_, (endtime - starttime).seconds))
        # return 
        return top1, top5, loss, self.lr_

    # Func:
    #   Validate multi label model.
    def val_multi_label(self, val_loader, epoch_idx):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.eval()
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # eval
        val_loader = tqdm(val_loader, desc='valid', ascii=True)
        with torch.no_grad():
            for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(val_loader):
                # set cuda
                imgs_tensor = imgs_tensor.cuda()
                imgs_label = imgs_label.cuda()
                # calc forward
                preds, _ = self.netloc_(imgs_tensor)
                # check 
                if preds.dim() == 1: preds = preds.unsqueeze(dim=0)
                # calc acc & loss
                loss = self.criterion_(preds, imgs_label)
                # accumulate loss & acc
                self.cls_meter_.add(preds.data, imgs_label.data)
                self.los_meter_.add(loss.data.item())
        # eval 
        aAP   = self.cls_meter_.calc_avg_precision()
        mAP   = self.cls_meter_.calc_avg_precision().mean()
        loss  = self.los_meter_.value()[0]
        endtime = datetime.datetime.now()
        print('log: epoch-%d, val_map   is %f, val_loss   is %f, time is %d' % (epoch_idx, mAP, loss, (endtime - starttime).seconds))
        # update lr 
        if self.cfg_.train.scheduler.name == 'step': self.scheduler_.step()
        elif self.cfg_.train.scheduler.name == 'max':self.scheduler_.step(mAP)
        elif self.cfg_.train.scheduler.name == 'min':self.scheduler_.step(loss)      
        # return
        return aAP, mAP, loss

    def val_multi_class(self, val_loader, epoch_idx):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.eval()
        self.los_meter_.reset()
        self.top1_meter_.reset()
        self.top5_meter_.reset()
        # eval
        with torch.no_grad():
            val_loader = tqdm(val_loader, desc='valid', ascii=True)
            for imgs_idx, (imgs_tensor, imgs_label, _, _) in enumerate(val_loader):
                # set cuda
                imgs_tensor = imgs_tensor.cuda(device=self.cfg_.train.device_ids[0])
                imgs_label = imgs_label.cuda(device=self.cfg_.train.device_ids[0])
                # calc forward
                preds, _ = self.netloc_(imgs_tensor)
                # calc acc & loss
                loss = self.criterion_(preds, imgs_label)
                # accumulate loss & acc
                acc1, acc5 = accuracy(preds, imgs_label, topk=(1, 5))
                self.los_meter_.update(loss.item())
                self.top1_meter_.update(acc1[0])
                self.top5_meter_.update(acc5[0])
        # eval 
        top1   = self.top1_meter_.mean
        top5   = self.top5_meter_.mean
        loss   = self.los_meter_.mean
        endtime= datetime.datetime.now()
        print('log: epoch-%d, val_top1   is %f, val_top5   is %f, val_loss   is %f, time is %d' % (epoch_idx, top1, top5, loss, (endtime - starttime).seconds))
        # update lr 
        if self.cfg_.train.scheduler.name == 'step': self.scheduler_.step()
        elif self.cfg_.train.scheduler.name == 'max':self.scheduler_.step(top1)
        elif self.cfg_.train.scheduler.name == 'min':self.scheduler_.step(mLoss)      
        # return
        return top1, top5, loss

    # Func:
    #   Save model.
    def save_checkpoint(self, file_root, epoch_idx, train_map, val_map):
        file_name = os.path.join(file_root, time.strftime('%Y%m%d-%H-%M', time.localtime()) + '-' + str(epoch_idx) + '.pth')
        torch.save(
            {
                'epoch_idx'      : epoch_idx,
                'state_dict'     : self.netloc_.state_dict(),
                'train_map'      : train_map,
                'val_map'        : val_map,
                'lr'             : self.lr_,
                'optimizer'      : self.optimizer_.state_dict(),
                'scheduler'      : self.scheduler_.state_dict()
            }, file_name)
