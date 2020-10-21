import os
import sys
import time
import datetime
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchnet as tnt
from torch.nn.parameter import Parameter
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from module.VGG16LocalizerYGAPCSAtt import VGG16Localizer
from module.VGG19BNLocalizerYGAPCSAtt import VGG19Localizer
from module.InceptionV3LocalizerYGAPCSAtt import InceptionV3Localizer
from module.GoogLeNetLocalizerGAPCSAtt import GoogLeNetLocalizer
from tkit.ClassificationEvaluator import ClassificationEvaluator
from tkit.utils import accuracy, AverageMeter

class TrainEngine(object):
    # Func:
    #   Constructor.
    def __init__(self, lr, mt, wd, ss):
        # init setting
        self.device_ = ("cuda" if torch.cuda.is_available() else "cpu")
        # init optimizer perameters
        self.lr_ = lr
        self.mt_ = mt
        self.wd_ = wd
        self.ss_ = ss
        # create tool
        self.cls_meter_ = ClassificationEvaluator()
        self.los_meter_ = tnt.meter.AverageValueMeter()
        self.top1_meter_ = AverageMeter()
        self.top5_meter_ = AverageMeter()
        self.losx_meter_ = AverageMeter()

    # Func:
    #   Create experimental environment 
    def create_env(self, net_name='VGG16', task_type='multi-label', cls_num=20, pretrained=True, optimizer_type='sgd', scheduler_type='step', gpus=[0,1,2]):
        # create network 
        if   net_name == 'VGG16':
            self.netloc_ = VGG16Localizer(cls_num=cls_num)
        elif net_name == 'VGG19':
            self.netloc_ = VGG19Localizer(cls_num=cls_num)            
        elif net_name == 'InceptionV3':
            self.netloc_ = InceptionV3Localizer(cls_num=cls_num)
        elif net_name == 'GoogLeNetBN':
            self.netloc_ = GoogLeNetLocalizer(cls_num=cls_num)            
        # load pretrained params
        self.netloc_.init(pretrained=pretrained)
        # set parallel & cuda
        self.netloc_ = torch.nn.DataParallel(module=self.netloc_, device_ids=gpus)
        self.netloc_.cuda(device=gpus[0])
        self.gpus_ = gpus
        print(self.netloc_)
        # create loss function
        if   task_type == 'multi-label':
            self.criterion_ = nn.MultiLabelSoftMarginLoss().cuda()
        elif task_type == 'multi-class':
            self.criterion_ = nn.CrossEntropyLoss().cuda()
        # create optimizer
        self.optimizer_type_ = optimizer_type
        if   optimizer_type == 'sgd':
            self.optimizer_ = torch.optim.SGD(self.netloc_.parameters(), lr=self.lr_, momentum=self.mt_, weight_decay=self.wd_)
        elif optimizer_type == 'adam':
            self.optimizer_ = torch.optim.Adam(self.netloc_.parameters(), lr=self.lr_, weight_decay=self.wd_)
        # create scheduler
        self.scheduler_type_ = scheduler_type
        if   scheduler_type == 'step':
            self.scheduler_ = torch.optim.lr_scheduler.StepLR(self.optimizer_, step_size=self.ss_, gamma=0.1)
        elif scheduler_type == 'max':
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='max', factor=0.1, patience=2)
        elif scheduler_type == 'min':
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='min', factor=0.1, patience=2)  

    # Func:
    #   Create test environment 
    def create_env_r(self, checkpoint_path, net_name='VGG16', task_type='multi-label', cls_num=20, pretrained=True, optimizer_type='sgd', scheduler_type='step', gpus=[0,1,2]):
        # create network 
        if   net_name == 'VGG16':
            self.netloc_ = VGG16Localizer(cls_num=cls_num)
        elif net_name == 'InceptionV3':
            self.netloc_ = InceptionV3Localizer(cls_num=cls_num)
        # load state_dict(with dataparallel)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        # : create new OrderedDict that does not contain `module.`
        state_dict_new = OrderedDict()
        # : remove `module.`
        for k, v in state_dict.items():
            state_dict_new[k[7:]] = v
        # : load params
        self.netloc_.load_state_dict(state_dict_new)

        # set parallel & cuda
        self.netloc_ = torch.nn.DataParallel(module=self.netloc_, device_ids=gpus)
        self.netloc_.cuda(device=gpus[0])
        self.gpus_ = gpus
        print(self.netloc_)
        # create loss function
        if   task_type == 'multi-label':
            self.criterion_ = nn.MultiLabelSoftMarginLoss().cuda()
        elif task_type == 'multi-class':
            self.criterion_ = nn.CrossEntropyLoss().cuda()
        # create optimizer
        self.optimizer_type_ = optimizer_type
        if   optimizer_type == 'sgd':
            self.optimizer_ = torch.optim.SGD(self.netloc_.parameters(), lr=self.lr_, momentum=self.mt_, weight_decay=self.wd_)
        elif optimizer_type == 'adam':
            self.optimizer_ = torch.optim.Adam(self.netloc_.parameters(), lr=self.lr_, weight_decay=self.wd_)
        # create scheduler
        self.scheduler_type_ = scheduler_type
        if   scheduler_type == 'step':
            self.scheduler_ = torch.optim.lr_scheduler.StepLR(self.optimizer_, step_size=self.ss_, gamma=0.1)
        elif scheduler_type == 'max':
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='max', factor=0.1, patience=2)
        elif scheduler_type == 'min':
            self.scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='min', factor=0.1, patience=2)  
  

    # Func:
    #   Train singe scale model.
    def train_single_scale(self, train_loader, epoch_idx):
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.train()
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # train
        train_loader = tqdm(train_loader, desc='train', ascii=True)
        for imgs_idx, (imgs_tensor, imgs_label, imgs_name, _) in enumerate(train_loader):
            # set cuda
            imgs_tensor = imgs_tensor.cuda()
            imgs_label = imgs_label.cuda()
            # calc forward
            preds, crms = self.netloc_(imgs_tensor)
            # check 
            if preds.dim() == 1: preds = preds.unsqueeze(dim=0)            
            # calc acc & loss
            loss = self.criterion_(preds, imgs_label)
            # clear gradients 
            self.optimizer_.zero_grad()
            # backpropagation
            loss.backward()
            # update parameters
            self.optimizer_.step()
            # accumulate loss & acc
            self.cls_meter_.add(preds.data, imgs_label.data)
            self.los_meter_.add(loss.data.item())

            import gc
            del preds
            del crms
            gc.collect()
        # return 
        aAP   = self.cls_meter_.calc_avg_precision()
        mAP   = self.cls_meter_.calc_avg_precision().mean()
        mLoss = self.los_meter_.value()[0]
        endtime = datetime.datetime.now()
        self.lr_ = self.optimizer_.param_groups[0]['lr']
        print('log: epoch-%d, train_map is %f, train_loss is %f, lr is %f, time is %d' % (epoch_idx, mAP, mLoss, self.lr_, (endtime - starttime).seconds))
        # return 
        return aAP, mAP, mLoss, self.lr_

    def train_single_scale_ilsvrc(self, train_loader, epoch_idx):
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.train()
        self.los_meter_.reset()
        self.top1_meter_.reset()
        self.top5_meter_.reset()
        self.losx_meter_.reset()
        # train
        train_loader = tqdm(train_loader, desc='train', ascii=True)
        for imgs_idx, (imgs_tensor, imgs_label) in enumerate(train_loader):
            # set cuda
            imgs_tensor = imgs_tensor.cuda(device=self.gpus_[0])
            imgs_label = imgs_label.cuda(device=self.gpus_[0])     
            # clear gradients(zero the parameter gradients)
            self.optimizer_.zero_grad()
            # calc forward
            preds, crms = self.netloc_(imgs_tensor)
            # calc acc & loss
            loss = self.criterion_(preds, imgs_label)
            # backpropagation
            loss.backward()
            # update parameters
            self.optimizer_.step()
            # accumulate loss & acc
            acc1, acc5 = accuracy(preds, imgs_label, topk=(1, 5))
            self.losx_meter_.update(loss.item(), imgs_tensor.size(0))
            self.top1_meter_.update(acc1[0], imgs_tensor.size(0))
            self.top5_meter_.update(acc5[0], imgs_tensor.size(0))
            self.los_meter_.add(loss.data.item())
        # eval 
        top1   = self.top1_meter_.avg
        top5   = self.top5_meter_.avg
        mLoss  = self.losx_meter_.avg
        mLoss2 = self.los_meter_.value()[0]
        endtime= datetime.datetime.now()
        self.lr_ = self.optimizer_.param_groups[0]['lr']
        print('log: epoch-%d, train_top1 is %f, train_top5 is %f, train_loss is %f(%f), lr is %f, time is %d' % (epoch_idx, top1, top5, mLoss, mLoss2, self.lr_, (endtime - starttime).seconds))
        # return 
        return top1, top5, mLoss, self.lr_

    # Func:
    #   Validate singe scale model.
    def val_single_scale(self, val_loader, epoch_idx):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.eval()
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # eval
        val_loader = tqdm(val_loader, desc='valid', ascii=True)
        with torch.no_grad():
            for imgs_idx, (imgs_tensor, imgs_label, imgs_name, _) in enumerate(val_loader):
                # set cuda
                imgs_tensor = imgs_tensor.cuda()
                imgs_label = imgs_label.cuda()
                # calc forward
                preds, crms = self.netloc_(imgs_tensor)
                # check 
                if preds.dim() == 1: preds = preds.unsqueeze(dim=0)
                # calc acc & loss
                loss = self.criterion_(preds, imgs_label)
                # accumulate loss & acc
                self.cls_meter_.add(preds.data, imgs_label.data)
                self.los_meter_.add(loss.data.item())

                import gc
                del preds
                del crms
                gc.collect()
        # eval 
        aAP   = self.cls_meter_.calc_avg_precision()
        mAP   = self.cls_meter_.calc_avg_precision().mean()
        mLoss = self.los_meter_.value()[0]
        endtime = datetime.datetime.now()
        print('log: epoch-%d, val_map   is %f, val_loss   is %f, time is %d' % (epoch_idx, mAP, mLoss, (endtime - starttime).seconds))
        # update lr 
        if self.scheduler_type_ == 'step': self.scheduler_.step()
        elif self.scheduler_type_ == 'max':self.scheduler_.step(mAP)
        elif self.scheduler_type_ == 'min':self.scheduler_.step(mLoss)      
        # return
        return aAP, mAP, mLoss

    def val_single_scale_ilsvrc(self, val_loader, epoch_idx):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # switch to train mode
        self.netloc_.eval()
        self.top1_meter_.reset()
        self.top5_meter_.reset()
        self.losx_meter_.reset()
        # eval
        with torch.no_grad():
            val_loader = tqdm(val_loader, desc='valid', ascii=True)
            for imgs_idx, (imgs_tensor, imgs_label) in enumerate(val_loader):
                # set cuda
                imgs_tensor = imgs_tensor.cuda(device=self.gpus_[0])
                imgs_label = imgs_label.cuda(device=self.gpus_[0])
                # clear gradients(zero the parameter gradients)
                self.optimizer_.zero_grad()
                # calc forward
                preds, crms = self.netloc_(imgs_tensor)
                # calc acc & loss
                loss = self.criterion_(preds, imgs_label)
                # accumulate loss & acc
                acc1, acc5 = accuracy(preds, imgs_label, topk=(1, 5))
                self.losx_meter_.update(loss.item(), imgs_tensor.size(0))
                self.top1_meter_.update(acc1[0], imgs_tensor.size(0))
                self.top5_meter_.update(acc5[0], imgs_tensor.size(0))
        # eval 
        top1   = self.top1_meter_.avg
        top5   = self.top5_meter_.avg
        mLoss  = self.losx_meter_.avg
        endtime= datetime.datetime.now()
        print('log: epoch-%d, val_top1   is %f, val_top5   is %f, val_loss   is %f, time is %d' % (epoch_idx, top1, top5, mLoss, (endtime - starttime).seconds))
        # update lr 
        if self.scheduler_type_ == 'step': self.scheduler_.step()
        elif self.scheduler_type_ == 'max':self.scheduler_.step(top1)
        elif self.scheduler_type_ == 'min':self.scheduler_.step(mLoss)      
        # return
        return top1, top5, mLoss

    # Func:
    #   Save model.
    def save_checkpoint(self, file_root, epoch_idx, best_train_map, best_val_map):
        file_name = os.path.join(file_root, time.strftime('%Y%m%d-%H-%M', time.localtime()) + '-' + str(epoch_idx) + '.pth')
        torch.save(
            {
                'epoch_idx'      : epoch_idx,
                'state_dict'     : self.netloc_.state_dict(),
                'best_train_map' : best_train_map,
                'best_val_map'   : best_val_map,
                'lr'             : self.lr_,
                'optimizer'      : self.optimizer_.state_dict(),
                'scheduler'      : self.scheduler_.state_dict()
            }, file_name)
