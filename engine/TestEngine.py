import os
import sys
import time
import datetime
import pickle
import copy
import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as transforms
import torchnet as tnt
from torch.nn.parameter import Parameter
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from module.VGG16BNLocalizerYGAPCSAtt import VGG16Localizer
# from module.InceptionV3LocalizerYGAPCBAM import InceptionV3Localizer
from module.InceptionV3LocalizerYGAPCSAtt import InceptionV3Localizer
# from module.VGG16BNLocalizerYGAPCBAM_vis import VGG16Localizer
# from module.VGG19BNLocalizerYGAPCSAtt import VGG19Localizer
# from module.InceptionV3LocalizerY import InceptionV3Localizer
from module.GoogLeNetLocalizerGAPCSAtt import GoogLeNetLocalizer
# from module.GoogLeNetBNLocalizerGAPCBAM import GoogLeNetBNLocalizer

from tkit.ClassificationEvaluator import ClassificationEvaluator
from tkit.DetectionEvaluator import DetectionEvaluator
from tkit.HeatMapHelper import HeatMapHelper
from tkit.utils import accuracy, AverageMeter

class TestEngine(object):
    # Func:
    #   Constructor.
    def __init__(self, ds_name, img_names, img_labels, obj_bboxes, obj_categories):
        # init setting
        self.device_ = ("cuda" if torch.cuda.is_available() else "cpu")
        self.ds_name_    = ds_name
        self.img_names_  = img_names
        self.img_labels_ = img_labels
        self.obj_bboxes_ = obj_bboxes
        self.obj_bboxes_arr_ = np.array(obj_bboxes)
        self.obj_categories_ = obj_categories
        # create tool
        self.cls_meter_  = ClassificationEvaluator()
        self.det_meter_  = DetectionEvaluator(img_names=img_names, img_labels=img_labels, obj_bboxes=obj_bboxes, obj_categories=obj_categories)
        self.los_meter_  = tnt.meter.AverageValueMeter()
        self.hm_helper_  = HeatMapHelper(obj_categories=obj_categories)
        self.top1_meter_ = AverageMeter()
        self.top5_meter_ = AverageMeter()
        self.losx_meter_ = AverageMeter()

    # Func:
    #   Create test environment 
    def create_env(self, checkpoint_path, net_name='VGG16', task_type='multi-label', cls_num=1000, parallel=True, gpus=[0]):
        # create network 
        if   net_name == 'VGG16':
            self.netloc_ = VGG16Localizer(cls_num=cls_num)
        elif net_name == 'VGG19':
            self.netloc_ = VGG19Localizer(cls_num=cls_num)                  
        elif net_name == 'InceptionV3':
            self.netloc_ = InceptionV3Localizer(cls_num=cls_num)
        elif net_name == 'GoogLeNetBN':
            self.netloc_ = GoogLeNetLocalizer(cls_num=cls_num)                  
        # load state_dict(with dataparallel)
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        if parallel:
            # : create new OrderedDict that does not contain `module.`
            state_dict_new = OrderedDict()
            # : remove `module.`
            for k, v in state_dict.items():
                state_dict_new[k[7:]] = v
            # : load params
            self.netloc_.load_state_dict(state_dict_new)
        else:
            # : load params
            self.netloc_.load_state_dict(state_dict)  
        # set cuda
        self.netloc_.cuda(device=gpus[0])
        # switch to eval mode
        self.netloc_.eval()
        self.gpus_ = gpus
        print(self.netloc_)
        # create loss function
        if   task_type == 'multi-label':
            self.criterion_ = nn.MultiLabelSoftMarginLoss().cuda()
        elif task_type == 'multi-class':
            self.criterion_ = nn.CrossEntropyLoss().cuda()        

    # Func:
    #   Validate single scale model.
    def val_single_scale_for_classification(self, val_loader):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # reset meter
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            img_label = img_label.cuda(device=self.gpus_[0])
            # calc forward
            preds, _ = self.netloc_(img_tensor)
            # check 
            if preds.dim() == 1: preds = preds.unsqueeze(dim=0)
            # calc acc & loss
            loss = self.criterion_(preds, img_label)            
            # accumulate loss & acc
            self.cls_meter_.add(preds.data, img_label.data)
            self.los_meter_.add(loss.data.item())
        # return 
        aAP   = self.cls_meter_.calc_avg_precision()
        mAP   = self.cls_meter_.calc_avg_precision().mean()
        mAP2  = self.cls_meter_.calc_avg_precision2().mean()
        mLoss = self.los_meter_.value()[0]
        endtime = datetime.datetime.now()
        print('log: val_map   is %f(%f), val_loss   is %f, time is %d' % (mAP, mAP2, mLoss, (endtime - starttime).seconds))
        print('log: val_ap')
        print(aAP)
        return aAP, mAP, mLoss


    def val_single_scale_for_classification_ilsvrc(self, val_loader, res_file=None):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # check
        if res_file is not None:
            res_list = []
        # switch to train mode
        self.netloc_.eval()
        self.top1_meter_.reset()
        self.top5_meter_.reset()
        self.losx_meter_.reset()
        # eval
        with torch.no_grad():
            val_loader = tqdm(val_loader, desc='valid', ascii=True)
            for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
                # set cuda
                img_tensor = img_tensor.cuda(device=self.gpus_[0])
                img_label = img_label.cuda(device=self.gpus_[0])
                # calc forward
                preds, _ = self.netloc_(img_tensor)
                # check 
                if preds.dim() == 1: preds = preds.unsqueeze(dim=0)                
                # calc acc & loss
                loss = self.criterion_(preds, img_label)
                # accumulate loss & acc
                acc1, acc5 = accuracy(preds, img_label, topk=(1, 5))
                self.losx_meter_.update(loss.item(), img_tensor.size(0))
                self.top1_meter_.update(acc1[0], img_tensor.size(0))
                self.top5_meter_.update(acc5[0], img_tensor.size(0))
                # check
                if res_file is not None:
                    res_list.append(preds.squeeze().cpu().data.numpy())
        # eval 
        top1   = self.top1_meter_.avg
        top5   = self.top5_meter_.avg
        mLoss  = self.losx_meter_.avg
        endtime= datetime.datetime.now()
        print('log: val_top1   is %f, val_top5   is %f, val_loss   is %f, time is %d' % (top1, top5, mLoss, (endtime - starttime).seconds))
        # check
        if res_file is not None:
            res_list = np.array(res_list)
            pickle.dump(res_list, open(res_file, 'wb+'))
        # return
        return top1, top5, mLoss

    # Func:
    #   Validate single scale model.
    def val_single_scale_for_classification_fpn(self, val_loader):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # reset meter
        self.cls_meter_1_ = ClassificationEvaluator()
        self.cls_meter_2_ = ClassificationEvaluator()
        self.cls_meter_3_ = ClassificationEvaluator()
        self.cls_meter_4_ = ClassificationEvaluator()
        self.cls_meter_5_ = ClassificationEvaluator()
        self.los_meter_1_ = tnt.meter.AverageValueMeter()
        self.los_meter_2_ = tnt.meter.AverageValueMeter()
        self.los_meter_3_ = tnt.meter.AverageValueMeter()
        self.los_meter_4_ = tnt.meter.AverageValueMeter()
        self.los_meter_5_ = tnt.meter.AverageValueMeter()
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            img_label = img_label.cuda(device=self.gpus_[0])
            # calc forward
            preds1, _, preds2, _, preds3, _, preds4, _, preds5, _ = self.netloc_(img_tensor)
            # check 
            if preds1.dim() == 1: preds1 = preds1.unsqueeze(dim=0)            
            if preds2.dim() == 1: preds2 = preds2.unsqueeze(dim=0)            
            if preds3.dim() == 1: preds3 = preds3.unsqueeze(dim=0)            
            if preds4.dim() == 1: preds4 = preds4.unsqueeze(dim=0)            
            if preds5.dim() == 1: preds5 = preds5.unsqueeze(dim=0)            
            # calc acc & loss
            loss_1 = self.criterion_(preds1, img_label)
            loss_2 = self.criterion_(preds2, img_label)
            loss_3 = self.criterion_(preds3, img_label)
            loss_4 = self.criterion_(preds4, img_label)
            loss_5 = self.criterion_(preds5, img_label)      
            # accumulate loss & acc
            self.cls_meter_1_.add(preds1.data, img_label.data)
            self.cls_meter_2_.add(preds2.data, img_label.data)
            self.cls_meter_3_.add(preds3.data, img_label.data)
            self.cls_meter_4_.add(preds4.data, img_label.data)
            self.cls_meter_5_.add(preds5.data, img_label.data)
            self.los_meter_1_.add(loss_1.data.item())
            self.los_meter_2_.add(loss_2.data.item())
            self.los_meter_3_.add(loss_3.data.item())
            self.los_meter_4_.add(loss_4.data.item())
            self.los_meter_5_.add(loss_5.data.item())

        # return 
        aAP1   = self.cls_meter_1_.calc_avg_precision()
        mAP1   = self.cls_meter_1_.calc_avg_precision().mean()
        mLoss1 = self.los_meter_1_.value()[0]
        aAP2   = self.cls_meter_2_.calc_avg_precision()
        mAP2   = self.cls_meter_2_.calc_avg_precision().mean()        
        mLoss2 = self.los_meter_2_.value()[0]
        aAP3   = self.cls_meter_3_.calc_avg_precision()
        mAP3   = self.cls_meter_3_.calc_avg_precision().mean()            
        mLoss3 = self.los_meter_3_.value()[0]
        aAP4   = self.cls_meter_4_.calc_avg_precision()
        mAP4   = self.cls_meter_4_.calc_avg_precision().mean()            
        mLoss4 = self.los_meter_4_.value()[0]
        aAP5   = self.cls_meter_5_.calc_avg_precision()
        mAP5   = self.cls_meter_5_.calc_avg_precision().mean()            
        mLoss5 = self.los_meter_5_.value()[0]     
        endtime = datetime.datetime.now()
        print('log: val_map   is %f - %f - %f - %f - %f, val_loss is %f - %f - %f - %f - %f, time is %d' % (mAP1, mAP2, mAP3, mAP4, mAP5, mLoss1, mLoss2, mLoss3, mLoss4, mLoss5, (endtime - starttime).seconds))


    def val_single_scale_for_classification_with_point_localization(self, val_loader, tol=18):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        all_pred_score = []
        all_pred_pointpre = []      # point single prediction
        # reset meter
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            img_label = img_label.cuda(device=self.gpus_[0])
            # calc forward
            preds, crms = self.netloc_(img_tensor)
            # detach
            crms = crms.detach().cpu().squeeze()            
            # check 
            if preds.dim() == 1: preds = preds.unsqueeze(dim=0)
            # calc acc & loss
            loss = self.criterion_(preds, img_label)            
            # accumulate loss & acc
            self.cls_meter_.add(preds.data, img_label.data)
            self.los_meter_.add(loss.data.item())
            # check
            all_pred_score += [preds.squeeze().detach().cpu().data.numpy(),]

            # point prediction: by crm
            for cls_idx in range(len(img_label.cpu().detach().data.numpy().ravel())):
                crm = crms[cls_idx]
                # : localize - single, max response point
                pred_points_single = self.hm_helper_.localize_point(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm,                # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=True,
                                                    thrd=tol)

                # point prediction && bbox prediction
                all_pred_pointpre += [(img_idx,) + p for p in pred_points_single]

        # evaluate
        # : point prediction - single
        all_pred_pointpre = np.array(all_pred_pointpre)
        all_pred_score = np.array(all_pred_score)
        self.det_meter_.prediction_with_point_loc(pred_points=all_pred_pointpre, pred_scores=all_pred_score, tol=tol)
        # : classification
        aAP   = self.cls_meter_.calc_avg_precision()
        mAP   = self.cls_meter_.calc_avg_precision().mean()
        mAP2  = self.cls_meter_.calc_avg_precision2().mean()
        mLoss = self.los_meter_.value()[0]
        endtime = datetime.datetime.now()
        # return 
        print('log: val_map   is %f(%f), val_loss   is %f, time is %d' % (mAP, mAP2, mLoss, (endtime - starttime).seconds))
        print('log: val_ap')
        print(aAP)
        return aAP, mAP, mLoss


    # Func:
    #   Validate single scale model.
    def val_single_scale_for_classification_res(self, val_loader, res_root):
        np.set_printoptions(suppress=True)
        if not os.path.exists(res_root): os.makedirs(res_root)
        starttime = datetime.datetime.now()
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)
        for img_idx, (img_tensor, img_name) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            # calc forward
            preds, _ = self.netloc_(img_tensor)
            # check 
            preds = preds.squeeze().cpu().detach().numpy()
            # save
            for cls_idx in range(len(preds)):
                cls_file = os.path.join(res_root, 'comp1_cls_test_' + self.obj_categories_[cls_idx] + '.txt')
                img_score= ("%.6f" % preds[cls_idx])
                with open(cls_file, 'a') as f:
                    f.write(img_name[0] + ' ' + str(img_score) + '\n')   


    def val_single_scale_for_point_prediction(self, val_loader, tols=(0,18)):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        all_pred_pointpre = []      # point single prediction
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            # forward
            _, crms = self.netloc_(img_tensor, img_size)
            # detach
            crms = crms.detach().cpu().squeeze()

            # point prediction: by crm
            for cls_idx in range(len(img_label.numpy().ravel())):
                crm = crms[cls_idx]
                # : localize - single, max response point
                pred_points_single = self.hm_helper_.localize_point(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm,                # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=0)
                # point prediction && bbox prediction
                all_pred_pointpre += [(img_idx,) + p for p in pred_points_single]


        # evaluate
        # : point prediction - single
        all_pred_pointpre = np.array(all_pred_pointpre)
        for tol in tols:
            self.det_meter_.point_loc_with_prediction(pred_points=all_pred_pointpre, tol=tol)


    def val_multi_scale_for_corloc(self, val_loader, img_root, img_scales, s_thrd1, s_thrd2, m_thrd1, m_thrd2, d_thrd=0.5):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        all_pred_bboxloc_singl = []
        all_pred_bboxloc_mtop1 = []
        all_pred_bboxloc_multi = []
        all_pred_bboxloc_combi = []
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, img_name, img_size) in enumerate(val_loader):
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name[0] + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name[0])
            img_pil  = Image.open(img_path).convert('RGB')
            # set label
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()
            # create crms
            crms_fuo = None
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                if img_size[0] > 600 or img_size[1] > 600:
                    print('log: cpu process... %s' % img_path)
                    print(img_size)
                    # predict
                    _, crms = self.netloc_(img_tensor)
                    # upsample
                    crms = crms.detach().cpu()
                    crms = F.interpolate(crms, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                    # fuse
                    if crms_fuo is None: crms_fuo = crms.squeeze()
                    else: crms_fuo = crms_fuo + crms.squeeze()
                else:
                    # predict
                    _, crms = self.netloc_(img_tensor, img_size)
                    # detach
                    crms = crms.detach().cpu().squeeze()
                    # fuse
                    if crms_fuo is None: crms_fuo = crms
                    else: crms_fuo = crms_fuo + crms  

            # point / bbox prediction(classification and localization) && detection && localization: by crm
            crm_fus = torch.zeros(img_size)
            for cls_idx in range(len(img_label)):
                crm_fuo = crms_fuo[cls_idx]
                # : localize - single point & box
                crm_fus.copy_(crm_fuo)
                crm_fus[crm_fus<crm_fus.mean()*s_thrd1] = 0
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=s_thrd2)

                # : localize - multi points & boxes
                crm_fus.copy_(crm_fuo)
                crm_fus[crm_fus<crm_fus.mean()*m_thrd1] = 0                                           
                pred_bboxes_multi = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,   
                                                    heatmap=crm_fus, 
                                                    cls_idx=cls_idx, 
                                                    multi_objects=True,
                                                    upsample=False,
                                                    thrd=m_thrd2)

                # --------------- Top-1 ------------------
                # top1-area
                pred_bboxes_mtop1 = self.hm_helper_.sort_by_area(np.array(pred_bboxes_multi))
                pred_bboxes_mtop1 = tuple(pred_bboxes_mtop1[-1,:])

                # : accumulate
                if cls_idx in img_label_sq:
                    all_pred_bboxloc_singl += [(img_idx,) + p for p in pred_bboxes_singl]
                    all_pred_bboxloc_multi += [(img_idx,) + p for p in pred_bboxes_multi]
                    all_pred_bboxloc_mtop1 += [(img_idx,) + pred_bboxes_mtop1]

                    all_pred_bboxloc_combi += [(img_idx,) + p for p in pred_bboxes_singl]
                    all_pred_bboxloc_combi += [(img_idx,) + p for p in pred_bboxes_multi]

            import gc
            del crms, crms_fuo, crm_fus
            gc.collect()

        # evaluate
        print('all_grth_bboxes        : ', len(self.obj_bboxes_))
        print('all_pred_bboxloc_singl : ', len(all_pred_bboxloc_singl))
        print('all_pred_bboxloc_multi : ', len(all_pred_bboxloc_multi))

        # bbox localization - single
        all_pred_bboxloc_singl = np.array(all_pred_bboxloc_singl)
        self.det_meter_.bbox_loc_without_prediction(pred_bboxes=all_pred_bboxloc_singl)
        # bbox localization - mtop1
        all_pred_bboxloc_mtop1 = np.array(all_pred_bboxloc_mtop1)
        self.det_meter_.bbox_loc_without_prediction(pred_bboxes=all_pred_bboxloc_mtop1)             
        # bbox localization - multiple
        all_pred_bboxloc_multi = np.array(all_pred_bboxloc_multi)
        self.det_meter_.bbox_loc_without_prediction(pred_bboxes=all_pred_bboxloc_multi)
        # bbox localization - combine
        all_pred_bboxloc_combi = np.array(all_pred_bboxloc_combi)
        self.det_meter_.bbox_loc_without_prediction(pred_bboxes=all_pred_bboxloc_combi)        
        # bbox corloc
        self.det_meter_.cor_loc(pred_bboxes=all_pred_bboxloc_singl)
        self.det_meter_.cor_loc(pred_bboxes=all_pred_bboxloc_mtop1)
        self.det_meter_.cor_loc(pred_bboxes=all_pred_bboxloc_multi)
        self.det_meter_.cor_loc(pred_bboxes=all_pred_bboxloc_combi)


    def val_multi_scale_for_corloc_ilsvrc(self, val_loader, img_root, img_scales, seg_thrd=0.5, res_file=None):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # check
        grth_bboxes = np.array(self.obj_bboxes_)
        all_pred_bboxloc_singl = []
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, img_path, img_size) in enumerate(val_loader):
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')
            # set label
            img_label = img_label.numpy().ravel()
            # create crms
            crms_fuo = None
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                if img_size[0] > 600 or img_size[1] > 600:
                    print('log: cpu process... %s' % img_path)
                    print(img_size)
                    # predict
                    _, crms = self.netloc_(img_tensor)
                    # upsample
                    crms = crms.detach().cpu()
                    crms = F.interpolate(crms, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                    # fuse
                    if crms_fuo is None: crms_fuo = crms.squeeze()
                    else: crms_fuo = crms_fuo + crms.squeeze()
                else:
                    # predict
                    _, crms = self.netloc_(img_tensor, img_size)
                    # detach
                    crms = crms.detach().cpu().squeeze()
                    # fuse
                    if crms_fuo is None: crms_fuo = crms
                    else: crms_fuo = crms_fuo + crms  

            # point / bbox prediction(classification and localization) && detection && localization: by crm
            cls_idx = img_label[0]
            crm_fuo = crms_fuo[cls_idx]
            # : localize - single point & box
            # crm_fuo[crm_fuo<crm_fuo.mean()] = 0
            pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                heatmap=crm_fuo,            # intput format: [h,w], e.g. [14,14]
                                                cls_idx=cls_idx, 
                                                multi_objects=False,
                                                upsample=False,
                                                thrd=seg_thrd)

            all_pred_bboxloc_singl += [(img_idx,) + p for p in pred_bboxes_singl]

        # evaluate
        print('all_grth_bboxes        : ', len(self.obj_bboxes_))
        print('all_pred_bboxloc_singl : ', len(all_pred_bboxloc_singl))

        # bbox localization - single
        all_pred_bboxloc_singl = np.array(all_pred_bboxloc_singl)
        # bbox corloc
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=all_pred_bboxloc_singl)
        # check
        if res_file is not None:
            pickle.dump(all_pred_bboxloc_singl, open(res_file, 'wb+'))        


    def val_multi_scale_for_corloc_basedon_segmentation(self, val_loader, img_root, seg_root, erosion_thrd=1):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        all_pred_bboxloc_multi = []
        # set erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, img_name, img_size) in enumerate(val_loader):
            # set label
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()     
            img_name = img_name[0]
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name)
            img_pil  = Image.open(img_path).convert('RGB')
            # open seg
            seg_path = os.path.join(seg_root, img_name + '.seg')
            seg_mat  = pickle.load(open(seg_path, 'rb+'))

            # loc
            for cls_idx in img_label_sq:
                # binary
                seg_mat_c = (seg_mat == cls_idx)
                # erosion
                seg_mat_c = seg_mat_c.astype(np.uint8)
                seg_mat_c = cv2.erode(src=seg_mat_c, kernel=kernel, iterations=erosion_thrd)
                # loc
                pred_bboxes_multi = self.hm_helper_.localize_bbox_basedon_seg(
                                                img_size=img_size, 
                                                seg_map=seg_mat_c, 
                                                cls_idx=cls_idx, 
                                                multi_objects=True)
                # check
                if len(pred_bboxes_multi) == 0: continue
                # --------------- Top-1 ------------------
                # top1-area
                pred_bboxes_multi = self.hm_helper_.sort_by_area(np.array(pred_bboxes_multi))
                pred_bboxes_multi = tuple(pred_bboxes_multi[-1,:])
                # save
                all_pred_bboxloc_multi += [(img_idx,) + pred_bboxes_multi]


                # --------------- All ------------------ 效果不好
                # all_pred_bboxloc_multi += [(img_idx,) + p for p in pred_bboxes_multi]


                # # vis binary-seg
                # seg_img = Image.fromarray(np.uint8(seg_mat_c*255))
                # savepath = os.path.join(seg_root, img_name + '_' + str(cls_idx) + '_seg.png')
                # seg_img.save(savepath)

                # # vis bbox
                # # : check
                # cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
                # img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                # # : check
                # if len(pred_bboxes_multi) == 0 or len(img_grth_bboxes) == 0: continue
                # # : check
                # phits, rhits = self.det_meter_.calc_bbox_hits(np.array([pred_bboxes_multi,])[:,1:5], img_grth_bboxes, 0.5)
                # iou = max(self.det_meter_.calc_ious(np.array([pred_bboxes_multi,])[:,1:5], img_grth_bboxes))
                # if phits == 0:
                #     savepath = os.path.join('loc-vgg16ybn-voc2007-c21-seg-gap-f14-sgd-max-b196-95.70-87.56-loc-err', img_name + '_' + str(cls_idx) + '_' + str(iou) + '.png')
                #     self.hm_helper_.visualize_bbox_with_grth(img_pil, [pred_bboxes_multi,], img_grth_bboxes, savepath=savepath, color='blue')
                                

        # bbox localization - multiple
        all_pred_bboxloc_multi = np.array(all_pred_bboxloc_multi)
        self.det_meter_.bbox_loc_without_prediction(pred_bboxes=all_pred_bboxloc_multi)
        # bbox corloc
        self.det_meter_.cor_loc2(pred_bboxes=all_pred_bboxloc_multi)


    def val_multi_scale_for_corloc_basedon_segmentation_ilsvrc(self, val_loader, img_root, seg_root, erosion_thrd=1):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        all_pred_bboxloc_multi = []
        # set erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, img_path, img_size) in enumerate(val_loader):
            img_path = img_path[0]
            # open image
            img_pil  = Image.open(img_path).convert('RGB')
            img_size = img_pil.size[::-1]  # [h, w]
            # set label
            img_name  = os.path.basename(img_path).replace('.jpg', '').replace('.JPG', '').replace('.JPEG', '')
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()
            # open seg
            seg_path = os.path.join(seg_root, img_name + '.seg')
            seg_mat  = pickle.load(open(seg_path, 'rb+'))

            # loc
            cls_idx = img_label[0]
            # binary
            seg_mat_c = (seg_mat == cls_idx)
            # erosion
            seg_mat_c = seg_mat_c.astype(np.uint8)
            seg_mat_c = cv2.erode(src=seg_mat_c, kernel=kernel, iterations=erosion_thrd)
            # loc
            pred_bboxes_multi = self.hm_helper_.localize_bbox_basedon_seg(
                                            img_size=img_size, 
                                            seg_map=seg_mat_c, 
                                            cls_idx=cls_idx, 
                                            multi_objects=True)
            # check
            if len(pred_bboxes_multi) == 0: continue
            # --------------- Top-1 ------------------
            # top1-area
            pred_bboxes_multi = self.hm_helper_.sort_by_area(np.array(pred_bboxes_multi))
            pred_bboxes_multi = tuple(pred_bboxes_multi[-1,:])
            # save
            all_pred_bboxloc_multi += [(img_idx,) + pred_bboxes_multi]


            # # visualize binary-seg
            # seg_img = Image.fromarray(np.uint8(seg_mat_c*255))
            # savepath = os.path.join('loc-vgg16ybn-cub200-seg-gap-f14-sgd-max-b128-100.0-100.0-81.02-95.09-vis', img_name + '.png')
            # seg_img.save(savepath)

        # bbox corloc
        all_pred_bboxloc_multi = np.array(all_pred_bboxloc_multi)
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=all_pred_bboxloc_multi)


    def val_multi_scale_for_clsloc(self, val_loader, img_scales, preds_file, res_file=None, seg_thrd=0.5):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        import operator
        from functools import reduce            
    
        # load preds & preds_bbox
        preds_scores = pickle.load(open(preds_file, 'rb+'))
        # calc cls top-k err
        preds_idx_top5 = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
  
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        # eval
        all_pred_bboxes_top5 = []
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, _, img_path, img_size) in enumerate(val_loader):
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')
            # create crms
            crms_fuo = None
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])
                if img_size[0] > 900 or img_size[1] > 900:
                    # predict
                    _, crms = self.netloc_(img_tensor)
                    # upsample
                    crms = crms.detach().cpu()
                    crms = F.interpolate(crms, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                    # fuse
                    if crms_fuo is None: crms_fuo = crms.squeeze()
                    else: crms_fuo = crms_fuo + crms.squeeze()
                else:
                    # predict
                    _, crms = self.netloc_(img_tensor, img_size)
                    # detach
                    crms = crms.detach().cpu().squeeze()
                    # fuse
                    if crms_fuo is None: crms_fuo = crms
                    else: crms_fuo = crms_fuo + crms    

            # point / bbox prediction(classification and localization) && detection && localization: by crm
            pred_bboxes_top5 = []
            for cls_idx in preds_idx_top5[img_idx]:
                crm_fuo = crms_fuo[cls_idx]
                # : localize - single point & box
                crm_fuo[crm_fuo<crm_fuo.mean()] = 0
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fuo,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=seg_thrd)

                pred_bboxes_top5 += [(img_idx,) + p for p in pred_bboxes_singl]
            # save
            all_pred_bboxes_top5 += [reduce(operator.add, pred_bboxes_top5)]

        # evaluate
        print('all_grth_bboxes      : ', len(self.obj_bboxes_))
        print('all_pred_bboxes_top5 : ', len(all_pred_bboxes_top5))

        # bbox localization - multi-class
        all_pred_bboxes_top5 = np.array(all_pred_bboxes_top5)
        # save
        if res_file is not None:
            pickle.dump(all_pred_bboxes_top5, open(res_file, 'wb+'))       


    def val_multi_scale_for_clsloc_err_analyze(self, preds_file, preds_bbox_file):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # load preds & preds_bbox
        preds_scores = pickle.load(open(preds_file, 'rb+'))
        preds_bboxes = pickle.load(open(preds_bbox_file, 'rb+'))
        # check - bbox corloc
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=preds_bboxes[:,0:7])
        # check
        gt_labels = np.tile(np.array(self.img_labels_)[:, np.newaxis], 5)
        gt_bboxes = self.obj_bboxes_arr_[:,2:]

        # calc cls top-k err
        top5_predidx = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        clserr = (top5_predidx != gt_labels)
        top1_clserr = clserr[:, 0].sum() / float(clserr.shape[0])
        top5_clserr = np.min(clserr, axis=1).sum() / float(clserr.shape[0])
        print('log: val_top1   is %f, val_top5   is %f' % ((1.0-top1_clserr), (1.0-top5_clserr)))

        # calc loc top-k err
        iou_val = np.zeros((len(self.img_labels_), 5))
        for k in range(5):
            preds_boxes_k = preds_bboxes[:, 2 + 7 * k: 2 + 7 * k + 4]
            iou_val[:, k] = self.det_meter_.cal_ious(preds_boxes_k, gt_bboxes)
        locerr = iou_val < 0.5

        # calc cls-loc top-k err
        clsloc_err = np.logical_or(locerr, clserr)
        top1_clsloc_err = clsloc_err[:, 0].sum() / float(clsloc_err.shape[0])
        top5_clsloc_err = np.min(clsloc_err, axis=1).sum() / float(clsloc_err.shape[0])
        print('log: clsloc_err_top1   is %f, clsloc_err_top5   is %f' % (top1_clsloc_err, top5_clsloc_err))


    def val_single_scale_for_visualize(self, val_loader, img_root, crm_root, seg_thrd=0.3):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)   
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name[0] + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name[0])
            img_pil  = Image.open(img_path)
            img_scale= np.min((img_tensor.shape[2], img_tensor.shape[3]))
            # set label
            img_name  = img_name[0]
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()

            # forward
            _, crms = self.netloc_(img_tensor, img_size)
            # detach
            crms = crms.detach().cpu().squeeze()

            # bbox prediction: by crm
            crm_fus = torch.zeros(img_size)
            for cls_idx in img_label_sq:
                crm = crms[cls_idx]
                crm_max = crm.max().data.numpy()
                crm_min = crm.min().data.numpy() - 10    
                crm_min = 0           

                # : localize - multi
                crm_fus.copy_(crm)
                pred_points_multi, pred_bboxes_multi = self.hm_helper_.localize_point_bbox(
                                                    img_size=img_size,   
                                                    heatmap=crm_fus, 
                                                    cls_idx=cls_idx, 
                                                    multi_objects=True,
                                                    upsample=False,
                                                    thrd=seg_thrd)

                # vis orginal crm
                crm_fus.copy_(crm)
                savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(img_scale) + '_0.png')
                self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=-1)
                # vis refined crm
                crm_fus.copy_(crm)
                savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(img_scale) + '_1.png')
                self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=seg_thrd)                        
                # vis bbox
                cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
                img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(img_scale) + '_6.png')
                self.hm_helper_.visualize_bbox_with_grth(img_pil, pred_bboxes_multi, img_grth_bboxes, savepath=savepath, color='blue')
                               

                # # vis filled image
                # self.hm_helper_.visualize_bbox(img_pil, pred_bboxes_multi, savepath=str(cls_idx) + '_' + str(img_scale) + '_6.png', color='red')
                # self.hm_helper_.visualize_bbox_fill(img_pil, pred_bboxes_multi, savepath=str(cls_idx) + '_' + str(img_scale) + '_5.png', color='red')
         
            break


    def val_single_scale_for_visualize_fpn(self, val_loader, img_root, crm_root, seg_thrd=0.3):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)   
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name[0] + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name[0])
            img_pil  = Image.open(img_path)
            img_scale= np.min((img_tensor.shape[2], img_tensor.shape[3]))
            # set label
            img_name  = img_name[0]
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()

            # forward
            _, crms1, _, crms2, _, crms3, _, crms4, _, crms5 = self.netloc_(img_tensor, img_size)

            # detach
            crms1 = crms1.detach().cpu().squeeze()
            crms2 = crms2.detach().cpu().squeeze()
            crms3 = crms3.detach().cpu().squeeze()
            crms4 = crms4.detach().cpu().squeeze()
            crms5 = crms5.detach().cpu().squeeze()
            crmsf = crms1 + crms2 + crms3 + crms4 + crms5

            crms_list = []
            crms_list.append(crms1)
            crms_list.append(crms2)
            crms_list.append(crms3)
            crms_list.append(crms4)
            crms_list.append(crms5)
            crms_list.append(crmsf)
            print(len(crms_list))
            for crms_idx in range(len(crms_list)):
                crms = crms_list[crms_idx]
                # bbox prediction: by crm
                crm_fus = torch.zeros(img_size)
                for cls_idx in img_label_sq:
                    crm = crms[cls_idx]
                    # : localize - multi
                    pred_points_multi, pred_bboxes_multi = self.hm_helper_.localize_point_bbox(
                                                        img_size=img_size,   
                                                        heatmap=crm, 
                                                        cls_idx=cls_idx, 
                                                        multi_objects=True,
                                                        upsample=False,
                                                        thrd=seg_thrd)

                    # vis orginal crm
                    crm_fus.copy_(crm)
                    savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(crms_idx) + '_' + str(img_scale) + '_0.png')
                    self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=-1)
                    # vis refined crm
                    crm_fus.copy_(crm)
                    savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(crms_idx) + '_' + str(img_scale) + '_1.png')
                    self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=seg_thrd)                        
                    # vis bbox
                    cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
                    img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                    savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(crms_idx) + '_' + str(img_scale) + '_6.png')
                    self.hm_helper_.visualize_bbox_with_grth(img_pil, pred_bboxes_multi, img_grth_bboxes, savepath=savepath, color='blue')
                                
            break


    def val_single_scale_for_visualize_ilsvrc(self, val_loader, img_root, crm_root, seg_thrd=0.3):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)   
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for _, (img_tensor, img_label, img_path, img_size, img_idx) in enumerate(val_loader):
            # check
            if img_idx == 60: break
            # check
            img_name = os.path.basename(img_path[0])
            # create dir
            crm_root_sub = os.path.join(crm_root, img_name)
            if not os.path.exists(crm_root_sub): os.makedirs(crm_root_sub)


            img_idx = img_idx.item()
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')
            img_scale= np.min((img_tensor.shape[2], img_tensor.shape[3]))
            # set label
            img_label = img_label.numpy().ravel()
            # forward
            _, crms = self.netloc_(img_tensor)
            # detach
            crms = crms.detach().cpu().squeeze()

            # bbox prediction: by crm
            cls_idx = img_label[0]
            crm = crms[cls_idx]
            # : localize - single point & box
            crm_fus = torch.zeros(crm.shape)
            crm_fus.copy_(crm)
            pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
                                                cls_idx=cls_idx, 
                                                multi_objects=False,
                                                upsample=True,
                                                thrd=seg_thrd)

            crm_fus.copy_(crm)
            savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_0.png')
            self.hm_helper_.visualize_fmap(img_pil, crm_fus, upsample=True, savepath=savepath, thrd=-1)

            # vis orginal crm with image
            crm_fus.copy_(crm)
            savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_1.png')
            self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=True, savepath=savepath, thrd=-1)


            crm_fus.copy_(crm)
            savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_2.png')
            self.hm_helper_.visualize_fmap(img_pil, crm_fus, upsample=True, savepath=savepath, thrd=seg_thrd)

            # vis refined crm
            crm_fus.copy_(crm)
            savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_3.png')
            self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=True, savepath=savepath, thrd=seg_thrd)                        
            # # vis bbox
            # cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
            # img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
            # savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_6.png')
            # self.hm_helper_.visualize_bbox_with_grth(img_pil, pred_bboxes_singl, img_grth_bboxes, savepath=savepath, color='blue')
            # vis filled
            # savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_5.png')
            # self.hm_helper_.visualize_bbox_fill_2(
            #                                     img_pil=img_pil,
            #                                     img_size=img_size,   
            #                                     heatmap=crm_fus, 
            #                                     cls_idx=cls_idx, 
            #                                     savepath=savepath,
            #                                     upsample=True,
            #                                     thrd=seg_thrd)
         

            # break

    def val_multi_scale_for_visualize_ilsvrc(self, val_loader, img_root, img_scales, crm_root, seg_thrd=0.3):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])                
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (_, img_label, img_path, img_size, img_idx) in enumerate(val_loader):
            # check
            # if img_idx == 1000: break
            # check
            img_name = os.path.basename(img_path[0])
            # create dir
            crm_root_sub = os.path.join(crm_root, img_name)
            if not os.path.exists(crm_root_sub): os.makedirs(crm_root_sub)
            # else: continue

            img_idx = img_idx.item()
            # set label
            img_label = img_label.numpy().ravel()
            cls_idx   = img_label[0]

            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')

            # create crms
            crms_fuo = None
            crms_list= {}
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # predict
                _, crms = self.netloc_(img_tensor)
                # detach
                crms = crms.detach().cpu().squeeze()
                # bbox prediction: by crm
                crm = crms[cls_idx]
                # upsample
                crm = F.interpolate(crm.unsqueeze(0).unsqueeze(0), size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                crm = crm.squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crm
                else: crms_fuo = crms_fuo + crm
                # list
                crms_list[img_scale] = copy.deepcopy(crm)
            # add fusion
            crms_list[999] = crms_fuo


            # visualize - single & fuse
            crm_cpy = torch.zeros(img_size)
            for img_scale, crm in crms_list.items():
                # if img_scale != 999: continue
                # : localize - single point & box
                crm_fus = torch.zeros(crm.shape)
                crm_fus.copy_(crm)
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=seg_thrd)

                crm_fus.copy_(crm)
                savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_0.png')
                self.hm_helper_.visualize_fmap(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=-1)

                # vis orginal crm with image
                crm_fus.copy_(crm)
                savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_1.png')
                self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=-1)


                # crm_fus.copy_(crm)
                # savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_2_0.2.png')
                # self.hm_helper_.visualize_fmap(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=0.2)
                # # vis refined crm
                # crm_fus.copy_(crm)
                # savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_3_0.2.png')
                # self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=0.2)    


                # crm_fus.copy_(crm)
                # savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_2_0.37.png')
                # self.hm_helper_.visualize_fmap(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=0.35)
                # vis refined crm
                crm_fus.copy_(crm)
                savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_3_0.37.png')
                self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=0.35)


                # vis results
                savepath = os.path.join(crm_root_sub, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_7.png')
                cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
                img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                self.hm_helper_.visualize_bbox_with_grth(img_pil, pred_bboxes_singl, img_grth_bboxes, savepath=savepath, color='blue')
                # break
            break


    def val_single_scale_for_visualize_ilsvrc_intermediate(self, val_loader, img_root, crm_root, writer, seg_thrd=0.3):
        from tensorboardX import SummaryWriter
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)   
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (img_tensor, img_label, img_path, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda(device=self.gpus_[0])
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')
            img_scale= np.min((img_tensor.shape[2], img_tensor.shape[3]))
            # set label
            img_label = img_label.numpy().ravel()
            # forward
            _, crms, conv4_conv, conv4_bn, conv4_pool, conv4_csat, conv7_conv, conv7_bn, conv7_pool, conv7_csat, conv10_conv, conv10_bn, conv10_pool, conv10_csat = self.netloc_(img_tensor, img_size)


            # conv4_conv = conv4_conv.unsqueeze(1)
            # conv4_bn   = conv4_bn.unsqueeze(1)
            # conv4_pool = conv4_pool.unsqueeze(1)
            # conv4_csat = conv4_csat.unsqueeze(1)
            # conv7_conv = conv7_conv.unsqueeze(1)
            # conv7_bn   = conv7_bn.unsqueeze(1)
            # conv7_pool = conv7_pool.unsqueeze(1)
            # conv7_csat = conv7_csat.unsqueeze(1)
            # conv10_conv = conv10_conv.unsqueeze(1)
            # conv10_bn   = conv10_bn.unsqueeze(1)
            # conv10_pool = conv10_pool.unsqueeze(1)
            # conv10_csat = conv10_csat.unsqueeze(1)

            # img_grid = torchvision.utils.make_grid(conv4_conv, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv4_conv', img_grid, global_step=None, walltime=None, dataformats='CHW')

            # img_grid = torchvision.utils.make_grid(conv4_bn, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv4_bn', img_grid, global_step=None, walltime=None, dataformats='CHW')

            # img_grid = torchvision.utils.make_grid(conv4_pool, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv4_pool', img_grid, global_step=None, walltime=None, dataformats='CHW')

            # img_grid = torchvision.utils.make_grid(conv4_csat, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv4_csat', img_grid, global_step=None, walltime=None, dataformats='CHW')

            # img_grid = torchvision.utils.make_grid(conv7_conv, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv7_conv', img_grid, global_step=None, walltime=None, dataformats='CHW')

            # img_grid = torchvision.utils.make_grid(conv7_bn, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv7_bn', img_grid, global_step=None, walltime=None, dataformats='CHW')            

            # img_grid = torchvision.utils.make_grid(conv7_pool, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv7_pool', img_grid, global_step=None, walltime=None, dataformats='CHW')   

            # img_grid = torchvision.utils.make_grid(conv7_csat, normalize=True, scale_each=True, nrow=10) 
            # writer.add_image('conv7_csat', img_grid, global_step=None, walltime=None, dataformats='CHW')   



            self.hm_helper_.visualize_fmap(img_pil, conv4_conv.mean(dim=0), upsample=True, savepath=crm_root + '/conv4_1_conv.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv4_bn.mean(dim=0),   upsample=True, savepath=crm_root + '/conv4_2_bn.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv4_pool.mean(dim=0), upsample=True, savepath=crm_root + '/conv4_3_pool.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv4_csat.mean(dim=0), upsample=True, savepath=crm_root + '/conv4_4_csat.png', thrd=-1)

            self.hm_helper_.visualize_fmap(img_pil, conv7_conv.mean(dim=0), upsample=True, savepath=crm_root + '/conv7_1_conv.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv7_bn.mean(dim=0),   upsample=True, savepath=crm_root + '/conv7_2_bn.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv7_pool.mean(dim=0), upsample=True, savepath=crm_root + '/conv7_3_pool.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv7_csat.mean(dim=0), upsample=True, savepath=crm_root + '/conv7_4_csat.png', thrd=-1)

            self.hm_helper_.visualize_fmap(img_pil, conv10_conv.mean(dim=0), upsample=True, savepath=crm_root + '/conv10_1_conv.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv10_bn.mean(dim=0),   upsample=True, savepath=crm_root + '/conv10_2_bn.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv10_pool.mean(dim=0), upsample=True, savepath=crm_root + '/conv10_3_pool.png', thrd=-1)
            self.hm_helper_.visualize_fmap(img_pil, conv10_csat.mean(dim=0), upsample=True, savepath=crm_root + '/conv10_4_csat.png', thrd=-1)


            '''
            # --- Visualization feature map for a kernel
            for i in range(conv4_conv.shape[0]):
                fmap = conv4_conv[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv4_1_conv' + str(i) + '.png', thrd=-1)

            for i in range(conv4_bn.shape[0]):
                fmap = conv4_bn[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv4_2_bn' + str(i) + '.png', thrd=-1)

            for i in range(conv4_pool.shape[0]):
                fmap = conv4_pool[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv4_3_pool' + str(i) + '.png', thrd=-1)

            for i in range(conv4_csat.shape[0]):
                fmap = conv4_csat[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv4_4_csat' + str(i) + '.png', thrd=-1)

            # --- 
            for i in range(conv7_conv.shape[0]):
                fmap = conv7_conv[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv7_1_conv' + str(i) + '.png', thrd=-1)

            for i in range(conv7_bn.shape[0]):
                fmap = conv7_bn[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv7_2_bn' + str(i) + '.png', thrd=-1)

            for i in range(conv7_pool.shape[0]):
                fmap = conv7_pool[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv7_2_pool' + str(i) + '.png', thrd=-1)

            for i in range(conv7_csat.shape[0]):
                fmap = conv7_csat[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv7_4_csat' + str(i) + '.png', thrd=-1)

            # --- 
            for i in range(conv10_conv.shape[0]):
                fmap = conv10_conv[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv10_1_conv' + str(i) + '.png', thrd=-1)

            for i in range(conv10_bn.shape[0]):
                fmap = conv10_bn[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv10_2_bn' + str(i) + '.png', thrd=-1)

            for i in range(conv10_pool.shape[0]):
                fmap = conv10_pool[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv10_2_pool' + str(i) + '.png', thrd=-1)

            for i in range(conv10_csat.shape[0]):
                fmap = conv10_csat[i]
                self.hm_helper_.visualize_fmap(img_pil, fmap, upsample=True, savepath=crm_root + '/conv10_4_csat' + str(i) + '.png', thrd=-1)
            '''






            # # detach
            # crms = crms.detach().cpu().squeeze()

            # # bbox prediction: by crm
            # cls_idx = img_label[0]
            # crm = crms[cls_idx]
            # # : localize - single point & box
            # crm_fus = torch.zeros(img_size)
            # crm_fus.copy_(crm)
            # pred_bboxes_singl = self.hm_helper_.localize_bbox(
            #                                     img_size=img_size,          # intput format: [h,w], e.g. [375,500]
            #                                     heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
            #                                     cls_idx=cls_idx, 
            #                                     multi_objects=False,
            #                                     upsample=False,
            #                                     thrd=seg_thrd)

            # # vis orginal crm
            # crm_fus.copy_(crm)
            # savepath = os.path.join(crm_root, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_0.png')
            # self.hm_helper_.visualize_crm(img_pil, crm_fus, upsample=False, savepath=savepath, thrd=-1)

            # # vis bbox
            # cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
            # img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
            # savepath = os.path.join(crm_root, str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_6.png')
            # self.hm_helper_.visualize_bbox_with_grth(img_pil, pred_bboxes_singl, img_grth_bboxes, savepath=savepath, color='blue')
                       
            break


    def val_multi_scale_for_visualize(self, val_loader, img_root, img_scales, crm_root, s_thrd1, s_thrd2, m_thrd1, m_thrd2, d_thrd=0.5):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])                
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (_, img_label, img_name, img_size) in enumerate(val_loader):
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name[0] + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name[0])
            img_pil  = Image.open(img_path)
            img_size = img_pil.size[::-1]  # [h, w]
            # set label
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()

            # create crms
            crms_fuo = None
            crms_list= {}
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # predict
                preds, crms = self.netloc_(img_tensor, img_size)
                # detach
                crms = crms.detach().cpu().squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms
                else: crms_fuo = crms_fuo + crms
                # list
                crms_list[img_scale] = copy.deepcopy(crms)
            # add fusion
            crms_list[999] = crms_fuo

            # visualize - single & fuse
            crm_cpy = torch.zeros(img_size)
            for img_scale, crms in crms_list.items():
                crms_max = crms.max().data.numpy()
                crms_min = crms.min().data.numpy() - 10
                # point localization && bbox localization
                crm_cpy = torch.zeros(img_size)
                for cls_idx in img_label_sq:
                    crm = crms[cls_idx]
                    # localize - multi
                    crm_cpy.copy_(crm)
                    # crm_cpy[crm_cpy<crm_cpy.mean()*m_thrd1] = crms_min       
                    pred_points_multi, pred_bboxes_multi = self.hm_helper_.localize_point_bbox(
                                                        img_size=img_size,   
                                                        heatmap=crm_cpy, 
                                                        cls_idx=cls_idx, 
                                                        multi_objects=True,
                                                        upsample=False,
                                                        thrd=m_thrd2)
                    # visualize
                    # : orginal crm
                    crm_cpy.copy_(crm)
                    self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, savepath=crm_root + '/' + str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_0.png', thrd=-1)
                    self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, savepath=crm_root + '/' + str(img_idx) + '_' + str(cls_idx) + '_' + str(img_scale) + '_0_thrd.png', thrd=m_thrd2)
                    # # : refined crm
                    # crm_cpy.copy_(crm)
                    # crm_cpy[crm_cpy<crm_cpy.mean()*1.0] = crms_min       
                    # self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, savepath='vis/' + str(cls_idx) + '_' + str(img_scale) + '_1.png', thrd=-1)
                    # self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, savepath='vis/' + str(cls_idx) + '_' + str(img_scale) + '_1_thrd.png', thrd=m_thrd2)
                    # # : refined crm
                    # crm_cpy.copy_(crm)
                    # crm_cpy[crm_cpy<crm_cpy.mean()*1.5] = crms_min                  
                    # self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, savepath='vis/' + str(cls_idx) + '_' + str(img_scale) + '_2.png', thrd=-1)
                    # self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, savepath='vis/' + str(cls_idx) + '_' + str(img_scale) + '_2_thrd.png', thrd=m_thrd2)
                    # # : bbox
                    # self.hm_helper_.visualize_bbox(img_pil, pred_bboxes_multi, savepath='vis/' + str(cls_idx) + '_' + str(img_scale) + '_5.png', color='red')

                    # # vis results
                    # cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
                    # img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                    # # self.hm_helper_.visualize_point_bbox(img_pil, pred_points_multi, pred_bboxes_multi, savepath=str(cls_idx) + '_' + str(img_scale) + '_5.png', color='red')
                    # # self.hm_helper_.visualize_bbox2(img_pil, pred_bboxes_multi, img_grth_bboxes, savepath=str(cls_idx) + '_' + str(img_scale) + '_7.png', color='blue')

            # break


    def val_multi_scale_for_visualize_norm(self, val_loader, img_root, img_scales, crm_root, seg_thrd=0.5):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])                
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (_, img_label, img_name, img_size) in enumerate(val_loader):
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name[0] + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name[0])
            img_pil  = Image.open(img_path)
            img_size = img_pil.size[::-1]  # [h, w]
            # set label
            img_name = img_name[0]
            img_label = img_label.numpy().ravel()
            img_label_sq = np.array(np.nonzero(img_label)).ravel()

            # create crms
            crms_fuo = None
            crms_list= {}
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # predict
                preds, crms = self.netloc_(img_tensor, img_size)
                # detach
                crms = crms.detach().cpu().squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms
                else: crms_fuo = crms_fuo + crms
            # add fusion
            crms_list[999] = crms_fuo

            # visualize - single & fuse
            crm_cpy = torch.zeros(img_size)
            for img_scale, crms in crms_list.items():
                # point localization && bbox localization
                for cls_idx in img_label_sq:
                    crm = crms[cls_idx]
                    # localize - multi
                    crm_cpy.copy_(crm)
                    pred_bboxes_multi = self.hm_helper_.localize_bbox(
                                                        img_size=img_size,   
                                                        heatmap=crm_cpy, 
                                                        cls_idx=cls_idx, 
                                                        multi_objects=True,
                                                        thrd=seg_thrd)
                    # visualize
                    # : orginal crm
                    crm_cpy.copy_(crm)
                    savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(img_scale) + '_0.png')
                    self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, norm=True, savepath=savepath, thrd=-1)
                    savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(img_scale) + '_0_thrd.png')
                    self.hm_helper_.visualize_crm(img_pil, crm_cpy, upsample=False, norm=True, savepath=savepath, thrd=seg_thrd)
                     # vis bbox
                    cls_grth_bboxes = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:,1] == cls_idx, :]
                    img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                    savepath = os.path.join(crm_root, img_name + '_' + str(cls_idx) + '_' + str(img_scale) + '_6.png')
                    self.hm_helper_.visualize_bbox_with_grth(img_pil, pred_bboxes_multi, img_grth_bboxes, savepath=savepath, color='blue')
                        
            # break


    def val_single_scale_for_segment(self, img_list, img_root, img_scales, crm_root):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])        
        # eval
        for img_idx, img_name in enumerate(img_list):
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name)
            img_pil  = Image.open(img_path)
            img_size = img_pil.size[::-1]   # [h, w]

            # create crms
            crms_fuo = None
            crms_list= {}
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # forward
                _, crms = self.netloc_(img_tensor, img_size)
                # detach
                crms = crms.detach().cpu().squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms
                else: crms_fuo = crms_fuo + crms
                # list
                crms_list[img_scale] = copy.deepcopy(crms)

            # visualize - single
            for img_scale in img_scales:
                crms = crms_list[img_scale]
                # : convert crm to lbl
                _, lbl_map = crms.max(dim=0)
                # : visualize lbl
                lbl_img = self.hm_helper_.visualize_label_map(lbl_map.data.numpy())
                # : save
                lbl_img = transforms.ToPILImage()(lbl_img)
                savepath = os.path.join(crm_root, img_name + '_epoch36_' + str(img_scale) + '.png')
                lbl_img.save(savepath)

            # # visualize - fuse
            # # : convert crm to lbl
            # _, lbl_map = crms_fuo.max(dim=0)
            # # : visualize lbl
            # lbl_img = self.hm_helper_.visualize_label_map(lbl_map.data.numpy())
            # # : save
            # lbl_img = transforms.ToPILImage()(lbl_img)
            # savepath = os.path.join(crm_root, img_name + '_' + '_fuse.png')
            # lbl_img.save(savepath)


    def val_single_scale_for_segment_basedon_dataloder(self, val_loader, img_root, img_scales, crm_root):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])        
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (_, img_label, img_name, img_size) in enumerate(val_loader):
            # open image
            if 'voc' in self.ds_name_: img_path = os.path.join(img_root, 'JPEGImages', img_name[0] + '.jpg')
            elif 'coco' in self.ds_name_: img_path = os.path.join(img_root, 'val2014', img_name[0])
            img_pil  = Image.open(img_path)
            img_size = img_pil.size[::-1]  # [h, w]
            # set label
            img_name  = img_name[0]

            # check
            savepath = os.path.join(crm_root, img_name + '.seg')
            if os.path.exists(savepath): continue

            # create crms
            crms_fuo = None
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # forward
                _, crms = self.netloc_(img_tensor, img_size)
                # detach
                crms = crms.detach().cpu().squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms
                else: crms_fuo = crms_fuo + crms

            # visualize - fuse
            # : convert crm to lbl
            _, lbl_map = crms_fuo.max(dim=0)
            # : save
            pickle.dump(lbl_map.data.numpy(), open(savepath, 'wb+'))


    def val_single_scale_for_segment_basedon_dataloder_ilsvrc(self, val_loader, img_root, img_scales, crm_root):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])        
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)        
        for img_idx, (_, img_label, img_path, img_size) in enumerate(val_loader):
            img_path = img_path[0]
            # open image
            img_pil  = Image.open(img_path).convert('RGB')
            img_size = img_pil.size[::-1]  # [h, w]
            # set label
            img_name  = os.path.basename(img_path).replace('.jpg', '').replace('.JPG', '').replace('.JPEG', '')

            # check
            savepath = os.path.join(crm_root, img_name + '.seg')
            if os.path.exists(savepath): continue

            # create crms
            crms_fuo = None
            for img_scale in img_scales:
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # forward
                _, crms = self.netloc_(img_tensor, img_size)
                # detach
                crms = crms.detach().cpu().squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms
                else: crms_fuo = crms_fuo + crms

            # visualize - fuse
            # : convert crm to lbl
            _, lbl_map = crms_fuo.max(dim=0)
            # : save
            pickle.dump(lbl_map.data.numpy(), open(savepath, 'wb+'))
