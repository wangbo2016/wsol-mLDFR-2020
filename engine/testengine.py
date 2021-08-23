import os
import sys
import time
import pickle
import datetime
import numpy as np
import operator
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchnet as tnt
from PIL import Image
from tqdm import tqdm
from functools import reduce
from collections import OrderedDict
from tools.cmetric import MultiClassificationMetric, MultilabelClassificationMetric, accuracy
from tools.dmetric import cor_loc_s, calc_ious, calc_ious_cross
from tools.dhelper import localize_bbox, draw_bbox, draw_hmap 
from tools.pytorch_tools import print_model_param_nums, print_model_param_flops
from models.vgg16bnlocalizer import VGG16BNLocalizer
# from models.vgg16bnlocalizerhma import VGG16BNLocalizer
# from models.vgg16bnlocalizercbam import VGG16BNLocalizer
# from models.densenet161localizer import DenseNet161Localizer
from models.densenet161localizercbam import DenseNet161Localizer
# from models.densenet161localizerhma import DenseNet161Localizer
from models.resnet50localizerhma import ResNet50Localizer

# from models.VGG16LocalizerYCSAtt import VGG16Localizer
# from backbone.VGG16LocalizerY import VGG16Localizer
# # from backbone.VGG16BNLocalizerY import VGG16BNLocalizer
# from backbone.VGG16BNLocalizerZCSAtt import VGG16BNLocalizer
# # from backbone.VGG16BNLocalizerYGAPCBAM import VGG16Localizer as VGG16BNLocalizer
# from backbone.InceptionV3LocalizerYCSAtt import InceptionV3Localizer
# # from backbone.ResNet50LocalizerX import ResNet50Localizer
# from backbone.ResNet50LocalizerYCSAtt import ResNet50Localizer
# from backbone.ResNet50Logits import ResNet50Logits
# # from backbone.MobileNetV2LocalizerY import MobileNetLocalizer
# # from backbone.MobileNetV2LocalizerX import MobileNetLocalizer
# from backbone.MobileNetV2LocalizerZCSAtt import MobileNetLocalizer
# from backbone.MobileNetV2Logits import MobileNetLogits
# from backbone.DenseNet161LocalizerXCSAtt import DenseNet161Localizer
# from backbone.GoogLeNetBNLocalizeXCSAtt import GoogLeNetBNLocalizer
# # from backbone.ResNet50LocalizerZCSAtt_NBN import ResNet50Localizer_NBN
# from backbone.ResNet50LocalizerY_NBN import ResNet50Localizer_NBN

class TestEngine(object):
    # Func:
    #   Constructor.
    def __init__(self, cfg, img_names, img_labels, obj_bboxes, categories):
        # init setting
        self.device_ = ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg_        = cfg
        self.img_names_  = img_names
        self.img_labels_ = np.array(img_labels)
        self.obj_bboxes_ = np.array(obj_bboxes)
        self.categories_ = categories        
        # create tool
        self.cls_meter_  = MultilabelClassificationMetric()
        self.los_meter_  = MultiClassificationMetric()
        self.top1_meter_ = MultiClassificationMetric()
        self.top5_meter_ = MultiClassificationMetric()

    # Func:
    #   Create experimental environment 
    def create_env(self):
        # create network 
        if self.cfg_.experiment_type == 'cls':
            if   self.cfg_.network.name == 'vgg16':
                self.netloc_ = VGG16Localizer(cls_num=self.cfg_.network.out_size)
            elif self.cfg_.network.name == 'vgg16bn':
                self.netloc_ = VGG16BNLocalizer(cls_num=self.cfg_.network.out_size)  
            elif self.cfg_.network.name == 'googlenetbn':
                self.netloc_ = GoogLeNetBNLocalizer(cls_num=self.cfg_.network.out_size)                  
            elif self.cfg_.network.name == 'resnet50':
                self.netloc_ = ResNet50Logits(cls_num=self.cfg_.network.out_size)             
            elif self.cfg_.network.name == 'mobilenetv2':
                self.netloc_ = MobileNetLogits(cls_num=self.cfg_.network.out_size)                        
        else:
            if   self.cfg_.network.name == 'vgg16':
                self.netloc_ = VGG16Localizer(cls_num=self.cfg_.network.out_size)
            elif self.cfg_.network.name == 'vgg16bn':
                self.netloc_ = VGG16BNLocalizer(cls_num=self.cfg_.network.out_size)  
            elif self.cfg_.network.name == 'googlenetbn':
                self.netloc_ = GoogLeNetBNLocalizer(cls_num=self.cfg_.network.out_size)                        
            elif self.cfg_.network.name == 'inceptionv3':
                self.netloc_ = InceptionV3Localizer(cls_num=self.cfg_.network.out_size)                
            elif self.cfg_.network.name == 'resnet50':
                self.netloc_ = ResNet50Localizer(cls_num=self.cfg_.network.out_size)  
            elif self.cfg_.network.name == 'resnet50_nbn':
                self.netloc_ = ResNet50Localizer_NBN(cls_num=self.cfg_.network.out_size)                  
            elif self.cfg_.network.name == 'mobilenetv2':
                self.netloc_ = MobileNetLocalizer(cls_num=self.cfg_.network.out_size)     
            elif self.cfg_.network.name == 'densenet161':
                self.netloc_ = DenseNet161Localizer(cls_num=self.cfg_.network.out_size)   
                
        # load state_dict(with dataparallel)
        checkpoint = torch.load(self.cfg_.test.params_path, map_location='cuda:0')
        state_dict = checkpoint['state_dict']
        # create new state_dict(single gpu)
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            state_dict_new[k[7:]] = v
        # load params
        self.netloc_.load_state_dict(state_dict_new)
        # set cuda
        self.netloc_.cuda(device=self.cfg_.test.device_ids[0])
        # switch to eval mode
        self.netloc_.eval()        
        # create loss function
        if   self.cfg_.train.type == 'multi-label':
            self.criterion_ = nn.MultiLabelSoftMarginLoss().cuda()
        elif self.cfg_.train.type == 'multi-class':
            self.criterion_ = nn.CrossEntropyLoss().cuda()
        # print(self.netloc_)

    # Func:
    #   Validate multi label classification.
    def val_multi_label(self, val_loader):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # reset meter
        self.cls_meter_.reset()
        self.los_meter_.reset()
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True)
        for img_idx, (img_tensor, img_label, img_name, img_size) in enumerate(val_loader):
            # set cuda
            img_tensor = img_tensor.cuda()
            img_label = img_label.cuda()
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
        loss  = self.los_meter_.mean
        endtime = datetime.datetime.now()
        print('log: val_map   is %f(%f), val_loss   is %f, time is %d' % (mAP, mAP2, loss, (endtime - starttime).seconds))
        print('log: val_ap')
        print(aAP)
        return aAP, mAP, loss

    # Func:
    #   Validate multi classification.
    def val_multi_class(self, val_loader, scores_file=None):
        np.set_printoptions(suppress=True)
        starttime = datetime.datetime.now()
        # check
        if scores_file is not None and scores_file != '':
            res_list = []
            res_arr  = None
        # switch to train mode
        self.netloc_.eval()
        self.los_meter_.reset()
        self.top1_meter_.reset()
        self.top5_meter_.reset()
        # eval
        with torch.no_grad():
            val_loader = tqdm(val_loader, desc='valid', ascii=True)
            for img_idx, (img_tensor, img_label, _, _) in enumerate(val_loader):
                # set cuda
                img_tensor = img_tensor.cuda()
                img_label = img_label.cuda()
                # calc forward
                preds, _ = self.netloc_(img_tensor)
                # check 
                if preds.dim() == 1: preds = preds.unsqueeze(dim=0)                        
                # calc acc & loss
                loss = self.criterion_(preds, img_label)
                # accumulate loss & acc
                acc1, acc5 = accuracy(preds, img_label, topk=(1, 5))
                self.los_meter_.update(loss.item())
                self.top1_meter_.update(acc1[0])
                self.top5_meter_.update(acc5[0])
                # check
                if scores_file is not None and scores_file != '':
                    if res_arr is None: res_arr = preds.cpu().data.numpy()
                    else: res_arr = np.concatenate((res_arr, preds.cpu().data.numpy()), axis=0)
        # eval 
        top1   = self.top1_meter_.mean
        top5   = self.top5_meter_.mean
        loss   = self.los_meter_.mean
        endtime= datetime.datetime.now()
        print('log: val_top1   is %f, val_top5   is %f, val_loss   is %f, time is %d' % (top1, top5, loss, (endtime - starttime).seconds))
        # check
        if scores_file is not None and scores_file != '':
            pickle.dump(res_arr, open(scores_file, 'wb+'))
        # return
        return top1, top5, loss

    # Func:
    #   Predict bbox for corloc & clsloc.
    def extract_boxes(self, val_loader, img_scales, locbox_root, scores_file, cam_segthd=0.5):
        # create transform
        transformers = {}
        for img_scale in img_scales:
            transformers[img_scale] = transforms.Compose([
                                        transforms.Resize(img_scale),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
        # load prediction scores
        preds_scores = pickle.load(open(scores_file, 'rb+'))
        # calc prediction top-k cls-idx
        preds_clsidx_top5 = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, img_path, img_size) in enumerate(val_loader):
            # check
            corloc_path = os.path.join(locbox_root, str(img_idx) + '.corloc')
            clsloc_path = os.path.join(locbox_root, str(img_idx) + '.clsloc')
            if os.path.exists(corloc_path) and os.path.exists(clsloc_path): continue
            # open image
            img_label = img_label.numpy().ravel()
            img_pil = Image.open(img_path[0]).convert('RGB')

            # create cams
            cams_lst = []
            for img_scale in img_scales:
                # trans
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda()  
                # predict
                _, cams = self.netloc_(img_tensor)
                # save
                cams_lst.append(cams)
            
            # --------------------------------------------------------
            # corloc
            # --------------------------------------------------------
            # load cls idx
            cls_idx = img_label[0]
            # extract cam
            cam_fus = None
            for cams in cams_lst:
                cam = cams[:,cls_idx,:,:].unsqueeze(dim=0)
                cam = cam.detach().cpu()
                # upsample
                cam = F.interpolate(cam, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                cam = cam.squeeze()
                # fuse
                if cam_fus is None: cam_fus = cam
                else: cam_fus = cam_fus + cam
            # thrd
            cam_fus[cam_fus<cam_fus.mean()] = 0
            # localize - single point & box
            corloc_bboxes = []  # correct localization
            pred_bbox = localize_bbox(heatmap=cam_fus, multiloc=False, thrd=cam_segthd)
            # save
            corloc_bboxes += [(img_idx, cls_idx,) + p for p in pred_bbox]

            # --------------------------------------------------------
            # clsloc
            # --------------------------------------------------------
            clsloc_bboxes = []  # prediction with localization                                          
            for cls_idx in preds_clsidx_top5[img_idx]:
                # extract cam
                cam_fus = None
                for cams in cams_lst:
                    cam = cams[:,cls_idx,:,:].unsqueeze(dim=0)
                    cam = cam.detach().cpu()
                    # upsample
                    cam = F.interpolate(cam, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                    cam = cam.squeeze()
                    # fuse
                    if cam_fus is None: cam_fus = cam
                    else: cam_fus = cam_fus + cam                    
                # thrd
                cam_fus[cam_fus<cam_fus.mean()] = 0
                # localize - single point & box
                pred_bbox = localize_bbox(heatmap=cam_fus, multiloc=False, thrd=cam_segthd)
                # save 
                clsloc_bboxes += [(img_idx, cls_idx,) + p for p in pred_bbox]
            clsloc_bboxes = [reduce(operator.add, clsloc_bboxes)]

            # save
            corloc_bboxes = np.array(corloc_bboxes)
            clsloc_bboxes = np.array(clsloc_bboxes)
            pickle.dump(corloc_bboxes, open(corloc_path, 'wb+'))  
            pickle.dump(clsloc_bboxes, open(clsloc_path, 'wb+'))  

            del cams, cam_fus
            gc.collect()

    # Func:
    #   Calculate corloc & clsloc.
    def analyze_performance(self, val_loader, locbox_root, scores_file, corloc_file=None, clsloc_file=None, iou_thrd=0.5):
        # load prediction scores
        preds_scores = pickle.load(open(scores_file, 'rb+'))
        # eval
        corloc_bboxes = None  # correct localization
        clsloc_bboxes = None  # prediction with localization                                          
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, _, img_size) in enumerate(val_loader):
            # set label
            img_label = img_label.numpy().ravel()
            # check
            corloc_path = os.path.join(locbox_root, str(img_idx) + '.corloc')
            clsloc_path = os.path.join(locbox_root, str(img_idx) + '.clsloc')
            # load box
            if corloc_bboxes is None: 
                corloc_bboxes = pickle.load(open(corloc_path, 'rb+'))
            else:
                corloc_bboxes = np.concatenate((corloc_bboxes, pickle.load(open(corloc_path, 'rb+'))), axis=0)
            if clsloc_bboxes is None: 
                clsloc_bboxes = pickle.load(open(clsloc_path, 'rb+'))
            else:
                clsloc_bboxes = np.concatenate((clsloc_bboxes, pickle.load(open(clsloc_path, 'rb+'))), axis=0)

        # check
        gt_labels = np.tile(np.array(self.img_labels_)[:, np.newaxis], 5)
        # calc cls top-k err
        top5_predidx = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        clserr = (top5_predidx != gt_labels)
        top1_clserr = clserr[:, 0].sum() / float(clserr.shape[0])
        top5_clserr = np.min(clserr, axis=1).sum() / float(clserr.shape[0])
        print('log: cls    err top1   is %f, cls    err top5   is %f' % (top1_clserr, top5_clserr))

        # calc loc top-k err
        iou_val = np.zeros((len(self.img_labels_), 5))
        if self.cfg_.dataset.name == 'cub200':
            for k in range(5):
                preds_boxes_k = clsloc_bboxes[:, 2 + 7 * k: 2 + 7 * k + 4]
                gt_bboxes = self.obj_bboxes_[:,2:]
                iou_val[:, k] = calc_ious(preds_boxes_k, gt_bboxes)
            locerr = iou_val < iou_thrd
        elif self.cfg_.dataset.name == 'ilsvrc':
            for k in range(5):
                preds_boxes_k = clsloc_bboxes[:, 2 + 7 * k: 2 + 7 * k + 4]
                for j in range(len(self.img_labels_)):
                    preds_boxes_k_j = preds_boxes_k[j][np.newaxis, :]
                    gt_bboxes_j = self.obj_bboxes_[self.obj_bboxes_[:, 0]==j, 2:]
                    iou_val[j, k] = max(calc_ious_cross(preds_boxes_k_j, gt_bboxes_j))
            locerr = iou_val < iou_thrd

        # calc cls-loc top-k err
        clsloc_err = np.logical_or(locerr, clserr)
        top1_clsloc_err = clsloc_err[:, 0].sum() / float(clsloc_err.shape[0])
        top5_clsloc_err = np.min(clsloc_err, axis=1).sum() / float(clsloc_err.shape[0])
        print('log: clsloc err top1   is %f, clsloc err top5   is %f' % (top1_clsloc_err, top5_clsloc_err))

        # calc gth-loc top-1 err
        corloc = cor_loc_s(grth_labels=self.img_labels_, grth_bboxes=self.obj_bboxes_, pred_bboxes=corloc_bboxes, categories=self.categories_, iou_thrd=iou_thrd)
        print('log: clsloc err(grth)  is %f (CorLoc=%f), IoU=%f' % ((1.0 - corloc), corloc, iou_thrd))

        # save
        if corloc_file is not None and corloc_file != '':
            pickle.dump(corloc_bboxes, open(corloc_file, 'wb+'))
        if clsloc_file is not None and clsloc_file != '':
            pickle.dump(clsloc_bboxes, open(clsloc_file, 'wb+'))

    
    def visualize(self, val_loader, img_scales, locbox_root, vis_root, cam_segthd=0.5):
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
            # set label
            img_label = img_label.numpy().ravel()
            # load box
            corloc_path = os.path.join(locbox_root, str(img_idx) + '.corloc')
            corloc_bbox = pickle.load(open(corloc_path, 'rb+'))
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')

            # load cls idx
            cls_idx = img_label[0]
            # create cams
            cam_fus = None
            for img_scale in img_scales:
                # trans
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda()  
                # predict
                _, cams = self.netloc_(img_tensor)
                # detach
                cam = cams[:,cls_idx,:,:].unsqueeze(dim=0)
                cam = cam.detach().cpu()
                # upsample
                cam = F.interpolate(cam, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                cam = cam.squeeze()
                # fuse
                if cam_fus is None: cam_fus = cam
                else: cam_fus = cam_fus + cam    
            
            # draw bbox
            cls_grth_bboxes = self.obj_bboxes_[self.obj_bboxes_[:,1] == cls_idx, :]
            img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
            savepath_bbox = os.path.join(vis_root, str(img_idx) + '-' + str(cls_idx) + '.bbox.png')
            img_pilz = copy.deepcopy(img_pil)
            draw_bbox(img_pilz, corloc_bbox, img_grth_bboxes, savepath_bbox, pred_color='red', grth_color='blue')
            # draw heatmap
            savepath_cam = os.path.join(vis_root, str(img_idx) + '-' + str(cls_idx) + '.hmap.png')
            img_pilz = copy.deepcopy(img_pil)
            draw_hmap(img_pilz, cam_fus, savepath=savepath_cam, thrd=-1)

            del cam_fus
            gc.collect() 


    def visualize_details(self, val_loader, img_scales, vis_root, cam_segthd=0.5):
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
            # set label
            img_label = img_label.numpy().ravel()
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')

            # load cls idx
            cls_idx = img_label[0]
            # create cams
            cam_fus = None
            cam_lst= {}
            for img_scale in img_scales:
                # trans
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda()  
                # predict
                _, cams = self.netloc_(img_tensor)
                # detach
                cam = cams[:,cls_idx,:,:].unsqueeze(dim=0)
                cam = cam.detach().cpu()
                # upsample
                cam = F.interpolate(cam, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                cam = cam.squeeze()
                # fuse
                if cam_fus is None: cam_fus = cam
                else: cam_fus = cam_fus + cam    
                # list
                cam_lst[img_scale] = copy.deepcopy(cam)        
            # check
            if len(cam_lst) > 1:
                cam_lst[999] = cam_fus

            # predict bbox
            box_list = {}
            for img_scale, cam in cam_lst.items():
                cam[cam<cam.mean()] = 0
                pred_bbox = localize_bbox(heatmap=cam, multiloc=False, thrd=cam_segthd)
                box_list[img_scale] = pred_bbox

            # visualize - single & fuse
            for img_scale, cam in cam_lst.items():
                # draw bbox
                cls_grth_bboxes = self.obj_bboxes_[self.obj_bboxes_[:,1] == cls_idx, :]
                img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
                savepath_bbox = os.path.join(vis_root, str(img_idx) + '-' + str(cls_idx) + '-' + str(img_scale) + '.bbox.png')
                img_pilz = copy.deepcopy(img_pil)
                draw_bbox(img_pilz, box_list[img_scale], img_grth_bboxes, savepath_bbox, pred_color='red', grth_color='blue')
                # draw heatmap
                savepath_cam = os.path.join(vis_root, str(img_idx) + '-' + str(cls_idx) + '-' + str(img_scale) +  '.hmap.png')
                img_pilz = copy.deepcopy(img_pil)
                draw_hmap(img_pilz, cam, savepath=savepath_cam, thrd=-1)

            del cam, cam_fus
            gc.collect() 