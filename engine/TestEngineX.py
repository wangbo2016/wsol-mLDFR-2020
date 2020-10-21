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
import torchvision.transforms as transforms
import torchnet as tnt
from torch.nn.parameter import Parameter
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from module.VGG16LocalizerYGAPCSAtt import VGG16Localizer
# from module.VGG16BNLocalizerYGAPCBAM import VGG16Localizer
from module.VGG19BNLocalizerYGAPCSAtt import VGG19Localizer
from module.InceptionV3LocalizerYGAPCBAM import InceptionV3Localizer
# from module.GoogLeNetBNLocalizerGAPCSAtt import GoogLeNetBNLocalizer
from module.GoogLeNetLocalizerGAPCSAtt import GoogLeNetLocalizer

from tkit.ClassificationEvaluator import ClassificationEvaluator
from tkit.DetectionEvaluator import DetectionEvaluator
from tkit.HeatMapHelper import HeatMapHelper
from tkit.utils import accuracy, AverageMeter
import gc

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


    def val_multi_scale_for_corloc_ilsvrc(self, val_loader, img_root, img_scales, crm_root, seg_thrd=0.5):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # check
        all_pred_bboxloc_singl = []
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, _, img_size) in enumerate(val_loader):
            # set label
            img_label = img_label.numpy().ravel()
            # load crms
            crms_fuo = None
            for img_scale in img_scales:
                crms_path = os.path.join(crm_root, str(img_idx) + '_' + str(img_scale) + '.crms')
                crms      = pickle.load(open(crms_path, 'rb+'))
                # upsample
                crms = F.interpolate(crms.unsqueeze(0), size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                crms = crms.squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms.squeeze()
                else: crms_fuo = crms_fuo + crms.squeeze()

            # point / bbox prediction(classification and localization) && detection && localization: by crm
            cls_idx = img_label[0]
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

            all_pred_bboxloc_singl += [(img_idx,) + p for p in pred_bboxes_singl]

        # evaluate
        print('all_grth_bboxes        : ', len(self.obj_bboxes_))
        print('all_pred_bboxloc_singl : ', len(all_pred_bboxloc_singl))

        # bbox localization - single
        all_pred_bboxloc_singl = np.array(all_pred_bboxloc_singl)
        # bbox corloc
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=all_pred_bboxloc_singl)


    def val_multi_scale_for_corloc_ilsvrc_save(self, val_loader, img_root, img_scales, crm_root):
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
            # if img_idx < 30000: continue
            # if img_idx > 10000: continue
            # if img_idx != 1208: continue    
            # set label
            img_label = img_label.numpy().ravel()
            # open image
            img_pil  = Image.open(img_path[0]).convert('RGB')
            # create crms
            for img_scale in img_scales:
                # check
                crm_path = os.path.join(crm_root, str(img_idx) + '_' + str(img_scale) + '.crms')
                if os.path.exists(crm_path): continue
                # trans
                img_tensor = transformers[img_scale](img_pil).unsqueeze(dim=0).cuda(device=self.gpus_[0])  
                # predict
                _, crms = self.netloc_(img_tensor)
                # detach
                crms = crms.detach().cpu().squeeze()
                # save
                pickle.dump(crms, open(crm_path, 'wb+'))  


    def val_multi_scale_for_clsloc(self, val_loader, img_scales, crm_root, preds_file, seg_thrd=0.5, res_file=None):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        import operator
        from functools import reduce            
        # load preds & preds_bbox
        preds_scores = pickle.load(open(preds_file, 'rb+'))
        # calc cls top-k err
        preds_idx_top5 = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        # eval
        all_pred_bboxes_top5 = []
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, _, _, img_size) in enumerate(val_loader):
            # load crms
            crms_fuo = None
            for img_scale in img_scales:
                crms_path = os.path.join(crm_root, str(img_idx) + '_' + str(img_scale) + '.crms')
                crms      = pickle.load(open(crms_path, 'rb+'))
                # upsample
                crms = F.interpolate(crms.unsqueeze(0), size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                crms = crms.squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms.squeeze()
                else: crms_fuo = crms_fuo + crms.squeeze()

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


    def val_multi_scale_for_clsloc_save(self, val_loader, img_scales, crm_root, preds_file, seg_thrd=0.5, res_dir=None):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        import operator
        from functools import reduce            
        # load preds & preds_bbox
        preds_scores = pickle.load(open(preds_file, 'rb+'))
        # calc cls top-k err
        preds_idx_top5 = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        # eval
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, img_path, img_size) in enumerate(val_loader):
            if img_idx < 40000: continue
            if img_idx > 50000: continue
            # # check
            # if img_size[0] > 1200 or img_size[1] > 1200: 
            #     print(img_path)
            #     print(img_size)                
            #     continue
            # set label
            img_label = img_label.numpy().ravel()            
            # check
            # res_path_01 = os.path.join(res_dir, str(img_idx) + '_01.pred_top5_bbox')
            res_path_02 = os.path.join(res_dir, str(img_idx) + '_02.pred_top5_bbox')
            res_path_03 = os.path.join(res_dir, str(img_idx) + '_03.pred_top5_bbox')
            # res_path_04 = os.path.join(res_dir, str(img_idx) + '_04.pred_top5_bbox')
            res_path_05 = os.path.join(res_dir, str(img_idx) + '.corloc_bbox')
            # if os.path.exists(res_path_01) and os.path.exists(res_path_02) and os.path.exists(res_path_03) and os.path.exists(res_path_04): continue
            if os.path.exists(res_path_02) and os.path.exists(res_path_03): continue
            # check
            resize = False
            if img_size[0] > 1000 and img_size[1] > 1000: 
                # print(img_size)
                img_size[0] = int(img_size[0] / 3.0)
                img_size[1] = int(img_size[1] / 3.0)
                resize = True
            # load crms
            crms     = None
            crms_fuo = None
            proc_mrk = True
            for img_scale in img_scales:
                crms_path = os.path.join(crm_root, str(img_idx) + '_' + str(img_scale) + '.crms')
                # check
                if not os.path.exists(crms_path): 
                    proc_mrk = False
                    break

                # load
                try:
                    crms = pickle.load(open(crms_path, 'rb+'))
                except EOFError:
                    print(crms_path)
                    proc_mrk = False
                    os.remove(crms_path)
                    break                    

                # upsample
                crms = F.interpolate(crms.unsqueeze(0), size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                crms = crms.squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms.squeeze()
                else: crms_fuo = crms_fuo + crms.squeeze()
            # check
            if proc_mrk == False: 
                print(img_path)
                print(img_size)
                continue

            # --------------------------------------------------------
            # corloc
            # --------------------------------------------------------
            all_bboxes_corloc = []
            crm_fus = torch.zeros(img_size)
            # : load cls id
            cls_idx = img_label[0]
            # : copy crm
            crm_fuo = crms_fuo[cls_idx]
            crm_fus.copy_(crm_fuo)
            # : localize - single point & box
            crm_fus[crm_fus<crm_fus.mean()] = 0
            pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
                                                cls_idx=cls_idx, 
                                                multi_objects=False,
                                                upsample=False,
                                                thrd=0.2)

            all_bboxes_corloc += [(img_idx,) + p for p in pred_bboxes_singl]


            # --------------------------------------------------------
            # clsloc
            # --------------------------------------------------------
            # point / bbox prediction(classification and localization) && detection && localization: by crm
            pred_bboxes_top5_01 = []
            pred_bboxes_top5_02 = []
            pred_bboxes_top5_03 = []
            pred_bboxes_top5_04 = []
            for cls_idx in preds_idx_top5[img_idx]:
                crm_fuo = crms_fuo[cls_idx]
                # : localize - single point & box
                crm_fuo[crm_fuo<crm_fuo.mean()] = 0

                '''
                # --------------------------------------------------
                # threshold: 0.1
                # --------------------------------------------------
                # : output - [cid, xmin, ymin, xmax, ymax, score]
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fuo,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=0.1)
                # : check 
                if resize:
                    for i in range(len(pred_bboxes_singl)):
                        cid, xmin, ymin, xmax, ymax, score = pred_bboxes_singl[i]
                        pred_bboxes_singl[i] = (cid, xmin * 3, ymin * 3, xmax * 3, ymax * 3, score)
                # : save                                                                
                pred_bboxes_top5_01 += [(img_idx,) + p for p in pred_bboxes_singl]
                '''

                # --------------------------------------------------
                # threshold: 0.2
                # --------------------------------------------------
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fuo,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=0.2)
                # : check 
                if resize:
                    for i in range(len(pred_bboxes_singl)):
                        cid, xmin, ymin, xmax, ymax, score = pred_bboxes_singl[i]
                        pred_bboxes_singl[i] = (cid, xmin * 3, ymin * 3, xmax * 3, ymax * 3, score)
                # : save                                                    
                pred_bboxes_top5_02 += [(img_idx,) + p for p in pred_bboxes_singl]

                # --------------------------------------------------
                # threshold: 0.3
                # --------------------------------------------------
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fuo,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=0.3)
                # : check 
                if resize:
                    for i in range(len(pred_bboxes_singl)):
                        cid, xmin, ymin, xmax, ymax, score = pred_bboxes_singl[i]
                        pred_bboxes_singl[i] = (cid, xmin * 3, ymin * 3, xmax * 3, ymax * 3, score)
                # : save                                                          
                pred_bboxes_top5_03 += [(img_idx,) + p for p in pred_bboxes_singl]

                '''
                # --------------------------------------------------
                # threshold: 0.4
                # --------------------------------------------------
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fuo,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=0.4)
                # : check 
                if resize:
                    for i in range(len(pred_bboxes_singl)):
                        cid, xmin, ymin, xmax, ymax, score = pred_bboxes_singl[i]
                        pred_bboxes_singl[i] = (cid, xmin * 3, ymin * 3, xmax * 3, ymax * 3, score)
                # : save                                                                
                pred_bboxes_top5_04 += [(img_idx,) + p for p in pred_bboxes_singl]
                '''

            # save clsloc
            # pickle.dump([reduce(operator.add, pred_bboxes_top5_01)], open(res_path_01, 'wb+'))   
            pickle.dump([reduce(operator.add, pred_bboxes_top5_02)], open(res_path_02, 'wb+'))   
            pickle.dump([reduce(operator.add, pred_bboxes_top5_03)], open(res_path_03, 'wb+'))   
            # pickle.dump([reduce(operator.add, pred_bboxes_top5_04)], open(res_path_04, 'wb+'))
            # save corloc
            pickle.dump([reduce(operator.add, all_bboxes_corloc)],   open(res_path_05, 'wb+'))

            del crms_fuo, crms
            gc.collect()

    def val_multi_scale_for_clsloc_save_postproc(self, val_loader, preds_bbox_dir=None):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # eval
        all_pred_bboxes_top5_01 = []
        all_pred_bboxes_top5_02 = []
        all_pred_bboxes_top5_03 = []
        all_pred_bboxes_top5_04 = []
        all_bboxes_corloc       = []
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, _, img_path, img_size) in enumerate(val_loader):
            # check
            # res_path_01 = os.path.join(preds_bbox_dir, str(img_idx) + '_01.pred_top5_bbox')
            res_path_02 = os.path.join(preds_bbox_dir, str(img_idx) + '_02.pred_top5_bbox')
            res_path_03 = os.path.join(preds_bbox_dir, str(img_idx) + '_03.pred_top5_bbox')
            # res_path_04 = os.path.join(preds_bbox_dir, str(img_idx) + '_04.pred_top5_bbox')
            res_path_05 = os.path.join(preds_bbox_dir, str(img_idx) + '.corloc_bbox')
            # if not os.path.exists(res_path_01) or not os.path.exists(res_path_02) or not os.path.exists(res_path_03) or not os.path.exists(res_path_04): 
            if not os.path.exists(res_path_02) or not os.path.exists(res_path_03): 
                print(img_path)
                continue
            # load pred bbox
            # pred_bboxes_top5_01 = pickle.load(open(res_path_01, 'rb+'))
            pred_bboxes_top5_02 = pickle.load(open(res_path_02, 'rb+'))
            pred_bboxes_top5_03 = pickle.load(open(res_path_03, 'rb+'))
            # pred_bboxes_top5_04 = pickle.load(open(res_path_04, 'rb+'))
            corloc_bboxes       = pickle.load(open(res_path_05, 'rb+'))

            # save
            # all_pred_bboxes_top5_01 += pred_bboxes_top5_01
            all_pred_bboxes_top5_02 += pred_bboxes_top5_02
            all_pred_bboxes_top5_03 += pred_bboxes_top5_03
            # all_pred_bboxes_top5_04 += pred_bboxes_top5_04
            all_bboxes_corloc       += corloc_bboxes

        # save
        # all_pred_bboxes_top5_01 = np.array(all_pred_bboxes_top5_01)
        all_pred_bboxes_top5_02 = np.array(all_pred_bboxes_top5_02)
        all_pred_bboxes_top5_03 = np.array(all_pred_bboxes_top5_03)
        # all_pred_bboxes_top5_04 = np.array(all_pred_bboxes_top5_04)
        all_bboxes_corloc       = np.array(all_bboxes_corloc)
        # pickle.dump(all_pred_bboxes_top5_01, open('loc-googlenet-ilsvrc-model-gap-csatt-f14-sgd-max-b256-69.86-87.75-71.24-90.20-52-224-336-448-0.1-mean.preds_bbox', 'wb+'))      
        pickle.dump(all_pred_bboxes_top5_02, open('loc-googlenet-ilsvrc-model-gap-csatt-f14-sgd-max-b256-69.86-87.75-71.24-90.20-52-224-336-448-0.2-mean.preds_bbox', 'wb+'))      
        pickle.dump(all_pred_bboxes_top5_03, open('loc-googlenet-ilsvrc-model-gap-csatt-f14-sgd-max-b256-69.86-87.75-71.24-90.20-52-224-336-448-0.3-mean.preds_bbox', 'wb+'))      
        # pickle.dump(all_pred_bboxes_top5_04, open('loc-googlenet-ilsvrc-model-gap-csatt-f14-sgd-max-b256-69.86-87.75-71.24-90.20-52-224-336-448-0.4-mean.preds_bbox', 'wb+'))      
        pickle.dump(all_bboxes_corloc,       open('loc-googlenet-ilsvrc-model-gap-csatt-f14-sgd-max-b256-69.86-87.75-71.24-90.20-52-224-336-448-0.2-mean.corloc_bbox', 'wb+'))      


    def val_multi_scale_for_corloc_clsloc_ilsvrc(self, val_loader, img_root, img_scales, crm_root, preds_file, seg_thrd=0.5, res_file=None):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # check
        import operator
        from functools import reduce            
        # load preds & preds_bbox
        preds_scores = pickle.load(open(preds_file, 'rb+'))
        # calc cls top-k err
        preds_idx_top5 = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        # eval
        all_bboxes_corloc = []
        all_bboxes_clsloc = []  # prediction with localization      
        val_loader = tqdm(val_loader, desc='validation', ascii=True) 
        for img_idx, (_, img_label, _, img_size) in enumerate(val_loader):
            # set label
            img_label = img_label.numpy().ravel()
            # load crms
            crms_fuo = None
            for img_scale in img_scales:
                crms_path = os.path.join(crm_root, str(img_idx) + '_' + str(img_scale) + '.crms')
                crms      = pickle.load(open(crms_path, 'rb+'))
                # upsample
                crms = F.interpolate(crms.unsqueeze(0), size=(img_size[0], img_size[1]), mode='bilinear', align_corners=False)
                crms = crms.squeeze()
                # fuse
                if crms_fuo is None: crms_fuo = crms.squeeze()
                else: crms_fuo = crms_fuo + crms.squeeze()
            crm_fus = torch.zeros(img_size)

            # --------------------------------------------------------
            # corloc
            # --------------------------------------------------------
            # : load cls id
            cls_idx = img_label[0]
            # : copy crm
            crm_fuo = crms_fuo[cls_idx]
            crm_fus.copy_(crm_fuo)
            # : localize - single point & box
            crm_fus[crm_fus<crm_fus.mean()] = 0
            pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
                                                cls_idx=cls_idx, 
                                                multi_objects=False,
                                                upsample=False,
                                                thrd=seg_thrd)

            all_bboxes_corloc += [(img_idx,) + p for p in pred_bboxes_singl]


            # --------------------------------------------------------
            # clsloc
            # --------------------------------------------------------
            pred_bboxes_top5 = []
            for cls_idx in preds_idx_top5[img_idx]:
                # : copy crm
                crm_fuo = crms_fuo[cls_idx]
                crm_fus.copy_(crm_fuo)
                # : localize - single point & box
                crm_fus[crm_fus<crm_fus.mean()] = 0
                pred_bboxes_singl = self.hm_helper_.localize_bbox(
                                                    img_size=img_size,          # intput format: [h,w], e.g. [375,500]
                                                    heatmap=crm_fus,            # intput format: [h,w], e.g. [14,14]
                                                    cls_idx=cls_idx, 
                                                    multi_objects=False,
                                                    upsample=False,
                                                    thrd=seg_thrd)

                pred_bboxes_top5 += [(img_idx,) + p for p in pred_bboxes_singl]
            # save
            all_bboxes_clsloc += [reduce(operator.add, pred_bboxes_top5)]

        # evaluate
        print('all_grth_bboxes   : ', len(self.obj_bboxes_))
        print('all_bboxes_corloc : ', len(all_bboxes_corloc))

        # bbox localization - single
        all_bboxes_corloc = np.array(all_bboxes_corloc)
        # bbox corloc
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=all_bboxes_corloc)

        # evaluate
        print('all_grth_bboxes   : ', len(self.obj_bboxes_))
        print('all_bboxes_clsloc : ', len(all_bboxes_clsloc))

        # bbox localization - multi-class
        all_bboxes_clsloc = np.array(all_bboxes_clsloc)
        # save
        if res_file is not None:
            pickle.dump(all_bboxes_clsloc, open(res_file, 'wb+'))       


    def val_multi_scale_for_corloc_err_analyze(self, corloc_bbox_file):
        # load corloc_bbox
        all_bboxes_corloc = pickle.load(open(corloc_bbox_file, 'rb+'))
        # corloc
        # : bbox corloc
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=all_bboxes_corloc)      

    def val_multi_scale_for_clsloc_err_analyze(self, preds_file, preds_bbox_file, multi_instances=False):
        np.seterr(divide='ignore', invalid='ignore')
        np.set_printoptions(suppress=True)
        # load preds & preds_bbox
        preds_scores = pickle.load(open(preds_file, 'rb+'))
        preds_bboxes = pickle.load(open(preds_bbox_file, 'rb+'))
        # check - bbox corloc
        preds_bboxes_corloc = preds_bboxes[:,0:7]
        preds_bboxes_corloc[:,1] = np.array(self.img_labels_)
        self.det_meter_.cor_loc_ilsvrc(pred_bboxes=preds_bboxes_corloc)        
        # check
        gt_labels = np.tile(np.array(self.img_labels_)[:, np.newaxis], 5)

        # calc cls top-k err
        top5_predidx = np.argsort(preds_scores, axis=1)[:, ::-1][:, :5]
        clserr = (top5_predidx != gt_labels)
        top1_clserr = clserr[:, 0].sum() / float(clserr.shape[0])
        top5_clserr = np.min(clserr, axis=1).sum() / float(clserr.shape[0])
        print('log: val_top1   is %f, val_top5   is %f' % ((1.0-top1_clserr), (1.0-top5_clserr)))


        # calc loc top-k err
        iou_val = np.zeros((len(self.img_labels_), 5))
        if multi_instances == False:
            for k in range(5):
                preds_boxes_k = preds_bboxes[:, 2 + 7 * k: 2 + 7 * k + 4]
                gt_bboxes = self.obj_bboxes_arr_[:,2:]
                iou_val[:, k] = self.det_meter_.cal_ious(preds_boxes_k, gt_bboxes)
            locerr = iou_val < 0.5
        else:
            for k in range(5):
                preds_boxes_k = preds_bboxes[:, 2 + 7 * k: 2 + 7 * k + 4]
                for j in range(len(self.img_labels_)):
                    preds_boxes_k_j = preds_boxes_k[j][np.newaxis, :]
                    gt_bboxes_j = self.obj_bboxes_arr_[self.obj_bboxes_arr_[:, 0]==j, 2:]
                    iou_val[j, k] = max(self.det_meter_.calc_ious(preds_boxes_k_j, gt_bboxes_j))
            locerr = iou_val < 0.5
    
        # calc cls-loc top-k err
        clsloc_err = np.logical_or(locerr, clserr)
        top1_clsloc_err = clsloc_err[:, 0].sum() / float(clsloc_err.shape[0])
        top5_clsloc_err = np.min(clsloc_err, axis=1).sum() / float(clsloc_err.shape[0])
        print('log: clsloc_err_top1   is %f, clsloc_err_top5   is %f' % (top1_clsloc_err, top5_clsloc_err))


        # calc gth-loc top-1 err
        clserr = np.zeros(shape=(len(preds_scores),5))
        clsloc_err = np.logical_or(locerr, clserr)
        top1_clsloc_err = clsloc_err[:, 0].sum() / float(clsloc_err.shape[0])
        print('log: gthloc_err_top1   is %f' % (top1_clsloc_err))


