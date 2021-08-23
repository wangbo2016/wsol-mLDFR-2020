import os
import math
import torch
import numpy as np
import numpy.ma as ma

# Func:
#   Calc Corrected Localization Rate. For VOC & COCO.
# Ref:
#   T.Deselaers, B.Alexe, and V.Ferrari. Weakly supervised localization and learning with generic knowledge. IJCV2012    
# Input:
#   pred_bboxes : predicted bboxes,   type is numpy.ndarray, format is [iid, cid, xmin, ymin, xmax, ymax, score]
# Return:
#   mean corloc
def cor_loc_m(grth_labels, grth_bboxes, pred_bboxes, categories, iou_thrd=0.5):
    img_grth_num_per_class = [] # number of images per category
    img_corr_num_per_class = [] # number of correct predictions for each category
    # calc by category
    for cls_idx, cls_name in enumerate(categories):
        cls_pred_bboxes = pred_bboxes[pred_bboxes[:,1] == cls_idx, :]
        cls_grth_bboxes = grth_bboxes[grth_bboxes[:,1] == cls_idx, :]
        cls_grth_iids   = (grth_labels[:,cls_idx] == 1).nonzero()
        cls_grth_iids   = cls_grth_iids[0]
        # count img_grth_num_per_class
        img_grth_num_per_class.append(len(cls_grth_iids))
        # count img_corr_num_per_class
        TP = 0
        for img_idx in cls_grth_iids:
            img_pred_bboxes = cls_pred_bboxes[cls_pred_bboxes[:, 0]==img_idx, 2:6]
            img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
            # check
            if len(img_pred_bboxes) > 0:
                phits, rhits = calc_bbox_hits(img_pred_bboxes, img_grth_bboxes, iou_thrd)
                if phits > 0:
                    TP += 1
        img_corr_num_per_class.append(TP)
    # calc corloc
    img_grth_num_per_class = np.array(img_grth_num_per_class)
    img_corr_num_per_class = np.array(img_corr_num_per_class)
    corloc = np.where(img_grth_num_per_class == 0, np.nan, img_corr_num_per_class / img_grth_num_per_class)
    mcorloc = sum(corloc) / len(corloc)
    return mcorloc

# Func:
#   Calc Corrected Localization Rate. For CUB200 & ILSVRC.
# Ref:
#   T.Deselaers, B.Alexe, and V.Ferrari. Weakly supervised localization and learning with generic knowledge. IJCV2012    
# Input:
#   grth_labels : groundtruth labels, type is numpy.ndarray, size is [N, C]
#   pred_bboxes : predicted bboxes, type is numpy.ndarray, format is [iidx, cid, xmin, ymin, xmax, ymax, score], the number of predicted boxes is limited to 1.
def cor_loc_s(grth_labels, grth_bboxes, pred_bboxes, categories, iou_thrd=0.5):
    img_grth_num_per_class = []
    img_corr_num_per_class = []
    # calc by category
    for cls_idx, cls_name in enumerate(categories):
        cls_pred_bboxes = pred_bboxes[pred_bboxes[:,1] == cls_idx, :]
        cls_grth_bboxes = grth_bboxes[grth_bboxes[:,1] == cls_idx, :]
        cls_grth_iids   = (grth_labels == cls_idx).nonzero()
        cls_grth_iids   = cls_grth_iids[0]
        # count img_grth_num_per_class
        img_grth_num_per_class.append(len(cls_grth_iids))
        # count img_corr_num_per_class
        TP = 0
        for img_idx in cls_grth_iids:
            img_pred_bboxes = cls_pred_bboxes[cls_pred_bboxes[:, 0]==img_idx, 2:6]
            img_grth_bboxes = cls_grth_bboxes[cls_grth_bboxes[:, 0]==img_idx, 2:]
            # check
            if len(img_pred_bboxes) > 0:
                phits, rhits = calc_bbox_hits(img_pred_bboxes, img_grth_bboxes, iou_thrd)
                if phits > 0:
                    TP += 1
        img_corr_num_per_class.append(TP)
    # calc corloc
    img_grth_num_per_class = np.array(img_grth_num_per_class)
    img_corr_num_per_class = np.array(img_corr_num_per_class)
    corloc = np.where(img_grth_num_per_class == 0, np.nan, img_corr_num_per_class / img_grth_num_per_class)
    mcorloc = sum(corloc) / len(corloc)
    return mcorloc


# Func:
#   Calc bbox hit number.
# Input:
#   pred_bboxes : predicted bboxes,   type is numpy.ndarray, format is [xmin, ymin, xmax, ymax]
#   grth_bboxes : groundtruth bboxes, type is numpy.ndarray, format is [xmin, ymin, xmax, ymax]
# Return:
#   hit number, type is int
def calc_bbox_hits(pred_bboxes, grth_bboxes, iou_thrd):
    # check
    if pred_bboxes.ndim == 1:
        pred_bboxes = pred_bboxes[np.newaxis, :]
    # calc
    pls = []
    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        for grth_bbox in grth_bboxes:
            iou = calc_iou(pred_bbox, grth_bbox)
            if  iou >= iou_thrd:
                pls.append((pred_idx, 1, iou))
            else:
                pls.append((pred_idx, 0, 0))
    pls = np.array(pls)
    # filter - phits
    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        temp = pls[pls[:,0]==pred_idx, :]
        max_pos = np.where(temp[:,-1]==np.max(temp[:,-1]))[0][0]
        temp[:max_pos, 1] = 0
        temp[max_pos+1:, 1] = 0
        pls[pls[:,0]==pred_idx, :] = temp
    # filter - rhits
    hits = np.zeros(shape=(len(grth_bboxes)))
    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        temp = pls[pls[:,0]==pred_idx, 1]
        hits = np.logical_or(hits, temp)
    # return
    hits_precision = sum(pls[:,1])
    hits_recall = sum(hits)
    return hits_precision, hits_recall


# Func:
#   Calc iou.
def calc_iou(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])
    inter_area = max(x_b - x_a + 1, 0) * max(y_b - y_a + 1, 0)
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    return inter_area / float(box_a_area + box_b_area - inter_area)        

# Func:
#   Calc ious.
# Input:
#   bboxes_a : type is numpy.ndarray, format is [xmin, ymin, xmax, ymax]
#   bboxes_b : type is numpy.ndarray, format is [xmin, ymin, xmax, ymax]
# Return:
#   ious, type is numpy.ndarray, format is [iou], e.g. [
#                                                        a1, iou(a1,b1); 
#                                                        a1, iou(a1,b2); 
#                                                        a1, iou(a1,b3); 
#                                                        a2, iou(a2,b1); 
#                                                        a2, iou(a2,b2); 
#                                                        a2, iou(a2,b3); 
#                                                      ]
def calc_ious_cross(bboxes_a, bboxes_b):
    obj_num_b = len(bboxes_b)
    obj_num_a = len(bboxes_a)
    bboxes_b = np.tile(bboxes_b, [obj_num_a, 1])
    bboxes_a = np.repeat(bboxes_a, obj_num_b, axis=0)

    area_bi = np.minimum(bboxes_a[:,2:], bboxes_b[:,2:]) - np.maximum(bboxes_a[:,:2], bboxes_b[:,:2]) + 1
    area_bi = np.prod(area_bi.clip(0), axis=1)
    area_bu = (bboxes_b[:,2] - bboxes_b[:,0] + 1) * (bboxes_b[:,3] - bboxes_b[:,1] + 1) + (bboxes_a[:,2] - bboxes_a[:,0] + 1) * (bboxes_a[:,3] - bboxes_a[:,1] + 1) - area_bi
    return area_bi / area_bu


# Func:
#   Calc iou.
# Input:
#   bboxes_a : type is numpy.ndarray, shape is [N, 4], format is [xmin, ymin, xmax, ymax]
#   bboxes_b : type is numpy.ndarray, shape is [N, 4], format is [xmin, ymin, xmax, ymax]
# Return:
#   ious, type is numpy.ndarray, format is [iou], e.g. [
#                                                        iou(a1,b1); 
#                                                        iou(a2,b2); 
#                                                        iou(a3,b3); 
#                                                      ]
def calc_ious(bboxes_a, bboxes_b):
    bboxes_a = np.asarray(bboxes_a, dtype=float)
    bboxes_b = np.asarray(bboxes_b, dtype=float)
    if bboxes_a.ndim == 1: bboxes_a = bboxes_a[np.newaxis, :]
    if bboxes_b.ndim == 1: bboxes_b = bboxes_b[np.newaxis, :]

    iw = np.minimum(bboxes_a[:, 2], bboxes_b[:, 2]) - np.maximum(bboxes_a[:, 0], bboxes_b[:, 0]) + 1
    ih = np.minimum(bboxes_a[:, 3], bboxes_b[:, 3]) - np.maximum(bboxes_a[:, 1], bboxes_b[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (bboxes_a[:, 2] - bboxes_a[:, 0] + 1) * (bboxes_a[:, 3] - bboxes_a[:, 1] + 1)
    box2_area = (bboxes_b[:, 2] - bboxes_b[:, 0] + 1) * (bboxes_b[:, 3] - bboxes_b[:, 1] + 1)

    iou_val = i_area / (box1_area + box2_area - i_area)

    return iou_val