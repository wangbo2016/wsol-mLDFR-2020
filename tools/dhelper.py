import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import scipy.ndimage  
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageDraw, ImageFont


# Func:
#   Localize object bbox.
# Input:
#   heatmap     : image heatmap, type is Tensor, format is [h,w].
def localize_bbox(heatmap, multiloc=True, thrd=0.3):
    # localize bboxes
    preds = []
    # norm[0-1]
    heatmap_norm = (heatmap-torch.min(heatmap))/(torch.max(heatmap)-torch.min(heatmap))
    # filter
    heatmap_norm[heatmap_norm<thrd] = 0
    # numpy
    heatmap = heatmap.squeeze().data.numpy()
    heatmap_norm = heatmap_norm.squeeze().data.numpy()
    # mask
    heatmap_mask = heatmap_norm >= (heatmap_norm.mean())
    # localize
    if multiloc:
        # find multiple object instances
        obj_maps, obj_num = scipy.ndimage.label(heatmap_mask)
        for obj_idx in range(obj_num):
            obj_map = (obj_maps == (obj_idx + 1))
            preds += [extract_bbox_from_map(obj_map) + (heatmap[obj_map].mean(),), ]
    else:
        # find only one object instance
        preds += [extract_bbox_from_map(heatmap_mask) + (heatmap.mean(),), ]
    # return
    return preds

# Func:
#   Extract bbox from heatmap.
def extract_bbox_from_map(heatmap):
    assert heatmap.ndim == 2, 'Invalid heatmap shape'
    rows = np.any(heatmap, axis=1)
    cols = np.any(heatmap, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

# Func:
#   Extract point from heatmap.
def extract_point_from_map(heatmap, max_value=None):
    assert heatmap.ndim == 2, 'Invalid heatmap shape'
    if max_value is None:
        ymax, xmax = np.where(heatmap==np.max(heatmap))
    else:
        ymax, xmax = np.where(heatmap==max_value)
    ymax = ymax[0]  # rows
    xmax = xmax[0]  # cols
    return xmax, ymax

# Func:
#   NMS.
# Ref: 
#   Ross Girshick(Fast R-CNN)    
# Input:
#   dets    : bbox array, type is numpy.ndarray, format is [xmin, ymin, xmax, ymax, score]
def nms(dets, nms_thrd):
    # iou constraint: nms
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thrd)[0]
        order = order[inds + 1]

    return keep