import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tools.config import get_config
from datapu.cub200 import CUBProcessor
from datapu.imagenet import ImageNetProcessor
from engine.testengine import TestEngine

# get experimental params
cfg = get_config()
print(cfg)

# create transformer
val_transformer = transforms.Compose([
        transforms.Resize(cfg.network.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
# create dataset
if   cfg.dataset.name == 'cub200':
    valset = CUBProcessor(dataset_root=cfg.dataset.root, dataset_type='test', load_obj_label=True, transform=val_transformer)
elif cfg.dataset.name == 'ilsvrc':
    valset = ImageNetProcessor(dataset_root=cfg.dataset.root, dataset_type='val', load_obj_label=True, transform=val_transformer)
# create dataloader
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.worker_num, pin_memory=True)
# create engine
test_engine = TestEngine(cfg=cfg, img_names=valset.img_names_, img_labels=valset.img_labels_, obj_bboxes=valset.obj_labels_, categories=valset.categories_)
test_engine.create_env()

# ----------------------------------
# Val - Classification
# ----------------------------------
# classification 
test_engine.val_multi_class(val_loader=val_loader, scores_file=cfg.test.scores_file)

# ----------------------------------
# Val - Localization
# ----------------------------------
# extract box
if not os.path.exists(cfg.test.locbox_root): os.makedirs(cfg.test.locbox_root)
test_engine.extract_boxes(
    val_loader=val_loader, 
    img_scales=cfg.test.img_scales, 
    cam_segthd=cfg.test.cam_segthd,    
    locbox_root=cfg.test.locbox_root, 
    scores_file=cfg.test.scores_file)

# analyze performance
test_engine.analyze_performance(
    val_loader=val_loader, 
    locbox_root=cfg.test.locbox_root, 
    scores_file=cfg.test.scores_file,
    corloc_file=cfg.test.corloc_file, 
    clsloc_file=cfg.test.clsloc_file,
    iou_thrd=0.3) 
test_engine.analyze_performance(
    val_loader=val_loader, 
    locbox_root=cfg.test.locbox_root, 
    scores_file=cfg.test.scores_file,
    corloc_file=cfg.test.corloc_file, 
    clsloc_file=cfg.test.clsloc_file,
    iou_thrd=0.5) 
test_engine.analyze_performance(
    val_loader=val_loader, 
    locbox_root=cfg.test.locbox_root, 
    scores_file=cfg.test.scores_file,
    corloc_file=cfg.test.corloc_file, 
    clsloc_file=cfg.test.clsloc_file,
    iou_thrd=0.7) 
