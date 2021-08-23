import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datapu.cub200 import CUBProcessor
from datapu.imagenet import ImageNetProcessor
from engine.trainengine import TrainEngine
from tools.config import get_config

# get experimental params
cfg = get_config()
writer = SummaryWriter(cfg.log_root)
print(cfg)

# ----------------------------------
# Init
# ----------------------------------
# create transformer
if   cfg.dataset.crop == 'RandomResizedCrop':
    train_transformer = transforms.Compose([
        transforms.RandomResizedCrop(cfg.network.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
elif cfg.dataset.crop == 'RandomCrop':
    train_transformer = transforms.Compose([
            transforms.Resize(cfg.network.img_size+32),
            transforms.RandomCrop(cfg.network.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])    
val_transformer = transforms.Compose([
        transforms.Resize(cfg.network.img_size+32),
        transforms.CenterCrop(cfg.network.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# create dataset
if   cfg.dataset.name == 'cub200':
    trainset = CUBProcessor(dataset_root=cfg.dataset.root, dataset_type='train', load_obj_label=False, transform=train_transformer)
    valset = CUBProcessor(dataset_root=cfg.dataset.root, dataset_type='test', load_obj_label=True, transform=val_transformer)
elif cfg.dataset.name == 'ilsvrc':
    trainset = ImageNetProcessor(dataset_root=cfg.dataset.root, dataset_type='train', load_obj_label=True, transform=train_transformer)
    valset = ImageNetProcessor(dataset_root=cfg.dataset.root, dataset_type='val', load_obj_label=True, transform=val_transformer)

# create dataloader
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.worker_num, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.worker_num, pin_memory=True, drop_last=True)

# ----------------------------------
# Create engine
# ----------------------------------
train_engine = TrainEngine(cfg=cfg)
train_engine.create_env()

# ----------------------------------
# Train & Val & Test
# ----------------------------------
best_test_mAP = 0.0
best_test_idx = 0.0
for epoch_idx in range(cfg.train.epoch_sp, cfg.train.epoch_ep):
    # train
    train_top1, train_top5, train_loss, train_lr = train_engine.train_multi_class(train_loader=train_loader, epoch_idx=epoch_idx)
    # test
    test_top1, test_top5, test_loss = train_engine.val_multi_class(val_loader=val_loader, epoch_idx=epoch_idx)
    # check mAP and save
    train_engine.save_checkpoint(cfg.log_root, epoch_idx, train_top1, test_top1)            

    if test_top1 > best_test_mAP:
        best_test_mAP = test_top1
        best_test_idx = epoch_idx

    # curve all mAP & mLoss
    writer.add_scalars('top1', {'train': train_top1, 'valid': test_top1}, epoch_idx)
    writer.add_scalars('top5', {'train': train_top5, 'valid': test_top5}, epoch_idx)
    writer.add_scalars('loss', {'train': train_loss, 'valid': test_loss}, epoch_idx)
    # curve lr
    writer.add_scalar('train_lr', train_lr, epoch_idx)    

print('best test top1: %f, idx: %d' % (best_test_mAP, best_test_idx))