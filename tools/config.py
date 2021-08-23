import time
import argparse
import datetime
from .yacs import CfgNode as CN

# Basic setting
_C = CN(new_allowed=True)
_C.experiment_name = 'vgg16 for cub200'
_C.experiment_mode = 'loc'
_C.experiment_extr = 'hma-cs'
_C.experiment_type = 'train'
_C.yaml_file       = 'yamls/vgg16_cub200.yaml'
_C.log_root        = ''

# Network
_C.network = CN(new_allowed=True)
_C.network.name     = 'vgg16'
_C.network.out_size = 200
_C.network.img_size = 224
_C.network.cam_size = 14

# Dataset
_C.dataset = CN(new_allowed=True)
_C.dataset.name     = 'cub200'  # ilsvrc
_C.dataset.root     = '/home/bowang/ds_research/CUB-200-2011'
_C.dataset.crop     = 'RandomResizedCrop'  # RandomCrop

# Train
_C.train = CN(new_allowed=True)
_C.train.type          = 'multi-class'
_C.train.epoch_sp      = 0
_C.train.epoch_ep      = 50
_C.train.batch_size    = 128
_C.train.device_ids    = [0,1,2]
_C.train.worker_num    = 6
_C.train.resume_path   = ''
_C.train.params_path   = ''
# optimizer params
_C.train.optimizer = CN(new_allowed=True)
_C.train.optimizer.name         = 'sgd'
_C.train.optimizer.lr           = 1e-2 * 3
_C.train.optimizer.weight_decay = 1e-4
_C.train.optimizer.momentum     = 0.9
# scheduler params
_C.train.scheduler = CN(new_allowed=True)
_C.train.scheduler.name         = 'max'
_C.train.scheduler.factor       = 0.1
_C.train.scheduler.patience     = 2
_C.train.scheduler.step_size    = 7

# Test
_C.test = CN(new_allowed=True)
_C.test.type          = 'multi-class'
_C.test.batch_size    = 128
_C.test.device_ids    = [0]
_C.test.img_scales    = [224, 336, 448] # 112, 224, 336, 448, 560, 672, 784
_C.test.cam_segthd    = 0.2
_C.test.worker_num    = 0
_C.test.params_path   = ''
_C.test.scores_file   = ''
_C.test.corloc_file   = ''
_C.test.clsloc_file   = ''
_C.test.locbox_root   = ''
_C.test.visual_root   = ''

def _get_cfg_defaults_():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def _parse_args_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='yaml_file', metavar='yaml_file', default=_C.yaml_file, help='experiment configure file name', type=str)
    args = parser.parse_args()
    return args


def get_config():
    args = _parse_args_()
    cfg = _get_cfg_defaults_()
    cfg.defrost()
    cfg.merge_from_file(args.yaml_file)
    # set config file
    cfg.yaml_file = args.yaml_file
    # set log root
    if cfg.experiment_type == 'train':
        cfg.log_root  = 'logs/' + cfg.experiment_mode + '-' + cfg.network.name + '-' + cfg.dataset.name 
        cfg.log_root  = cfg.log_root + '-b' + str(cfg.train.batch_size) + '-f' + str(cfg.network.cam_size)
        cfg.log_root  = cfg.log_root + '-' + cfg.train.optimizer.name + '-' + cfg.train.scheduler.name
        if cfg.train.scheduler.name == 'step':
            cfg.log_root = cfg.log_root + str(cfg.train.scheduler.step_size)
        cfg.log_root  = cfg.log_root + '-' + cfg.experiment_extr + '-' + time.strftime('%Y%m%d%H%M', time.localtime())    
    cfg.freeze()
    # return
    return cfg
    

if __name__ == "__main__":
    print(get_config())