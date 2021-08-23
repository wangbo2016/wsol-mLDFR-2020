import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .recalibration.attention import *

class VGG16BNLocalizer(nn.Module):
    def __init__(self, cls_num=20):
        super(VGG16BNLocalizer, self).__init__()
        self.cls_num_ = cls_num
        # load basic network - vgg16bn
        vgg16 = models.vgg16_bn(pretrained=True)

        # create feature module - backbone - vgg16[features[0-29]], input_shape=224,224, output_shape=512,14,14
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])    

        # create attention module for 56x56
        self.csatt56 = HMAttention(128, ['channel', 'spatial'], 7)    # 'channel', 'spatial'

        # create attention module for 28x28
        self.csatt28 = HMAttention(256, ['channel', 'spatial'], 5)

        # create attention module for 14x14
        self.csatt14 = HMAttention(512, ['channel', 'spatial'], 3)

        # create extension module
        self.extensio = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # create logits
        self.fc = nn.Conv2d(in_channels=1024, out_channels=self.cls_num_, kernel_size=1, stride=1, padding=0)             

    def forward(self, x):        
        # feature                       input_size:  [1,3,224,298],  output_size: [1,128,56,56]
        x = self.features[0:14](x)

        # attention for 56              output_size: [1,128,56,56]
        x = self.csatt56(x)

        # feature                       input_size:  [1,128,56,56],  output_size: [1,256,28,28]
        x = self.features[14:24](x)

        # attention for 28              output_size: [1,256,28,28]
        x = self.csatt28(x)

        # feature                       input_size:  [1,256,28,28],  output_size: [1,512,14,14]
        x = self.features[24:34](x)

        # attention for 14              output_size: [1,512,14,14]
        x = self.csatt14(x)

        # feature                       input_size:  [1,512,14,14],  output_size: [1,512,14,14]
        x = self.features[34:](x)

        # extension                     output_size: [1,1024,14,18]
        x = self.extensio(x)

        # logits
        crms = self.fc(x)
        pred = F.adaptive_avg_pool2d(crms, output_size=(1,1)).squeeze()

        # return
        return pred, crms
        
    def init(self, params_path=None):
        # init weights by distribution
        for n, m in self.named_modules():
            if 'fc' in n or 'csatt' in n:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
        print('log: init by xavier_normal_!')
