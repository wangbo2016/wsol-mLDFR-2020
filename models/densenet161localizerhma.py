import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .recalibration.attention import *

class DenseNet161Localizer(nn.Module):
    def __init__(self, cls_num=20):
        super(DenseNet161Localizer, self).__init__()
        self.cls_num_ = cls_num
        # load basic network
        densenet = models.densenet161(pretrained=True)

        # create feature module
        self.features = densenet.features    
        # remove last downsample
        self.features.transition3 = nn.Sequential(*list(densenet.features.transition3)[:-1])

        # create attention module for 56x56
        self.csatt56 = HMAttention(96, ['channel', 'spatial'], 7)    # 'channel', 'spatial'

        # create attention module for 28x28
        self.csatt28 = HMAttention(192, ['channel', 'spatial'], 5)

        # create attention module for 14x14
        self.csatt14 = HMAttention(1056, ['channel', 'spatial'], 3)

        # create logits module
        self.classifier = nn.Conv2d(in_channels=2208, out_channels=self.cls_num_, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # feature           
        x = self.features.conv0(x)     # output: [32, 96, 112, 112]
        x = self.features.norm0(x)     # output: [32, 96, 112, 112]
        x = self.features.relu0(x)     # output: [32, 32, 112, 112]
        x = self.features.pool0(x)     # output: [32, 96, 56, 56]
        # attention for 56                   
        x = self.csatt56(x)            # output: [32, 96, 56, 56]

        x = self.features.denseblock1(x)     # output: [32, 384, 56, 56]
        x = self.features.transition1(x)     # output: [32, 192, 28, 28] 
        # attention for 28                   
        x = self.csatt28(x)                  # output: [32, 192, 28, 28]
        x = self.features.denseblock2(x)     # output: [32, 768, 28, 28] 

        x = self.features.transition2(x)     # output: [32, 384,  14, 14] 
        x = self.features.denseblock3(x)     # output: [32, 2112, 14, 14] 
        x = self.features.transition3(x)     # output: [32, 1056, 14, 14] 
        # attention for 14                   
        x = self.csatt14(x)                  # output: [32, 1056, 14, 14]

        x = self.features.denseblock4(x)     # output: [32, 2208, 14, 14] 
        x = self.features.norm5(x)           # output: [32, 2208, 14, 14] 
        x = F.relu(x, inplace=True)

        # logits                    
        cams = self.classifier(x)
        pred = F.adaptive_avg_pool2d(cams, output_size=(1,1)).squeeze()

        # return
        return pred, cams

    def init(self, params_path=None):
        # init weights by distribution
        for n, m in self.named_modules():
            if 'csatt' in n or 'classifier' in n:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
        print('log: init by xavier_normal_!')
