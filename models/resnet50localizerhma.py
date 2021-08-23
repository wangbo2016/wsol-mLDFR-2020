import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .recalibration.attention import *

class ResNet50Localizer(nn.Module):
    def __init__(self, cls_num=200):
        super(ResNet50Localizer, self).__init__()
        self.cls_num_ = cls_num
        # load basic network - resnet50
        resnet50 = models.resnet50(pretrained=True)

        # create feature module - backbone - resnet50
        self.conv1 = resnet50.conv1
        self.bn1   = resnet50.bn1
        self.relu  = resnet50.relu
        self.maxpool = resnet50.maxpool # output: 64x112x112
        self.layer1  = resnet50.layer1  # output: 64x56x56
        self.layer2  = resnet50.layer2  # output: 512x28x28
        self.layer3  = resnet50.layer3  # output: 1024x14x14
        self.layer4  = resnet50.layer4 
        # remove last downsample
        self.layer4[0].conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.layer4[0].downsample[0] = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0)

        # create attention module for 56x56
        self.csatt56 = HMAttention(64, ['channel', 'spatial'], 7)    # 'channel', 'spatial'

        # create attention module for 28x28
        self.csatt28 = HMAttention(512, ['channel', 'spatial'], 5)

        # create attention module for 14x14
        self.csatt14 = HMAttention(1024, ['channel', 'spatial'], 3)

        # create logits
        self.fc = nn.Conv2d(in_channels=2048, out_channels=self.cls_num_, kernel_size=1, stride=1, padding=0)        

    def forward(self, x, im_size=None):
        # feature           
        x = self.conv1(x)   # output: [1, 64, 112, 112] 
        x = self.bn1(x)    
        x = self.relu(x)    
        x = self.maxpool(x) # output: [1, 64, 56, 56] 
        # attention for 56  
        x = self.csatt56(x) # output: [1, 64, 56, 56]
        x = self.layer1(x)  # output: [1, 256, 56, 56]  
        x = self.layer2(x)  # output: [1, 512, 28, 28]
        # attention for 28  
        x = self.csatt28(x) # output: [1, 512, 28, 28]
        x = self.layer3(x)  # output: [1, 1024, 14, 14]    
        # attention for 14  
        x = self.csatt14(x) # output: [1, 1024, 14, 14]
        x = self.layer4(x)  # output: [1, 2048, 14, 14]

        # logits
        crms = self.fc(x)
        pred = F.adaptive_avg_pool2d(crms, output_size=(1,1)).squeeze()

        # return
        return pred, crms

    def init(self, params_path=None):
        # init weights by distribution
        for n, m in self.named_modules():
            if 'layer4.0.downsample' in n or 'layer4.0.conv2' in n or 'layer4.0.bn2' in n:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)    
            if 'csatt' in n or 'fc' in n:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)   
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)    
        print('log: init by xavier_normal_!')        

