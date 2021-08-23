import os
import torch
import scipy.stats as stats
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .recalibration.attention import *

class GoogLeNetBNLocalizer(nn.Module):
    def __init__(self, cls_num=20):
        super(GoogLeNetBNLocalizer, self).__init__()
        self.cls_num_ = cls_num
        # load basic network - googlenet_bn
        googlenet_bn = models.googlenet(pretrained=True, aux_logits=False)
        
        # create feature module - backbone, googlenet_bn/inception_v2
        self.conv1    = googlenet_bn.conv1      # output: N x 64 x 112 x 112 (input : N x 3 x 224 x 224) 
        self.maxpool1 = googlenet_bn.maxpool1   # output: N x 64 x 56 x 56
        self.conv2    = googlenet_bn.conv2      # output: N x 64 x 56 x 56  
        self.conv3    = googlenet_bn.conv3      # output: N x 192 x 56 x 56
        self.maxpool2 = googlenet_bn.maxpool2   # output: N x 192 x 28 x 28

        self.inception3a = googlenet_bn.inception3a     # output: N x 256 x 28 x 28
        self.inception3b = googlenet_bn.inception3b     # output: N x 480 x 28 x 28
        self.maxpool3    = googlenet_bn.maxpool3        # output: N x 480 x 14 x 14

        self.inception4a = googlenet_bn.inception4a     # output: N x 512 x 14 x 14
        self.inception4b = googlenet_bn.inception4b     # output: N x 512 x 14 x 14
        self.inception4c = googlenet_bn.inception4c     # output: N x 512 x 14 x 14
        self.inception4d = googlenet_bn.inception4d     # output: N x 528 x 14 x 14
        self.inception4e = googlenet_bn.inception4e     # output: N x 832 x 14 x 14
        self.dropout     = nn.Dropout2d(p=0.2)

        # create transition module
        self.extensio = nn.Sequential(
            nn.Conv2d(in_channels=832, out_channels=832, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(832),
            nn.ReLU(inplace=True), 
        )  
        
        self.inception5a = googlenet_bn.inception5a     # output: N x 832 x 14 x 14
        self.inception5b = googlenet_bn.inception5b     # output: N x 1024 x 14 x 14

        # create attention module for 56x56
        self.csatt56 = CSAttention(64, ['channel', 'spatial'], 7)

        # create attention module for 28x28
        self.csatt28 = CSAttention(192, ['channel', 'spatial'], 5)

        # create attention module for 14x14
        self.csatt14 = CSAttention(480, ['channel', 'spatial'], 3)

        # create logits
        self.fc = nn.Conv2d(in_channels=1024, out_channels=self.cls_num_, kernel_size=1, stride=1, padding=0)        

    # input: 1, 3, 224, 224
    def forward(self, x, im_size=None):
        # feature
        x = self.conv1(x)       # output: N x 64 x 112 x 112
        x = self.maxpool1(x)    # output: N x 64 x 56 x 56
        # attention for 56      # output: [1,64,56,56]
        x = self.csatt56(x)

        x = self.conv2(x)       # N x 64 x 56 x 56
        x = self.conv3(x)       # N x 192 x 56 x 56
        x = self.maxpool2(x)    # N x 192 x 28 x 28
        # attention for 28      # output: [1,192,28,28]
        x = self.csatt28(x)

        x = self.inception3a(x) # N x 256 x 28 x 28
        x = self.inception3b(x) # N x 480 x 28 x 28
        x = self.maxpool3(x)    # N x 480 x 14 x 14
        # attention for 14      # output: [1,480,14,14]
        x = self.csatt14(x)

        x = self.inception4a(x) # N x 512 x 14 x 14
        x = self.inception4b(x) # N x 512 x 14 x 14
        x = self.inception4c(x) # N x 512 x 14 x 14
        x = self.inception4d(x) # N x 528 x 14 x 14
        x = self.inception4e(x) # N x 832 x 14 x 14
        # extension
        x = self.extensio(x)    # N x 832 x 14 x 14              
        x = self.inception5a(x) # N x 832 x 14 x 14
        x = self.inception5b(x) # N x 1024 x 14 x 14
        x = self.dropout(x)
        
        # logits
        cams = self.fc(x)
        pred = F.adaptive_avg_pool2d(cams, output_size=(1,1)).squeeze()

        # return
        return pred, cams


    def init(self, params_path=None):
        # init weights by distribution
        for n, m in self.named_modules():
            if 'fc' in n or 'csatt' in n or 'extensio' in n:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    X = stats.truncnorm(-2, 2, scale=0.01)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        print('log: init by xavier_normal_!')