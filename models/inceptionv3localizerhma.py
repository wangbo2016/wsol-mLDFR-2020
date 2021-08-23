import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .recalibration.attention import *

class InceptionV3Localizer(nn.Module):
    def __init__(self, cls_num=20):
        super(InceptionV3Localizer, self).__init__()
        self.cls_num_ = cls_num
        # load basic network - inception_v3
        inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
        
        # create feature module - backbone, inception_v3
        self.Conv2d_1a_3x3 = inception_v3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception_v3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception_v3.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception_v3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception_v3.Conv2d_4a_3x3
        self.Mixed_5b = inception_v3.Mixed_5b
        self.Mixed_5c = inception_v3.Mixed_5c
        self.Mixed_5d = inception_v3.Mixed_5d
        self.Mixed_6a = inception_v3.Mixed_6a
        self.Mixed_6b = inception_v3.Mixed_6b
        self.Mixed_6c = inception_v3.Mixed_6c
        self.Mixed_6d = inception_v3.Mixed_6d
        self.Mixed_6e = inception_v3.Mixed_6e
        # create extension module
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=1280, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),            
        )        
        self.Mixed_7b = inception_v3.Mixed_7b
        self.Mixed_7c = inception_v3.Mixed_7c

        # create attention module for 73x73
        self.csatt73 = HMAttention(64, ['channel', 'spatial'], 7)

        # create attention module for 28x28
        self.csatt35 = HMAttention(192, ['channel', 'spatial'], 5)

        # create attention module for 14x14
        self.csatt17 = HMAttention(768, ['channel', 'spatial'], 3)

        # create logits
        self.fc = nn.Conv2d(in_channels=2048, out_channels=self.cls_num_, kernel_size=1, stride=1, padding=0)             

    # input: 1, 3, 299, 398
    def forward(self, x):
        # feature
        x = self.Conv2d_1a_3x3(x)   # output: 1, 32,  149, 149
        x = self.Conv2d_2a_3x3(x)   # output: 1, 32,  147, 147
        x = self.Conv2d_2b_3x3(x)   # output: 1, 64,  147, 147
        # pool 147->73
        x = F.max_pool2d(x, 3, 2)   # output: 1, 64,  73,  73
        # attention for 73          
        x = self.csatt73(x)         # output: 1, 64,  73,  73

        x = self.Conv2d_3b_1x1(x)   # output: 1, 80,  73,  73
        x = self.Conv2d_4a_3x3(x)   # output: 1, 192, 71,  71
        # pool 71->35
        x = F.max_pool2d(x, 3, 2)   # output: 1, 192, 35,  35
        # attention for 35          
        x = self.csatt35(x)         # output: 1, 192, 35,  35

        x = self.Mixed_5b(x)        # output: 1, 256, 35, 35
        x = self.Mixed_5c(x)        # output: 1, 288, 35, 35
        x = self.Mixed_5d(x)        # output: 1, 288, 35, 35
        # pool 35->17
        x = self.Mixed_6a(x)        # output: 1, 768, 17, 17
        # attention for 17          
        x = self.csatt17(x)         # output: 1, 768, 17, 17

        x = self.Mixed_6b(x)        # output: 1, 768, 17, 17
        x = self.Mixed_6c(x)        # output: 1, 768, 17, 17
        x = self.Mixed_6d(x)        # output: 1, 768, 17, 17
        x = self.Mixed_6e(x)        # output: 1, 768, 17, 17
        # transition
        x = self.transition(x)      # output: 1, 1024,17, 17        
        x = self.Mixed_7b(x)        # output: 1, 2048,17, 17
        x = self.Mixed_7c(x)        # output: 1, 2048,17, 17        
        
        # logits
        cams = self.fc(x)
        pred = F.adaptive_avg_pool2d(cams, output_size=(1,1)).squeeze()

        # return
        return pred, cams


    def init(self, params_path=None):
        # init weights by distribution
        for n, m in self.named_modules():
            if 'transition' in n:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.Tensor(X.rvs(m.weight.data.numel()))
                    values = values.view(m.weight.data.size())
                    m.weight.data.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()  
            if 'csatt' in n or 'fc' in n:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)                    
        print('log: init by xavier_normal_!')     
