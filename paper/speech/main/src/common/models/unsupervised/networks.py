import torch
from torch import nn
import sys
import math
sys.path.append("../../../../src")
from common.models.unsupervised import BasicBlockNormal, selfattention


class Generator(torch.nn.Module):
    def __init__(self,d_model):
        super(Generator, self).__init__()
        self.d_model = d_model
        
        self.fc = nn.Linear(d_model,512)
        self.d_model = 512

        up = nn.Upsample(scale_factor=2, mode='bilinear')

        dconv1 = nn.Conv2d(self.d_model, self.d_model//2, 3, 1, 1) # 2*2 512
        dconv2 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 4*4 256
        dconv3 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 16*16 256
        dconv4 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 32 * 32 * 256
        dconv5 = nn.Conv2d(self.d_model//2, self.d_model//4, 3, 1, 1) #  64 * 64 *128
        #dconv6 = nn.Conv2d(self.d_model//4, self.d_model//8, 3, 1, 1) # 128 * 128 *32
        dconv7 = nn.Conv2d(self.d_model//4, 3, 3, 1, 1)

        # batch_norm2_1 = nn.BatchNorm2d(self.d_model//8)
        batch_norm4_1 = nn.BatchNorm2d(self.d_model//4)
        batch_norm8_4 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_5 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_6 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_7 = nn.BatchNorm2d(self.d_model//2)

        relu = nn.ReLU()
        tanh = nn.Tanh()

        self.model = torch.nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                             relu, up, dconv2, batch_norm8_5, relu,
                             up, dconv3, batch_norm8_6, relu, up, dconv4,
                             batch_norm8_7, relu, up, dconv5, batch_norm4_1,
                             relu, up, dconv7, tanh)

    def forward(self,x):
        x = self.fc(x)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        out = self.model(x)
        return out

class FaceCycleBackbone(nn.Module):
    def __init__(self):
        super(FaceCycleBackbone, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    selfattention(64),
                                    nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1))  # 64

        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      selfattention(128),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.resblock1 = BasicBlockNormal(128, 128)
        self.resblock2 = BasicBlockNormal(128, 128)

        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_2_exp = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      )  # 64

        self.layer3_2_pose = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                          nn.LeakyReLU(negative_slope=0.1))  # 64


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #encoder
        '''

        :param x: [batch,3,64,64]
        :return:
        '''
        out_1 = self.conv1(x) # [batch,64,32,32]
        out_1 = self.layer1(out_1) # [batch,64,32,32]
        out_2 = self.layer2_1(out_1) # [batch,128,16,16]
        out_2 = self.resblock1(out_2) # [batch,128,16,16]
        out_2 = self.resblock2(out_2) # [batch,128,16,16]
        out_2 = self.layer2_2(out_2) # [batch,128,8,8]
        out_3 = self.layer3_1(out_2) # [batch,256,4,4]

        out_3_exp = self.layer3_2_exp(out_3) # [batch,128,4,4]
        out_3_exp = out_3_exp.view(x.size()[0],-1) # [batch,2048]

        out_3_pose = self.layer3_2_pose(out_3)
        out_3_pose = out_3_pose.view(x.size()[0],-1)

        out_3 = out_3.view(x.size()[0],-1)
        return out_3, out_3_exp,out_3_pose