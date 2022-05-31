from os import X_OK
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)#torch.Size([2, 2048, 128])
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)#torch.Size([2, 128, 2048])
        energy = torch.bmm(proj_query, proj_key)#(2,2048,2048)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)#torch.Size([2, 2048, 2048])

        proj_value = x.view(m_batchsize, C, -1)#(2,2048,128)
        out = torch.bmm(attention, proj_value)#torch.Size([2, 2048, 128])
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x

class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 4 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x



class Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        features = 64
        kernel_size=3
        padding = 1
        groups = 1
        self.pconv1_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))

        self.pconv1_7 = nn.Sequential(nn.Conv2d(in_channels=2*features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_9 = CAM_Module(features)
       
        self.pconv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.pconv1_16 = nn.Conv2d(in_channels=features,out_channels=32,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.pconv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()


        self.conv1_1 = BasicConv2d(in_channels, 16,3,1,1)
        self.conv1_2 = BasicConv2d(16, 32,3,1,1)
        self.conv1_3 = BasicConv2d(32, 32,3,1,1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = BasicConv2d(32, 64,3,1,1)
        self.conv2_2 = BasicConv2d(64, 64,3,1,1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = BasicConv2d(64, 128,3,1,1)
        self.conv3_2 = BasicConv2d(128, 128,3,1,1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = BasicConv2d(128, 256,3,1,1)
        self.conv4_2 = BasicConv2d(256, 256,3,1,1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = BasicConv2d(256, 512,3,1,1)
        self.conv5_2 = BasicConv2d(512, 512,3,1,1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv6_1 = BasicConv2d(512, 256,3,1,1)
        self.conv6_2 = BasicConv2d(256, 256,3,1,1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv7_1 = BasicConv2d(256, 128,3,1,1)
        self.conv7_2 = BasicConv2d(128, 128,3,1,1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv8_1 = BasicConv2d(128, 64,3,1,1)
        self.conv8_2 = BasicConv2d(64, 64,3,1,1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv9_1 = BasicConv2d(64,32,3,1,1)
        self.conv9_2 = BasicConv2d(32, 32,3,1,1)
        self.conv9_3 = BasicConv2d(32, 32,3,1,1)

        self.conv10_1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)

        self.espcn = ESPCN(upscale_factor=2)

    def forward(self, x):
        
        m = F.interpolate(x, size=(304, 304), mode='bilinear', align_corners=True)

        m1 = self.pconv1_1(m) 
        m1 = self.pconv1_2(m1) #dilation
        m1 = self.pconv1_3(m1)
        m1 = self.pconv1_4(m1) 
        m1 = self.pconv1_5(m1) #dilation
        m1 = self.pconv1_6(m1) 

        conv1 =  self.conv1_1(x)
        conv1 =  self.conv1_2(conv1)
        conv1 =  self.conv1_3(conv1)
        pool1 = self.pool1(conv1)

        conv2 =  self.conv2_1(pool1)
        conv2 =  self.conv2_2(conv2)
        pool2 = self.pool1(conv2)

        conv3 =  self.conv3_1(pool2)
        conv3 =  self.conv3_2(conv3)

        pool2_2 = F.interpolate(pool2, size=(304, 304), mode='bilinear', align_corners=True)
        m1 = torch.cat([m1,pool2_2],1)
        m1 = self.pconv1_7(m1)   
        m1 = self.pconv1_8(m1)
        m1 = self.pconv1_9(m1) #dilation
        m1 = self.pconv1_10(m1)
        m1 = self.pconv1_11(m1)
        m1 = self.pconv1_16(m1) # 64->4 

        out = torch.cat([pool1,m1],1) #4+4=8
        out= self.Tanh(out)
        out = self.pconv3(out) # 8->4
        out = out*m1
        out2 = pool1 - out
        out2 = F.interpolate(out2, size=(608, 608), mode='bilinear', align_corners=True)


        pool3 = self.pool1(conv3)

        conv4 =  self.conv4_1(pool3)
        conv4 =  self.conv4_2(conv4)
        pool4 = self.pool1(conv4)

        conv5 =  self.conv5_1(pool4)
        conv5 =  self.conv5_2(conv5)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 =  self.conv6_1(up6)
        conv6 =  self.conv6_2(conv6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 =  self.conv7_1(up7)
        conv7 =  self.conv7_2(conv7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 =  self.conv8_1(up8)
        conv8 =  self.conv8_2(conv8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, out2], 1)
        conv9 =  self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)
        conv9 = self.conv9_3(conv9)

        conv10 = self.conv10_1(conv9)
        
        out3 = self.espcn(conv10)
        out3 = F.interpolate(out3, size=(608, 608), mode='bilinear', align_corners=True)

        return conv10,out3

    def leaky_relu(self, x):
        out = torch.max(0.1 * x, x)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    test_input = torch.from_numpy(np.random.randn(6, 4, 608, 608)).float()
    
    net = Unet(4,4)
    conv10,out3 = net(test_input)
    print(out3.shape)
