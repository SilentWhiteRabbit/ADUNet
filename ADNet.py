import torch
import torch.nn as nn
import numpy as np

class Conv_BN_Relu_first(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_first,self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class Conv_BN_Relu_other(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bias):
        super(Conv_BN_Relu_other,self).__init__()
        kernel_size = 3
        padding = 1
        features = out_channels
        groups =1 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,groups,bais):
        super(Conv,self).__init__()
        kernel_size = 3
        padding = 1
        features = 1
        groups =1 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding,groups=groups, bias=False)
    def forward(self,x):
        return self.conv(x)

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, width,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        # print(proj_query.size())
        # print(proj_key.size())
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) 
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out, attention

class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features,out_channels=4,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x 
        x1 = self.conv1_1(x) 
        x1 = self.conv1_2(x1) #dilation
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1) 
        x1 = self.conv1_5(x1) #dilation
        x1 = self.conv1_6(x1) 
        x1 = self.conv1_7(x1)   
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t) #dilation
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1) #dilation
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1) # 64->4 
        out = torch.cat([x,x1],1) #4+4=8
        out= self.Tanh(out)
        out = self.conv3(out) # 8->4
        out = out*x1
        out2 = x - out
        return out2

if __name__ == "__main__":
    test_input = torch.from_numpy(np.random.randn(12, 4, 608, 608)).float()
    net = ADNet(4)

    output = net(test_input)
    print("test over")
