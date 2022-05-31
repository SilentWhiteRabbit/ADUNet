import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class dilation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(dilation, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class LeakyReluBottleneck(models.resnet.Bottleneck):  # 直接继承Bottleneck类，不用自己去一个个添加了
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):  # 在bottleneck中无下采样
        super(LeakyReluBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                                  norm_layer)
        self.relu = nn.LeakyReLU(inplace=True)


class LeakyReluBasicBlock(models.resnet.BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(LeakyReluBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation,
                                                  norm_layer)
        self.relu = nn.LeakyReLU(inplace=True)

class ResNetWithoutPool(models.ResNet):
    def __init__(self, block, layers):
        super(ResNetWithoutPool, self).__init__(block, layers)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(4, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 8, layers[0], stride=2)  # layers：每个block的个数，如resnet50， layers=[3,4,6,3]
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)  # 128是当前块的输入通道数
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)  # stride=1or2判断是否需要下采样，将通道数由64变为128
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 此处没有self.avgpool = nn.AdaptiveAvgPool2d((1, 1))，说的就是上面的ResNetWithoutPool
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def build_backbone(num_layers, pretrained=False):
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: LeakyReluBasicBlock, 50: LeakyReluBottleneck}[num_layers]
    model = ResNetWithoutPool(block_type, blocks)
    return model


class Encoder(nn.Module):
    """
    Resnet without maxpool
    """

    def __init__(self, num_layers=50, pre_trained=True):
        super(Encoder, self).__init__()
        # make backbone
        backbone = build_backbone(num_layers, pre_trained)
        # blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        ])
        self.num_ch_enc = np.array([64, 64, 128, 256])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        # from shallow to deep
        features = [x]
        # features = [x]
        for block in self.blocks:
            features.append(block(features[-1]))
            
        return features[0:]

class PNet(nn.Module):
    """
    Resnet without maxpool
    """

    def __init__(self):
        super(PNet, self).__init__()
        self.DepthEncoder = Encoder(50, pre_trained=False)
        self.conv1_1 = BasicConv2d(4,8,2,2,0,1)
        self.conv1_cut = dilation(16,8)
        self.conv1_2 = BasicConv2d(16,16,2,2,0,1)
        self.conv2_cut= dilation(32,16)
        self.conv1_4 = BasicConv2d(32,32,2,2,0,1)
        self.conv3_cut= dilation(64,32)
        self.conv1_5= BasicConv2d(64,64,2,2,0,1)
        self.conv4_cut= dilation(128,64)
        self.conv1_6= BasicConv2d(128,128,3,1,1,1)
        self.conv1_de= nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_7= BasicConv2d(64,64,3,1,1,1)
        self.conv2_de= nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.conv1_8= BasicConv2d(32,32,3,1,1,1)
        self.conv3_de= nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2),nn.BatchNorm2d(16),nn.ReLU(inplace=True))
        self.conv1_9= BasicConv2d(16,16,3,1,1,1)
        self.conv4_de= nn.Sequential(nn.ConvTranspose2d(16, 8, 2, stride=2),nn.BatchNorm2d(8),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Conv2d(8, 4, kernel_size=3, stride=1,padding=1,groups=1,bias=False)
        self.conv3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()



    def forward(self,x):
        p1,p2,p3,p4,p5 = self.DepthEncoder(x)
        x1 = self.conv1_1(p1) # 8,304,304
        p2_cut = self.conv1_cut(p2) # 8,304,304
        x2 = torch.cat([p2_cut,x1],1) # 16,304,304
        x2 = self.conv1_2(x2) # 16,152,152

        p3_cut = self.conv2_cut(p3) # 16,152,152
        x2 = torch.cat([p3_cut,x2],1) # 32,152,152
        x2 = self.conv1_4(x2) # 32,76,76

        p4_cut = self.conv3_cut(p4) # 32,76,76 
        x3 =torch.cat([p4_cut,x2],1) #64,76,76
        x3 =self.conv1_5(x3) # 64,38,38

        p5_cut = self.conv4_cut(p5) # 64,38,38
        x4 =torch.cat([p5_cut,x3],1) #128,38,38

        y4 = self.conv1_6(x4) # 128,38,38
        y4 = self.conv1_de(y4) # 64,76,76
        y4 = 0.2*p4+0.8*y4 # 64,76,76
        y5 = self.conv1_7(y4) # 64,76,76
        y5 = self.conv2_de(y5) # 32,152,152
        y5 = 0.3*p3+0.7*y5 # 32,152,152
        y6 = self.conv1_8(y5) # 32,152,152
        y6 = self.conv3_de(y6) # 16,304,304

        y6 = 0.4*p2+0.6*y6 # 16,304,304 
        y7 = self.conv1_9(y6) # 16,304,304

        y7 = self.conv4_de(y7) # 8,608,608
        y8 = self.conv1_10(y7) # 4,608,608

        out = torch.cat([p1,y8],1) #4+4=8
        out= self.Tanh(out)
        out = self.conv3(out) # 8->4
        out = out*y8
        out2 = p1 - out
        return out2




if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    test_input = torch.from_numpy(np.random.randn(4, 4, 608, 608)).float().to(device)
    
    net = PNet().to(device)
    output = net(test_input)
    print(output.shape)

