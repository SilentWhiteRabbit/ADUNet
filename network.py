import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import numpy as np
import cv2


# if __name__ == '__main__':
#     eps = 0.01
#     winSize = (5,5)
#     image = cv2.imread(r'./5921.png', cv2.IMREAD_ANYCOLOR)
#     image = cv2.resize(image, None,fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
#     I = image/255.0        #将图像归一化
#     p =I
#     guideFilter_img = guideFilter(I, p, winSize, eps)

#     # 保存导向滤波结果
#     guideFilter_img  = guideFilter_img  * 255
#     guideFilter_img [guideFilter_img  > 255] = 255
#     guideFilter_img  = np.round(guideFilter_img )
#     guideFilter_img  = guideFilter_img.astype(np.uint8)
#     cv2.imshow("image",image)
#     cv2.imshow("winSize_5", guideFilter_img )
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


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

class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = BasicConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = BasicConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = BasicConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = BasicConv2d(512, 256, kernel_size=3, stride=1, padding=1)
   

        # self.upv6 = BasicDeConv2d(512, 256, 2, stride=2)

        self.conv6_1 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = BasicConv2d(256, 128, kernel_size=3, stride=1, padding=1)
     

        # self.upv7 = BasicDeConv2d(256, 128, 2, stride=2)

        self.conv7_1 = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
    

        # self.upv8 = BasicDeConv2d(128, 64, 2, stride=2)

        self.conv8_1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
  

        # self.upv9 = BasicDeConv2d(64, 32, 2, stride=2)

        self.conv9_1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = BasicConv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv10_1 = nn.Conv2d(16, out_channels, kernel_size=1, stride=1)
        self.gf1 = FastGuidedFilter(2,1e-2)
        self.gf2 = FastGuidedFilter(2,1e-2)
        self.gf3 = FastGuidedFilter(3,1e-2)
        self.gf4 = FastGuidedFilter(3,1e-2)
        self.attanction1 = cbam_block(32)
        self.attanction2 = cbam_block(64)
        self.attanction3 = cbam_block(128)
        self.attanction4 = cbam_block(256)

    def forward(self, x):
        n, c, h, w = x.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        conv11 = self.conv1_1(padded_image)
        conv1 = self.conv1_2(conv11)
        atten1 =self.attanction1(conv1)
        pool1 = self.pool1(conv1)

        conv22 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv22)
        atten2 =self.attanction2(conv2)
        pool2 = self.pool1(conv2)

        conv33 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv33)
        atten3 =self.attanction3(conv3)
        pool3 = self.pool1(conv3)

        conv44 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv44)
        atten4 =self.attanction4(conv4)
        pool4 = self.pool1(conv4)

        conv55 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv55)
        N, C, H, W = conv4.size()

        down6 = F.interpolate(atten4, size=(int(H/2),int(W/2)), mode='bilinear',align_corners=True)
        up6 = self.gf1(conv5,down6,atten4)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)

        N, C, H, W = conv3.size()
        down7 = F.interpolate(atten3, size=(int(H/2),int(W/2)), mode='bilinear',align_corners=True)
        up7 = self.gf2(conv6,down7,atten3)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)


        N, C, H, W = conv2.size()
        down8 = F.interpolate(atten2, size=(int(H/2),int(W/2)), mode='bilinear',align_corners=True)
        up8 = self.gf3(conv7,down8,atten2)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)


        N, C, H, W = conv1.size()
        down9 = F.interpolate(atten1, size=(int(H/2),int(W/2)), mode='bilinear',align_corners=True)
        up9 = self.gf4(conv8,down9,atten1)
        conv9 = self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)

        conv10 = self.conv10_1(conv9)

        out = conv10[:, :, :h, :w]

        return out

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        mean_x = self.boxfilter(lr_x) / N
        mean_y = self.boxfilter(lr_y) / N
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A*hr_x+mean_b

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


if __name__ == "__main__":
    test_input = torch.from_numpy(np.random.randn(1, 16, 868,1156)).float()
    net = Unet(16,16)
    output = net(test_input)
    print("test over")
