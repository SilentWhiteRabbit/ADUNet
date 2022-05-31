# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : Peter
# @date       : 2020-02-03 14:10:00
# @brief      : 通用函数
"""

import numpy as np
import torch
import random
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        # img_ = np.array(img_) * 255
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


def set_seed(seed):
    """
    进行随机种子的设置
    :param seed: 种子数
    :return: 无
    """
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def rand_crop(data, label, img_w, img_h):
    width1 = random.randint(0, data.size[0] - img_w)
    height1 = random.randint(0, data.size[1] - img_h)
    width2 = width1 + img_w
    height2 = height1 + img_h

    data = data.crop((width1, height1, width2, height2))
    label = label.crop((width1, height1, width2, height2))

    return data, label


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def IOU(box1, box2):
    """
    iou loss
    :param box1: tensor [batch, w, h, num_anchor, 4], xywh 预测值
    :param box2: tensor [batch, w, h, num_anchor, 4], xywh 真实值
    :return: tensor [batch, w, h, num_anchor, 1]
    """
    box1_xy, box1_wh = box1[..., :2], box1[..., 2:4]
    box1_wh_half = box1_wh / 2.
    box1_mines = box1_xy - box1_wh_half
    box1_maxes = box1_xy + box1_wh_half

    box2_xy, box2_wh = box2[..., :2], box2[..., 2:4]
    box2_wh_half = box2_wh / 2.
    box2_mines = box2_xy - box2_wh_half
    box2_maxes = box2_xy + box2_wh_half

    # 求真实值和预测值所有的iou
    intersect_mines = torch.max(box1_mines, box2_mines)
    intersect_maxes = torch.min(box1_maxes, box2_maxes)
    intersect_wh = torch.max(intersect_maxes-intersect_mines, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0]*intersect_wh[..., 1]
    box1_area = box1_wh[..., 0]*box1_wh[..., 1]
    box2_area = box2_wh[..., 0]*box2_wh[..., 1]
    union_area = box1_area+box2_area-intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)
    return iou





