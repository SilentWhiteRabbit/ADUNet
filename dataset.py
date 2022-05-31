from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import rawpy
from matplotlib import pyplot as plt
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data

def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:4, 0:width:4, :],
                                        raw_data_expand[0:height:4, 1:width:4, :],
                                        raw_data_expand[0:height:4, 2:width:4, :],
                                        raw_data_expand[0:height:4, 3:width:4, :],
                                        raw_data_expand[1:height:4, 0:width:4, :],
                                        raw_data_expand[1:height:4, 1:width:4, :],
                                        raw_data_expand[1:height:4, 2:width:4, :],
                                        raw_data_expand[1:height:4, 3:width:4, :],
                                        raw_data_expand[2:height:4, 0:width:4, :],
                                        raw_data_expand[2:height:4, 1:width:4, :],
                                        raw_data_expand[2:height:4, 2:width:4, :],
                                        raw_data_expand[2:height:4, 3:width:4, :],
                                        raw_data_expand[3:height:4, 0:width:4, :],
                                        raw_data_expand[3:height:4, 1:width:4, :],
                                        raw_data_expand[3:height:4, 2:width:4, :],
                                        raw_data_expand[3:height:4, 3:width:4, :]), axis=2)
    return raw_data_expand_c, height, width

def make_dataset(root):
    # root = "./data/train"
    imgs = []
    ori_path = os.path.join(root, "noisy")
    ground_path = os.path.join(root, "gt")
    names_noi = sorted(os.listdir(ori_path))
    names_gt = sorted(os.listdir(ground_path))
    n = len(names_noi)
    for i in range(n):
        if names_noi[i].split('.')[1]=='dng':
            img = os.path.join(ori_path, names_noi[i])
            gt = os.path.join(ground_path, names_gt[i])
            imgs.append((img, gt))
        else:
            continue
    return imgs

def make_dataset_test(root):
    # root = "./data/train"
    imgs = []
    names_noi = sorted(os.listdir(root))
    n = len(names_noi)
    for i in range(n):
        if names_noi[i].split('.')[1]=='dng':
            img = os.path.join(root, names_noi[i])
            imgs.append(img)
        else:
            continue
    return imgs

class RawDataset(Dataset):
    def __init__(self, root,black_level, white_level):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.black_level = black_level
        self.white_level = white_level

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]

        gt, height, width = read_image(y_path)
        gt_normal = normalization(gt,self.black_level, self.white_level)
        raw_data_expand_c_normal_y = torch.from_numpy(np.transpose(
            gt_normal.reshape(height//4, width//4, 16), (2,0,1))).float()

        raw_data_expand_c, height, width = read_image(x_path)
        img_norm = normalization(raw_data_expand_c,self.black_level, self.white_level)
        raw_data_expand_c_normal_x = torch.from_numpy(np.transpose(
            img_norm.reshape( height//4, width//4, 16), (2,0,1))).float()


        return raw_data_expand_c_normal_x, raw_data_expand_c_normal_y 
    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, root,black_level, white_level):
        imgs = make_dataset_test(root)
        self.imgs = imgs
        self.black_level = black_level
        self.white_level = white_level

    def __getitem__(self, index):
        x_path = self.imgs[index]
        name = x_path.split('/')[-1]

        raw_data_expand_c, height, width = read_image(x_path)
        # print(raw_data_expand_c.shape)
        raw_data_expand_c_normal_x = normalization(raw_data_expand_c,self.black_level, self.white_level)
        # print(raw_data_expand_c_normal_x.shape)
        raw_data_expand_c_normal_x = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal_x.reshape(height//4, width//4, 16), (2, 0, 1))).float()


        return raw_data_expand_c_normal_x, x_path, name
    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    valid_dataset = RawDataset("/data/PLH/dataset/Trainset/", black_level=1024,white_level=16383)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True)

    with torch.no_grad():
        for x,y,x1,x2 in valid_loader:
            print(x.shape)
            print(y.shape)
            print(x1)
            print(x2)
            break

    # valid_dataset = TestDataset("/home/test/PLH/baseline/testset/", black_level=1024,white_level=16383)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    # with torch.no_grad():
    #     for x,x_path in valid_loader:
    #         print(x.shape)

    #         break