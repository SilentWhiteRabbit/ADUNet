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
    output_data = (input_data.astype(float) - input_data.min()) / (white_level - black_level)
    output_data = (output_data+1e-12)**0.4
    return output_data

def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
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
            gt_normal.reshape(height//2, width//2, 4), (2,0,1))).float()

        raw_data_expand_c, height, width = read_image(x_path)
        raw_data_expand_c_normal_x = normalization(raw_data_expand_c,self.black_level, self.white_level)
        raw_data_expand_c_normal_x = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal_x.reshape( height//2, width//2, 4), (2,0,1))).float()

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
            raw_data_expand_c_normal_x.reshape(height//2, width//2, 4), (2, 0, 1))).float()


        return raw_data_expand_c_normal_x, x_path, name
    def __len__(self):
        return len(self.imgs)

if __name__=="__main__":
    valid_dataset = RawDataset("/data/PLH/dataset/Trainset/", black_level=1024,white_level=16383)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x,y in valid_loader:
            bs,c,h,w = x.shape
            output_data = torch.zeros(bs,c,608,608)
            for channel_x in range(3):
                for channel_y in range(4):
                    output_data_step = x[:, :, channel_x*564:channel_x*564+608, channel_y*568:channel_y*568+608]
                    output_data = torch.cat([output_data,output_data_step],dim=0)
            x =output_data[bs:,:,:,:]
            b,cs,hs,ws=x.shape
            output_data =torch.zeros(1,4,h,w)

            x0 = x[0,:,0:586,0:588]
            x1 = x[1,:,0:586,20:588]
            x2 = x[2,:,0:586,20:588]
            x3 = x[3,:,0:586,20:608]
            x4 = x[4,:,22:586,0:588]
            x5 = x[5,:,22:586,20:588]
            x6 = x[6,:,22:586,20:588]
            x7 = x[7,:,22:586,20:608]
            x8 = x[8,:,22:608,0:588]
            x9 = x[9,:,22:608,20:588]
            x10 = x[10,:,22:608,20:588]
            x11 = x[11,:,22:608,20:608]
            output_data[0,:,0:586,0:588] = x0
            output_data[0,:,0:586, 588:1156] = x1
            output_data[0,:,0:586,1156:1724] = x2
            output_data[0,:,0:586,1724:2312] = x3
            output_data[0,:,586:1150,0:588] = x4
            output_data[0,:,586:1150,588:1156] = x5
            output_data[0,:,586:1150,1156:1724] = x6
            output_data[0,:,586:1150,1724:2312] = x7
            output_data[0,:,1150:1736,0:588] = x8
            output_data[0,:,1150:1736,588:1156] = x9
            output_data[0,:,1150:1736,1156:1724] = x10
            output_data[0,:,1150:1736,1724:2312] = x11
            print(output_data)

            
            
            print(x.shape)
            break
