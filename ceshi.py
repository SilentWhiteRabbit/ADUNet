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

def write_image(input_data, height, width):
    batchsize = input_data.shape[0]
    output_data = np.zeros((batchsize,height, width),dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[:,channel_y:height:2, channel_x:width:2] = input_data[:, :, :, 2 * channel_y + channel_x]
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8


    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()
    # print(dest_path)
    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - input_data.min()) / (white_level - black_level)
    # print(output_data)
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
    valid_dataset = TestDataset("/home/test/PLH/baseline/testset", black_level=1024,white_level=16383)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for x,path,name in valid_loader:
            path=path[0]
            name =name[0]
            save_root = '/home/test/PLH/baseline/results/'
            # x = 1-x
            # print(x)
            print(x)
            x = x**0.4+1e-8
            print(x)
            y = (x -1e-8)**2.5
            print(y)
            bs,c,h,w = x.shape
            result_data = y.cpu().detach().numpy().transpose(0, 2, 3, 1)
            # print(result_data)
            result_data = inv_normalization(result_data, 1024, 16383)
            result_write_data = write_image(result_data, h*2, w*2).squeeze(0)
            # print(result_write_data)
            # break
            save_root = os.path.join(save_root,'fanse' + name[-5:-4]+'.dng')
            write_back_dng(path, save_root, result_write_data)

