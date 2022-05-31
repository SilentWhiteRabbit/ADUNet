from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from baseline2 import Unet
from dataset2 import RawDataset, TestDataset
from common_tools import transform_invert, set_seed
from tensorboardX import SummaryWriter
# from utils.matric import dice_coef, iou_score
import skimage.metrics
from torch.autograd import Variable
import torch.optim
from matric import TV_Loss
import random



def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_image(input_data, height, width):
    batchsize = input_data.shape[0]
    output_data = np.zeros((batchsize,height, width),dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[:,channel_y:height:2, channel_x:width:2] = input_data[:, :, :, 2 * channel_y + channel_x]
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = (np.clip(input_data, 0., 1.)**2.5-1e-12) * (white_level - black_level) + black_level
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  # lower()将字符串中所有大写字符为小写。
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reconstruct_img(x):
        output_data =torch.zeros(1,4,1736,2312)
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
        return output_data

def train_model(args, model, criterion1,criterion2, optimizer, dataload, writer, tv):
    num_epochs = args.num_epochs
    black_level=1024
    white_level=16383

    makedir('./models_ADUnet_enhance_espcn')
    model_path = "/home/test/PLH/baseline/models_ADUnet_enhance_espcn/weights_50.pth"
    if os.path.exists(model_path) and args.use_trained_ckpt:
        model.load_state_dict(torch.load(model_path))
        start_epoch = 50
        print('加载成功！')
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    best_score = 0
    best_epoch = 0
    for epoch in range(start_epoch + 1, num_epochs):
        cur_lr = adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        mse_loss = 0
        tv_loss=0
        step = 0
        for x, y in dataload:
            step += 1
            rand = random.randrange(1,13)
            # print(rand)
            bs,c,h,w = x.shape
            output_data = torch.zeros(bs,c,608,608)
            for channel_x in range(3):
                for channel_y in range(4):
                    output_data_step = x[:, :, channel_x*564:channel_x*564+608, channel_y*568:channel_y*568+608]
                    output_data = torch.cat([output_data,output_data_step],dim=0)
            # x =output_data[bs*rand:bs*(rand+1),:,:,:]
            x =output_data[bs::2,:,:,:]
            # print(x.shape)
            # print(x.shape)

            output_data_y = torch.zeros(bs,c,608,608)
            for channel_x in range(3):
                for channel_y in range(4):
                    output_data_step_y =y[:, :, channel_x*564:channel_x*564+608, channel_y*568:channel_y*568+608]
                    output_data_y = torch.cat([output_data_y,output_data_step_y],dim=0)
            # y =output_data_y[bs*rand:bs*(rand+1),:,:,:]
            y =output_data_y[bs::2,:,:,:]
            # print(y.shape)

            inputs = x.cuda(device=device_ids[0])
            labels = y.cuda(device=device_ids[0])
            outputs = model(inputs)
            loss1 = criterion1(outputs[0], labels)
            loss2 = tv(outputs[1], labels)
            loss3 = criterion1(outputs[1], labels)
            loss = 0.6*loss1 + 0.4*loss3
            optimizer.zero_grad()
            loss.backward()
            # print(loss)
            # clip_gradient(optimizer, args.clip)
            optimizer.step()
            epoch_loss += loss.item()
            mse_loss+=loss1.item()
            tv_loss+=loss2.item()
            train_curve.append(loss.item())
            # print("%d/%d,train_loss:%0.6f,MSE_loss:%0.6f,TV_loss:%0.6f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(),loss1.item(),loss2.item()))
        epoch_avg_loss = np.mean(train_curve)
        writer.add_scalar('Train_loss', epoch_avg_loss.item(), epoch)
        print("epoch %d loss:%0.6f  MSE:%0.6f  TV:%0.6f  Lr:%0.6f"% (epoch, epoch_loss / step,mse_loss/step,tv_loss/step,cur_lr))
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), './models_ADUnet_enhance_espcn/weights_%d.pth' % (epoch + 1))

        # Validate the model
        valid_dataset = RawDataset("/data/PLH/dataset/Valset/", black_level=1024,white_level=16383)
        valid_loader = DataLoader(valid_dataset, batch_size=1 ,shuffle=True, num_workers=4)  # 这边bs取1，取其他数的话，我不想改
        if (epoch + 2) % val_interval == 0:
            psnr_val = np.zeros(shape=(len(valid_loader.dataset),1))
            ssim_val = np.zeros(shape=(len(valid_loader.dataset),1))
            model.eval()
            score = 0
            with torch.no_grad():
                step_val = 0

                for x, y in valid_loader:
                    step_val += 1
                    bs,c,h,w = x.shape
                    output_data = torch.zeros(bs,c,608,608)
                    for channel_x in range(3):
                        for channel_y in range(4):
                            output_data_step = x[:, :, channel_x*564:channel_x*564+608, channel_y*568:channel_y*568+608]
                            output_data = torch.cat([output_data,output_data_step],dim=0)
                    x =output_data[bs:,:,:,:]
                    # print(x.shape)

                    output_data_y = torch.zeros(bs,c,608,608)
                    for channel_x in range(3):
                        for channel_y in range(4):
                            output_data_step_y =y[:, :, channel_x*564:channel_x*564+608, channel_y*568:channel_y*568+608]
                            output_data_y = torch.cat([output_data_y,output_data_step_y],dim=0)
                    y =output_data_y[bs:,:,:,:]
                    # print(y.shape)

                    inputs = x.cuda(device=device_ids[0])
                    
                    outputs = model(inputs)
                    outputs =reconstruct_img(outputs[1])
                    result_data = outputs.cpu().numpy().transpose(0,2,3,1)
                    result_data = inv_normalization(result_data, black_level,white_level)
                    height=3472
                    width=4624
                    result_write_data = write_image(result_data, height, width)

                    labels = reconstruct_img(y).cpu().numpy().transpose(0,2,3,1)
                    labels = inv_normalization(labels, black_level,white_level)
                    labels = write_image(labels, height, width)
                    bs = labels.shape[0]
                    # print(labels.shape)
                    # print(result_write_data.shape)
                    psnr_bs =0.0
                    ssim_bs =0.0

                    for i in range(bs):
                        psnr = skimage.metrics.peak_signal_noise_ratio(
                            labels[i,:,:].astype(np.float), result_write_data[i,:,:].astype(np.float), data_range=white_level)
                        # print(psnr)
                        psnr_bs+=psnr
                        ssim = skimage.metrics.structural_similarity(
                            labels[i,:,:].astype(np.float), result_write_data[i,:,:].astype(np.float), multichannel=True, data_range=white_level)
                        ssim_bs+=ssim
                    psnr_val[step_val-1] = psnr_bs/bs
                    ssim_val[step_val-1] = ssim_bs/bs

                # w=0.8
                # psnr_min = min(psnr_val)
                # psnr_max = max(psnr_val)
                # ssim_min = min(ssim_val)
                # score = np.mean((w * (psnr_val - psnr_min)/(psnr_max - psnr_min) + (1 - w) * (ssim_val - ssim_min) / (1 - ssim_min)) * 100)
                # score = np.mean((psnr-30)*2.6667+(ssim-0.8)*100)
                score = (np.mean(psnr_val)-30)*2.6667+(np.mean(ssim_val)-0.8)*100
                writer.add_scalar('psnr_avg', np.mean(psnr_val), epoch)
                writer.add_scalar('ssim_avg', np.mean(ssim_val), epoch)
                writer.add_scalar('score_avg', (np.mean(psnr_val)-30)*2.6667+(np.mean(ssim_val)-0.8)*100, epoch)
                print("epoch %d psnr_avg:%0.3f ssim_avg: %0.3f  " % (epoch,np.mean(psnr_val), np.mean(ssim_val)))
                if score > best_score:
                    best_epoch = epoch
                    best_score = score
                    torch.save(model.state_dict(), './models_ADUnet_enhance_espcn/weights_bestscore.pth')
                print("epoch %d valid_score:%0.3f best_epoch: %d best_score: %0.3f  " % (epoch, score, best_epoch, best_score))

    return model


# 训练模型
def train(args):
    seed = args.random_seed
    set_seed(seed)
    writer = SummaryWriter(args.events)

    model = torch.nn.DataParallel(Unet(4,4), device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    # model = Unet(4,4).to(device)

    batch_size = args.batch_size
    criterion1 =  nn.MSELoss(reduction='mean')
    criterion2 =  nn.L1Loss(reduction='mean')
    tv = TV_Loss()
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    traindataset = RawDataset("/data/PLH/dataset/Trainset/", black_level=1024,white_level=16383)
    dataloaders = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=False)
    train_model(args, model, criterion1,criterion2, optimizer, dataloaders, writer, tv)


# 显示模型的输出结果
def test(args):
    model = nn.DataParallel(Unet(4,4), device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    black_level=1024
    white_level=16383
    height=868*4
    width=1156*4
    model.load_state_dict({k.replace('',''):v for k,v in torch.load(args.ckpt).items()})
    testdataset = TestDataset("/home/test/PLH/baseline/testset/", black_level,white_level)
    dataloaders = DataLoader(testdataset, batch_size=1,shuffle=True)


    model.eval()
    with torch.no_grad():
        for x, x_path,name in dataloaders:
            save_root = '/home/test/PLH/baseline/results/'
            bs,c,h,w = x.shape
            output_data = torch.zeros(bs,c,608,608)
            for channel_x in range(3):
                for channel_y in range(4):
                    output_data_step = x[:, :, channel_x*564:channel_x*564+608, channel_y*568:channel_y*568+608]
                    output_data = torch.cat([output_data,output_data_step],dim=0)
            x =output_data[bs:,:,:,:]

            x_path=x_path[0]
            name =name[0]
            x = x.cuda(device=device_ids[0])
            y = model(x)
            y =reconstruct_img(y[1])
            result_data = y.cpu().detach().numpy().transpose(0, 2, 3, 1)
            result_data = inv_normalization(result_data, black_level, white_level)
            result_write_data = write_image(result_data, height, width).squeeze(0)
            save_root = os.path.join(save_root,'denoise' + name[-5:-4]+'.dng')
            write_back_dng(x_path, save_root, result_write_data)




if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train, test", default="test")
    parse.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parse.add_argument('--decay_rate', type=float, default=0.4, help='decay rate of learning rate')
    parse.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parse.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default="/home/test/PLH/baseline/models_ADUnet_enhance_espcn/weights_bestscore.pth")
    parse.add_argument("--events", type=str, help="the path of model loss logger", default="./models_ADUnet_enhance_espcn/logger")
    parse.add_argument("--random_seed", type=int, default=1234)
    parse.add_argument("--use_trained_ckpt", type=str2bool, default=False)
    parse.add_argument("--num_epochs", type=int, default=303)
    # parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    val_interval = 1
    # 是否使用cuda
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # 均为灰度图像，只需要转换为tensor
    # transforms.ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]


    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    print('USE GPU 1,2')
    # build the model
    device_ids = [0, 1]

    train_curve = list()
    valid_curve = list()
    torch.backends.cudnn.benchmark = True
    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
    else:
        print("Please choose train or test") 