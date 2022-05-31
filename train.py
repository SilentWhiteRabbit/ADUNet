import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import skimage.metrics
from baseline import Unet
import argparse
from dataset import RAW_Dataset, write_image, inv_normalization, read_image
from torch.utils.data import DataLoader
import time



def choice_device():
    # GPU or CPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    return device


def trains(train_noise_path, train_gt_path, val_noise_path, val_gt_path, black_level, white_level, epoch, batch_size, learn_rate, test0_path, save_model_path):

    # set some parameter
    device = choice_device()
    print("the device is: ", device)
    best_psnr = 0.0  # to save the best model
    _, height, width = read_image(test0_path)

    # dataset and dataloder
    # train data
    train_set = RAW_Dataset(train_gt_path, train_noise_path, black_level, white_level)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    train_n = len(train_loader)
    # val data
    val_set = RAW_Dataset(val_gt_path, val_noise_path, black_level, white_level)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    val_n = len(val_loader)

    print("Finished dataloader!")

    # define:net loss_function optimizer
    # net
    net = Unet()
    net.to(device)
    # loss_function
    loss_function = nn.MSELoss(reduction='sum').to(device)  # MSE
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.4)  # adjust the learn_rate

    # train and val
    for i in range(epoch):
        print("Now, the epoch is: ", i)
        start_time = time.time()
        # train
        net.train()
        for step1, data in enumerate(train_loader, start=0):
            train_noise, train_gt = data
            optimizer.zero_grad()
            train_result = net(train_noise.to(device), device)
            loss = loss_function(train_result, train_gt.to(device)).to(device)
            loss.backward()
            optimizer.step()

            # show the spead of train
            print("\r", "The progress of train:", "▋" * step1, "{}/{}".format((step1+1), train_n), end="", flush=True)
            time.sleep(0.1)

        print("\n", "Finished training of this epoch.")
        # val
        net.eval()
        n = len(val_loader)
        psnr_ave = 0.0
        ssim_ave = 0.0
        with torch.no_grad():
            for step2, data in enumerate(val_loader, start=0):
                val_noise, val_gt = data
                val_result = net(val_noise.to(device), device)

                # val_result inv
                result_data = val_result.cpu().detach().numpy().transpose(0, 2, 3, 1)
                result_data = inv_normalization(result_data, black_level, white_level)
                result_write_data = write_image(result_data, height, width)
                # val_gt inv
                gt_data = val_gt.cpu().detach().numpy().transpose(0, 2, 3, 1)
                gt_data = inv_normalization(gt_data, black_level, white_level)
                gt_write_data = write_image(gt_data, height, width)

                # calculate the index
                psnr = skimage.metrics.peak_signal_noise_ratio(
                    gt_write_data.astype(float), result_write_data.astype(float), data_range=white_level)
                ssim = skimage.metrics.structural_similarity(
                    gt_write_data.astype(float), result_write_data.astype(float), channel_axis=True, data_range=white_level)
                psnr_ave += psnr / n
                ssim_ave += ssim / n
                
                # show the spead of val
                print("\r", "The progress of val:", "▋" * step2, "{}/{}".format((step2+1), val_n), end="", flush=True)
                time.sleep(0.1)
        print("\n", "Finished val of this epoch.")
        end_time=time.time()
        run_time = end_time - start_time
        print('in this epoch ,the psnr_ave is:%.3f  in this epoch ,the ssim_ave is：%.3f' % (psnr_ave, ssim_ave))
        print('The time it takes for this epoch is:', run_time,'s')

        # save the best model
        if psnr_ave > best_psnr:
            best_psnr = psnr_ave
            torch.save(net.state_dict(), save_model_path)

        scheduler.step()

    print('Finished training of all epoch.')


def main(args):
    train_noise_path = args.train_noise_path
    train_gt_path = args.train_gt_path
    val_noise_path = args.val_noise_path
    val_gt_path = args.val_gt_path
    black_level = args.black_level
    white_level = args.white_level
    epoch = args.epoch
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    test0_path = args.test0_path
    save_model_path = args.save_model_path

    trains(train_noise_path, train_gt_path, val_noise_path, val_gt_path, black_level, white_level, epoch, batch_size, learn_rate, test0_path, save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_noise_path', type=str, default="../data/train/noise/")
    parser.add_argument('--train_gt_path', type=str, default="../data/train/gt/")
    parser.add_argument('--val_noise_path', type=str, default="../data/val/noise/")
    parser.add_argument('--val_gt_path', type=str, default="../data/val/gt/")
    parser.add_argument('--test0_path', type=str, default="../data/test/noisy0.dng")
    parser.add_argument('--save_model_path', type=str, default="../data/result/algorithm/models/model5.pth")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learn_rate', type=int, default=0.0002)

    args = parser.parse_args()
    main(args)
