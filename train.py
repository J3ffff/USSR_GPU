import numpy as np
from net import *
from data_process import DataSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from PIL import Image
import sys
from torchvision import transforms
from torchvision import models
import tqdm
import argparse
import csv
from math import log10
from torchsummary import summary

# python train.py --num_batches 10 --img G0021_4.JPG

def PSNR(img_1,img_2):
    img_1 = 255 * img_1
    img_2 = 255 * img_2
    diff = np.abs(img_1 - img_2)
    rmse = np.sqrt(diff).sum()
    psnr = 20*np.log10(255/rmse)
    return psnr

def Data_Save(epoch,train_loss,train_dice):
    data = [epoch,float(train_loss),float(train_dice)]
    result_data_dir = "./Results/Result3.csv"
    # result_data_dir = "./result/data/pixel/vgg_vgg_pixel.csv"
    # result_data_dir = "./result/data/combine/vgg_vgg_combine.csv"
    writer = csv.writer(open(result_data_dir, 'a+')) #a+ do not overwrite origenal data,'wb',overwrite origenal data
    writer.writerow(data)


def MSE(img_1, img_2):
    mse = nn.MSELoss()
    out = mse(img_1, img_2)
    return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(model, img, sr_factor, num_batches, learning_rate, crop_size):
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sampler = DataSampler(img, sr_factor, crop_size)
    if args.cuda:
        model.cuda()
    with tqdm.tqdm(total=num_batches, miniters=1, mininterval=0) as progress:
        for iter, (hr, lr) in enumerate(sampler.generate_data()):
            model.zero_grad()
            if args.cuda:
                hr = hr.cuda()
                lr = lr.cuda()
            lr = Variable(lr)
            hr = Variable(hr)
            if args.residual:
                output = model(lr) + lr
            else:
                output = model(lr)
            error = loss(output, hr)
            # psnr = PSNR(output.data.cpu().numpy(), hr.data.cpu().numpy())
            mse = MSE(output, hr)
            psnr = 10 * log10(1 / mse.data[0])

            cpu_loss = error.data.cpu().numpy()[0]
            progress.set_description("Iteration: {iter} Loss: {loss}, PSNR: {PSNR}, Learning Rate: {lr}".format(
                iter=iter, loss=cpu_loss, PSNR = psnr, lr=learning_rate))
            progress.update()
            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr=learning_rate))

            mse.backward()
            optimizer.step()
            if iter % 10 ==0:
                Data_Save(iter, float(error.data.cpu().numpy()), psnr)

            if iter == num_batches:
                print('Done training.')
                break
            

def test(model, img, sr_factor):
    print('--------------------------------------')
    print('Testing!')
    model.eval()

    img = img.resize((int(img.size[0]*sr_factor),
                      int(img.size[1]*sr_factor)),
                     resample=Image.BICUBIC)
    # img.save(img_name +'_LR.png')

    img = transforms.ToTensor()(img)[0,:,:]
    img = torch.unsqueeze(img, 0)
    img = torch.unsqueeze(img, 0)
    if args.cuda:
        img = img.cuda()
    input = Variable(img)

    if args.residual:
        residual = model(input)
        output = input + residual

        # residual_numpy = residual.cpu().data.numpy()[0, 0, :, :]
        # residual_numpy = np.stack((residual_numpy, residual_numpy, residual_numpy), 0)
        # residual_numpy[np.where(residual_numpy < 0)] = 0.0
        # residual_numpy[np.where(residual_numpy > 1)] = 1.0
        # residual_output = torch.from_numpy(10 * residual_numpy)
        # residual_output = transforms.ToPILImage()(residual_output)
        # residual_output.save(img_name + '_Residual.png')

    else:
        output = model(input)

    output = output.cpu().data #(1, 1, 1024, 1024)
    output_np = output.numpy()[0, 0, :, :] #(1024,1024)
    output_np = np.stack((output_np, output_np, output_np), 0) #(3, 1024, 1024)

    output_np[np.where(output_np < 0)] = 0.0
    output_np[np.where(output_np > 1)] = 1.0
    # o = np.stack((o, o, o),1)
    output = torch.from_numpy(output_np)
    output = transforms.ToPILImage()(output) 
    output.save(img_name + '_SR.png')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=3, help='Depth of network.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel of network.')
    parser.add_argument('--residual', type=int, default= 0, help='Residual learning?')
    parser.add_argument('--dilation', type=int, default= 0, help='Dilation')
    parser.add_argument('--num_batches', type=int, default=15000, help='Number of batches to run.')
    parser.add_argument('--crop', type=int, default=128, help='Random crop size.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Base learning rate for Adam.')
    parser.add_argument('--factor', type=int, default=2, help='Interpolation factor.')
    parser.add_argument('--img', type=str, help='Path to input img.')
    parser.add_argument('--cuda', type=int, default=0 , help='Use cuda?')



    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    print(args.residual)
    img = Image.open(args.img)
    img_name = args.img
    img_name = str(img_name)
    if args.depth == 3:
        if args.dilation == 0:
            model = USSRNet_3(kernel_size=args.kernel_size)
        else: model = USSRNet_delated(dilation=args.dilation)
    elif args.depth == 5:
        if args.dilation == 0:
            model = USSRNet_5(kernel_size=args.kernel_size)
        else: model = USSRNet_5_delated(dilation=args.dilation)
    elif args.depth == 8:
        if args.dilation == 0:
            model = USSRNet_8(kernel_size=args.kernel_size)
        else: model = USSRNet_8_delated(dilation=args.dilation)
    else:
        print("Expecting appropriate model, instead got wrong depth")
        sys.exit(1)
    summary(model, (1, 256, 256))
    # num_channels = 1
    # if num_channels == 3:
    #     model = ZSSRNet_5(input_channels=3)
    # elif num_channels == 1:
    #     model = ZSSRNet_5(input_channels=1)
    # else:
    #     print("Expecting RGB or gray image, instead got", img.size)
    #     sys.exit(1)

    # Weight initialization
    model.apply(weights_init_kaiming)

    train(model, img, args.factor, args.num_batches, args.lr, args.crop)
    # test(model, img, args.factor)