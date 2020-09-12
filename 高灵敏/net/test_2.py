# -*- coding: utf-8 -*-
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import cv2
import os
import scipy.misc
from skimage import io,data
from erfnet import ERFNet
from model_hourglass import *
# from pspnet import PSPNet
from netldy import ldyNet
from densenet import DenseNet
from torchsummary import summary

# custom modules

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader
import PIL.Image as Image
# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (15, 10)
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from pytorch_msssim import msssim, ssim
from netldy import *
from netldyse import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def return_arguments():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--savemodel', default='./modelout/',
                        help='save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--maxdisp', type=int ,default=160,
                    help='maxium disparity')
    parser.add_argument('--data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images',
                        default='/media/usr134/数据备份/SY/134net/端到端/test/'
                        )
    parser.add_argument('--loadmodel', default='/media/usr134/数据备份/SY/134net/model/1205_data_cpt_7700.pth',
                    help='load model')                    
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256) 
    parser.add_argument('--input_width', type=int, help='input width',
                        default=256)
    parser.add_argument('--model', default='stackhourglass',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=1,
                        help='number of total epochs to run')

    parser.add_argument('--batch_size', default=1,
                        help='mini-batch size (default: 256)')

    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose gpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
                        
    parser.add_argument('--num_workers', default=0,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False)

    args = parser.parse_args()
    return args



l1loss = nn.L1Loss(size_average=True)
mseloss = nn.MSELoss()

class Model:

    def __init__(self, args):
        self.args = args

        self.device = args.device
        if args.model == 'stackhourglass':
            
                #self.model = ERFNet(2)
                #self.model = FAN(1,1)
                #self.model = ERFNet(1)
                self.model = ldyNet(1)
                #self.model = DenseNet()

       
        self.model.cuda()
        summary(self.model,(1,256,256))
        self.model.load_state_dict(torch.load(args.loadmodel))
            # print(self.model.state_dict())
            # print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, 
                                                     args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)
 
        if 'cuda' in self.device:
            torch.cuda.synchronize()

    

    def test(self):
        
        i = 0
        sum_loss = 0.0
        sum_ssim = 0.0
        average_ssim = 0.0
        average_loss = 0.0
        PSNRLoss = 0.0
        sum_PSNRLoss = 0.0
        PSNR = 0.0

        for epoch in range(self.args.epochs):


            self.model.eval()   #?start?

            for data in self.loader:       #(test-for-train)
                i = i + 1


                data = to_device(data, self.device)

                left = data['left_image']
                bg_image = data['bg_image']

                disps = self.model(left)

                #print(disps.shape)

                l_loss = l1loss(disps,bg_image)
                ssim_loss = ssim(disps,bg_image)
                #PSNR = mseloss(disps,bg_image)
                PSNRLoss = 10*torch.log10(255/torch.sqrt( mseloss(disps,bg_image)))

                sum_loss = sum_loss + l_loss.item()
                sum_ssim += ssim_loss.item()
                sum_PSNRLoss = sum_PSNRLoss + PSNRLoss.item()

                average_ssim = sum_ssim / i
                average_loss = sum_loss / i
                average_PSNR = sum_PSNRLoss / i

                # print average_loss
                '''
                disp_show = disps.squeeze()
                bg_show = bg_image.squeeze()
                print(bg_show.shape)
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(disp_show.data.cpu().numpy())
                plt.subplot(1,2,2)
                plt.imshow(bg_show.data.cpu().numpy())
                plt.show() 
                '''
        print('average loss:',average_loss,'\naverage_ssim:',average_ssim,'\naverage_PSNR:',average_PSNR)

    
def get_layer_param(model):
    print(sum([torch.numel(param) for param in model.parameters()])) 

def main():
    print('nobackground_noAug')
    args = return_arguments()
    model = Model(args)
    get_layer_param(model)

    model.test()


if __name__ == '__main__':
    main()#
