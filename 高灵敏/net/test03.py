# -*- coding: utf-8 -*-
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import cv2
import os
import scipy.misc
# from skimage import io,data
from models_resnetx import *##############################################
# from models_resnetx_wide import *
# from pspnet import PSPNet
import imageio

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
                        It    should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images',
                        default='./端到端-butongweizhi_copy/test/'
                        )
    parser.add_argument('--loadmodel', 
                        default='./model/0913_cpt.pth',
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
def psnr1(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

mse_loss = nn.MSELoss(size_average=True)  

l1loss = nn.L1Loss(size_average=True)
class Model:

    def __init__(self, args):
        self.args = args

        self.device = args.device
        if args.model == 'stackhourglass':
            
                # self.model = ERFNet(2)
                self.model = Resnet50_md(1)##############################################
                # self.model = resnet_wide(1)
                # self.model = Resnet_psp_md(1)

       
        self.model.cuda()
        # self.model = nn.DataParallel(self.model)
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
        msee=0
        i = 0
        sum_loss = 0.0
        sum_ssim = 0.0
        sum_psnr = 0.0
        average_ssim = 0.0
        average_loss = 0.0

        adir='./xiaoguo0912/'
        isExists=os.path.exists(adir)
        if not isExists:
            os.mkdir(adir)
        for epoch in range(self.args.epochs):


            self.model.eval()   #?start?
            
            for data in self.loader:       #(test-for-train)
                i = i + 1


                data = to_device(data, self.device)

                left = data['left_image']
                bg_image = data['bg_image']
                left=left.permute(0,3,1,2)
                
                ps = nn.PixelShuffle(32)
                Net2Iput = ps(left)
                Net2Iput=Net2Iput/10000.0
                # print(Net2Iput)
                # print("ppppppppppppp",Net2Iput.shape)
                # print("aaaaaaaa",bg_image.shape)

            
                # plt.figure()
                # plt.imshow(left.squeeze().cpu().detach().numpy())
                # plt.show()  
                # plt.imshow(bg_image.squeeze().cpu().detach().numpy())
                # plt.show()  
                
                disps = self.model(Net2Iput)

                # print(disps.squeeze().shape)
                # print(bg_image.squeeze().shape)
                print('mseloss: ',mse_loss(disps,bg_image).item())
                msee+=mse_loss(disps,bg_image).item()

                l_loss = l1loss(disps,bg_image)
                ssim_loss = ssim(disps,bg_image)
                adisp=disps.cpu().detach().numpy().squeeze()
                abg_image=bg_image.cpu().detach().numpy().squeeze()

                
                psnr_loss = psnr1(adisp,abg_image)


                sum_loss = sum_loss + l_loss.item()
                sum_ssim += ssim_loss.item()
                sum_psnr += psnr_loss

                average_ssim = sum_ssim / i
                average_loss = sum_loss / i
                average_psnr = sum_psnr / i

                # print average_loss

                disp_show = disps.squeeze()
                bg_show = bg_image.squeeze()
                # print(bg_show.shape)
                # plt.figure()
                # plt.subplot(1,2,1)
                # plt.imshow(disp_show.data.cpu().numpy())
                # plt.subplot(1,2,2)
                # plt.imshow(bg_show.data.cpu().numpy())
                # plt.show() 


                leftimag = os.path.join(adir, str('85_%06d' % i) + '-.bmp')
                # leftimag = os.path.join('./xiaoguo256-1000/', str('85_%06d' % i) + '-.bmp')
                bimag=bg_image.cpu().numpy().squeeze()
                imageio.imwrite(leftimag, bimag)
                disp_showa = disp_show.data.cpu().detach().numpy()
                dst = os.path.join(adir, str('85_%06d' % i) + '.bmp')
                # dst = os.path.join('./xiaoguo256-1000/', str('85_%06d' % i) + '.bmp')
                imageio.imwrite(dst, disp_showa)
        print('average loss',average_loss,average_ssim,average_psnr)
        print('averageMSE: ',msee)

    


def main():
    print('nobackground_noAug')
    args = return_arguments()
    model = Model(args)
    model.test()


if __name__ == '__main__':
    main()#
