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
# from model_concat191125 import ERFNet
from models_resnetx_down import *
# from model import *
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def return_arguments():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

    parser.add_argument('--data_dir',
                        help='path to the dataset folder',
                        default='./端到端-butongweizhi_copy/train/'
                        )
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images',
                        default='./端到端-butongweizhi_copy/test/'
                        )
    parser.add_argument('--model_path', help='path to the trained model',
                        default='./model256/0220_dataTO_512_cpt.pth')

    parser.add_argument('--loadmodel', default='./model//1205_data_cpt_1020.pth',
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
    parser.add_argument('--epochs', default=1200,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=32,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
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

    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
                        
    parser.add_argument('--num_workers', default=2,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False)

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""
      
    if epoch >=100 and epoch < 250:
         lr = learning_rate/5
    elif epoch >=250 and epoch < 800:
         lr = learning_rate/10
    # if epoch >= 1000 and epoch <=1500:
    #      lr = learning_rate/100
    # if epoch >= 600 :
    #      lr = learning_rate    
    elif epoch >= 800 :
         lr = learning_rate/20
   
    else:
         lr = learning_rate  
    # if epoch >=0 :
    #     lr=  learning_rate/4  
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


      




class Model:

    def __init__(self, args):
        self.args = args

        self.device = args.device
        if args.model == 'stackhourglass':
            
                # self.model = ERFNet(2)
                # self.model = Resnet101_1(1)
                self.model = Resnet50_md(1)
       
       
        self.model.cuda() #= self.model.to(self.device)

        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=3,
                SSIM_w=0.8,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, 'test',
                                                                 args.augment_parameters,
                                                                 False, 1,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)


            # self.model.load_state_dict(torch.load(args.loadmodel))#jiazai yuxunlian moxing jiezhu ciju 
            # print("load_state_dict")
                                      
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1
        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, 
                                                     args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)


     
  
        
        if 'cuda' in self.device:
            torch.cuda.synchronize()

    

        
    def disparity_loader(self,path):
        return Image.open(path)
    def train(self):
        
        # losses = []
        #val_losses = []

        best_loss = float('Inf')
        best_val_loss = 1000000.0
        cashesTrueloss =  [([0]) for i in range(self.args.epochs)]
        cashesValloss  =  [([0]) for i in range(self.args.epochs)]

        for epoch in range(self.args.epochs):

           
            downsize=24 # THE downsample size
            # downsize=28 # THE downsample size
       

       
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()   #?start?
            icout=1
            for data in self.loader:       #(test-for-train)
                # Load data
                
                
                self.optimizer.zero_grad()
                # loss=0
               
                data = to_device(data, self.device)

               

                left = data['left_image']
                bg_image = data['bg_image']

                left=left.permute(0,3,1,2)
                left=left[:,0:downsize*downsize,:,:]
                
                ps = nn.PixelShuffle(downsize)
                Net2Iput = ps(left)
                Net2Iput=Net2Iput/10000.0
                # print(Net2Iput)
                # print("ppppppppppppp",Net2Iput.shape)
                # print("aaaaaaaa",bg_image.shape)

                disps = self.model(Net2Iput)
                # print('gggggggggggggggggggggggggggggggggggggggggg',left.shape,disps.shape)
                # plt.imshow(disps.squeeze().cpu().detach().numpy())
                # plt.show() 

                loss1 = self.loss_function(disps,bg_image)
                loss1.backward()
       
                
             
        
                
                self.optimizer.step()
              
                
                # losses.append(loss.item())
                running_loss += loss1.item()
            # print(' time = %.2f' %(time.time() - c_time))
            running_loss /=( self.n_img / self.args.batch_size)
           
   
            # print('running_loss:', running_loss)
      
            # running_val_loss /= (self.val_n_img / self.args.batch_size)
            print('Epoch:',epoch + 1,'train_loss:',running_loss,'time:',round(time.time() - c_time, 3),'s')
                

            
            TrueLoss=math.sqrt( running_loss )*255
            cashesTrueloss[epoch]=TrueLoss
            print ('TrueLoss:',TrueLoss)
                



            if epoch%1==0:
                self.model.eval()
                i=0
                running_val_loss = 0.0
                for data in self.val_loader:
                    data = to_device(data, self.device)

                    left = data['left_image']
                    bg_image = data['bg_image']
                    left=left.permute(0,3,1,2)
                
                    left=left[:,0:downsize*downsize,:,:]
                
                    ps = nn.PixelShuffle(downsize)
                    Net2Iput = ps(left)
                    Net2Iput=Net2Iput/10000.0

            
   
                    with torch.no_grad():
                        # newinput=torch.cat([left,bg],1)
                        # disps = self.model(newinput)

                        disps = self.model(Net2Iput)
            
                    
                    loss1 = self.loss_function(disps,bg_image)        
                    # loss1 = self.loss_function((disps+left1.float()),target,mask)
                    running_val_loss+=loss1
                        
                
                running_val_loss/=self.val_n_img
                running_val_loss1 =math.sqrt( running_val_loss )*255
                print( 'running_val_loss = %.12f' %(running_val_loss1))
                cashesValloss[epoch] =running_val_loss1

            if running_val_loss < best_val_loss:
            
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')

            self.saveLoss(cashesTrueloss, epoch+1,'True')
            self.saveLoss(cashesValloss,  epoch+1,'Val')

            if epoch % 20 == 0:
                self.save(self.args.model_path[:-4] + '_'+str(epoch)+'.pth')
            # self.save(self.args.model_path[:-4] + '_'+str(epoch+1)+'.pth')
            # if  epoch==100:
            #     self.save(self.args.model_path[:-4] + '_100.pth')  
            # if  epoch==150:
            #     self.save(self.args.model_path[:-4] + '_150.pth')  
            # if  epoch==200:
            #         self.save(self.args.model_path[:-4] + '_200.pth')
            # if  epoch==250:
            #     self.save(self.args.model_path[:-4] + '_250.pth')        
            # self.save(self.args.model_path[:-4] + '_last.pth')#上一回
            # if running_loss < best_val_loss:
            #     #print(running_val_loss)
            #     #print(best_val_loss)
            #     self.save(self.args.model_path[:-4] + '_cpt.pth')
            #     best_val_loss = running_loss
            #     print('Model_saved')

        # print ('Finished Training. Best loss:', best_loss)
        #self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def saveLoss(self,value, epoch, txtType):
        x = range(epoch)
        plt.plot(x,value[0:epoch])
        lossFigPath = self.args.model_path[:-4] + '_' + txtType + 'loss.jpg'
        plt.savefig(lossFigPath)
        
        txtPath = self.args.model_path[:-4] + '_' + txtType + 'loss.txt'
        with open(txtPath,"w") as f:
            f.writelines(str(value[0:epoch]))
        f.close
        # print(txtType, ' Lossfig & LossTXT saved.') 

    


def main():
    print('nobackground_noAug')
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()#


if __name__ == '__main__':
    main()#
