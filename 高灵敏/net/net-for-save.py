# -*- coding: utf-8 -*-=
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import cv2
import os
import scipy.misc
from skimage import io,data
import png
import h5py

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
                        default='/media/a521/CHASESKY/SANWEI/train'
                        )
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images',
                        default='/home/a521/wqx/MonoDepth-PyTorch-master/data/test'
                        )
    parser.add_argument('--model_path', help='path to the trained model',default='/home/a521/wqx/frame/model3/1.pth')
    parser.add_argument('--output_directory',
                        help='where save dispairities\
                        for tested images',
                        default='/home/a521/wqx/MonoDepth-PyTorch-master3/data/output'
                        )
    parser.add_argument('--loadmodel', default='/home/a521/wqx/frame/model3/1_4.pth',
                    help='load model')                    
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256) 
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='stackhourglass',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=True,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=900,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-3,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=1,
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
    parser.add_argument('--use_multiple_gpu', default=True)

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""
      

    if epoch >= 30 and epoch < 40:
         lr = learning_rate / 2
    elif epoch >= 40:
         lr = learning_rate / 4
    else:
         lr = learning_rate   
    # if epoch >=0 :
    #     lr=  learning_rate/4  
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(F.relu(x)))
        out = self.conv2(F.relu(out))
        out += x
        return self.bn1(out)



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 50

        self.conv1 = nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(50)
        self.layer1 = self._make_layer(block, 50, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 50, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 50, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 50, num_blocks[3], stride=1)
        # self.linear = nn.Linear(50*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
      
        out = self.layer1(x)
        out = self.layer2(out)
     
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.ResNet = ResNet(BasicBlock, [2, 2, 2, 2])
        self.bn1 = nn.BatchNorm2d(50)
        self.inplanes = 32
        


    def forward(self, x):
        #########CNN
        output1      = self.ResNet(x)
        return output1




class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

########
        self.firstconv = nn.Sequential(convbn(1, 50, 3, 1, 1, 1),
                                        nn.BatchNorm2d(50)
                                      )
                             
        self.firstconv1 = nn.Sequential(
                                      nn.ReLU(inplace=True),
                                        convbn(50, 50, 3, 1, 1, 1),
                                        nn.BatchNorm2d(50),
                                       nn.ReLU(inplace=True),
                                       convbn(50, 1, 3, 1, 1, 1)
                                      )
                         


   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()



    def forward(self, left):
        # print(left.shape)

        output      = self.firstconv(left)
        backgroundfeature    = self.feature_extraction(output)
        Backgroundimage      = self.firstconv1(backgroundfeature)

        return Backgroundimage

class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        if args.model == 'stackhourglass':
                self.model = PSMNet(args.maxdisp)
        self.model = self.model.to(self.device)
        
        # if args.cuda:
        #     model = torch.nn.DataParallel(self.model)
        #     model.cuda()
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
            # self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode,
            #                                                      args.augment_parameters,
            #                                                      False, args.batch_size,
            #                                                      (args.input_height, args.input_width),
            #                                                      args.num_workers)
            # self.pretrained_dict = torch.load(args.loadmodel)      
            # self.model_dict = self.model.state_dict() 
            # self.pretrained_dict = {k: v for k, v in self.pretrained_dict.items() if k in self.model_dict} 
            # self.model_dict.update(self.pretrained_dict) 
            # self.model.load_state_dict(self.model_dict)
            # self.model.load_state_dict(torch.load(args.loadmodel))


            



            # if args.loadmodel is not None:
            #     pretrained_dict = torch.load(args.loadmodel)
           
            #     self.model_dict = self.model.state_dict() 
            #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict} 
            #     self.model_dict.update(pretrained_dict)
            #     self.model.load_state_dict(self.model_dict)
   
 
            #     print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))                                         
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


        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width
  
        
        if 'cuda' in self.device:
            torch.cuda.synchronize()

    

        
    def disparity_loader(self,path):
        return Image.open(path)
    def train(self):
        
        losses = []
        #val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        for epoch in range(self.args.epochs):
            #input()
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()   #?start?
            pp=1
            oo=1
            ii=1
            index=0
            for data in self.loader:       #(test-for-train)
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                target = torch.tensor(data['right_image']).float()
                target=target.permute(0,2,1).unsqueeze(1)

                mask=(data['mask_image']*255).int()
                # plt.imshow((left).squeeze().cpu().detach().numpy())
                # plt.show() 
                # plt.imshow((target).squeeze().cpu().detach().numpy())
                # plt.show() 
                # plt.imshow((mask).squeeze().cpu().detach().numpy())
                # plt.show() 
                self.optimizer.zero_grad()
  
                
             
                left = F.interpolate(left,size=(480,640),mode='bilinear')
                target = F.interpolate(target,size=(480,640),mode='bilinear')
                mask = F.interpolate(mask.float(),size=(480,640),mode='nearest').int()

                disps = self.model(left)
                haahah=np.transpose(disps.squeeze().cpu().detach().numpy(),(1,0))

                
                ii=int(index%18+1)
                oo=int(index/18+1)
                pp=int(index/180+1)
                aa=str(pp)+'-'+str(oo)+'-'+str(ii)+'.h5'
                path='/home/a521/wqx/frame/train/data/predict/'+aa
                f = h5py.File(path, 'w')
                dset = f.create_dataset("mydataset", data= haahah)
                f.close()
                index+=1
                
                loss = self.loss_function(disps,target,mask)
                
                

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                running_loss += loss.item()
            print(' time = %.2f' %(time.time() - c_time))
            #running_loss /=( self.n_img / self.args.batch_size)
           
   
            print('running_loss:', running_loss)
           
            running_loss /= (self.n_img / self.args.batch_size)
            # running_val_loss /= (self.val_n_img / self.args.batch_size)
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
           
            self.save(self.args.model_path[:-4] + '_'+str(epoch+1)+'.pth')
            # if  epoch==100:
            #     self.save(self.args.model_path[:-4] + '_100.pth')  
            # if  epoch==150:
            #     self.save(self.args.model_path[:-4] + '_150.pth')  
            # if  epoch==200:
            #         self.save(self.args.model_path[:-4] + '_50.pth')
            # if  epoch==250:
            #     self.save(self.args.model_path[:-4] + '_100.pth')        
            self.save(self.args.model_path[:-4] + '_last.pth')#上一回
            if running_loss < best_val_loss:
                #print(running_val_loss)
                #print(best_val_loss)
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                best_val_loss = running_loss
                print('Model_saved')

        print ('Finished Training. Best loss:', best_loss)
        #self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
 
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    


def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()#


if __name__ == '__main__':
    main()#
