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
import png

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
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')
    parser.add_argument('--maxdisp', type=int ,default=96,
                    help='maxium disparity')
    parser.add_argument('--data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images',
                        default='/home/a521/wqx/MonoDepth-PyTorch-master3/train1 (copy)/1'
                        )
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images',
                        default='/home/a521/wqx/MonoDepth-PyTorch-master/data/test'
                        )
    parser.add_argument('--model_path', help='path to the trained model',default='/home/a521/wqx/MonoDepth-PyTorch-master3/model-small/1.pth')
    parser.add_argument('--output_directory',
                        help='where save dispairities\
                        for tested images',
                        default='/home/a521/wqx/MonoDepth-PyTorch-master3/data/output'
                        )
    # parser.add_argument('--loadmodel', default='/home/a521/wqx/MonoDepth-PyTorch-master3/model-small/1_cpt.pth',
    #                 help='load model')                    
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='stackhourglass',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='test',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=900,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=0,
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
      

    if epoch >= 0 and epoch <300:
        lr = learning_rate
    if epoch >= 300 and epoch <600:
        lr = learning_rate/10
    if epoch >= 600 and epoch <=900:
        lr = learning_rate/100    


    # if epoch >=0 :
    #     lr=  learning_rate/4  
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    #print(disp.shape)
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)

    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    #print( _.shape)
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    #print(l_mask)
    #print(l_mask.shape)
    r_mask = np.fliplr(l_mask)#翻转函数 zhengzhong fanzhuan(left and right)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))
class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        #print(disp.shape)
        #print(x.shape)
        out = torch.sum(x*disp,1)
        #print(out.shape)
        return out
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out    
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        #########CNN
        output      = self.firstconv(x)#conv0_1  to conv0_3
        output      = self.layer1(output)#conv1_1 
        output_raw  = self.layer2(output)#conv1_2
        output      = self.layer3(output_raw)#conv1_3
        output_skip = self.layer4(output)#conv1_4
        ##########CNN


        ##########SPP
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        
        return output_feature
class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        
        pre  = self.conv2(out) #in:1/8 out:1/8

        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)
     
        out  = self.conv3(pre) #in:1/8 out:1/16

        out  = self.conv4(out) #in:1/16 out:1/16
     

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post   


def sigmoid(disp):
   
    sigmoid = torch.nn.Sigmoid()
   
    return 0.3*sigmoid(disp)

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

########
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 2, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 2, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 2, kernel_size=3, padding=1, stride=1,bias=False))

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



    def forward(self, left, right):
        # print(left.shape)
        refimg_fea     = self.feature_extraction(left)
        # print(refimg_fea.shape)
        targetimg_fea  = self.feature_extraction(right)
        

        #matching volume
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, int(self.maxdisp/4),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        # print(cost.shape)
        for i in range(int(self.maxdisp/4)):#chuyi 4 yinwei tupiansuoxiao 4 bei
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i] #everything except the last i items
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea  #quchu bianjiede rongyuxinxi 
        cost = cost.contiguous()   #[1, 64, 38, 64, 128]
        # print(cost.shape)
        ###############cost volume

        # cost0 = self.dres0(cost)
        # cost0 = self.dres1(cost0) + cost0

        # cost0 = self.dres2(cost0) + cost0 
        # cost0 = self.dres3(cost0) + cost0 
        # cost0 = self.dres4(cost0) + cost0
        # cost1 = self.classify(cost0)
        

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0   
        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0       
        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0          
        cost1 = self.classif1(out1)  
        # print(out1.shape)  
        # print(cost1.shape)
        cost2 = self.classif2(out2) + cost1   
        cost3 = self.classif3(out3) + cost2 

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')     #[1,1,192,256,512] 
            # print(cost1.shape)
            cost11 = torch.squeeze(cost1,1)[:,0,:,:,:]   
            cost12 = torch.squeeze(cost1,1)[:,1,:,:,:]                                    #[1,192,256,512]
            pred1 = F.softmax(cost11,dim=1)    
            pred11 = F.softmax(cost12,dim=1)                                                              #[1,192,256,512]
            pred1 = disparityregression(self.maxdisp)(pred1)  #[1,2,256,512]
            pred11= disparityregression(self.maxdisp)(pred11)                                 
            pred1=torch.transpose(pred1.unsqueeze(2), 1, 2)
            pred11=torch.transpose(pred11.unsqueeze(2), 1, 2)
            # print(pred1.shape)
            pre1 = [pred1, pred11]

            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')     #[1,1,192,256,512] 
            cost21 = torch.squeeze(cost2,1)[:,0,:,:,:]   
            cost22 = torch.squeeze(cost2,1)[:,1,:,:,:]                                    #[1,192,256,512]
            pred2 = F.softmax(cost21,dim=1)    
            pred21 = F.softmax(cost22,dim=1)                                                              #[1,192,256,512]
            pred2 = disparityregression(self.maxdisp)(pred2)  #[1,2,256,512]
            pred21= disparityregression(self.maxdisp)(pred21)                                 
            pred2=torch.transpose(pred2.unsqueeze(2), 1, 2)
            pred21=torch.transpose(pred21.unsqueeze(2), 1, 2)
            pre2 = [pred2, pred21]

            cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')     #[1,1,192,256,512] 
            cost31 = torch.squeeze(cost3,1)[:,0,:,:,:]   
            cost32 = torch.squeeze(cost3,1)[:,1,:,:,:]                                    #[1,192,256,512]
            pred3 = F.softmax(cost31,dim=1)    
            pred31 = F.softmax(cost32,dim=1)                                                              #[1,192,256,512]
            pred3 = disparityregression(self.maxdisp)(pred3)  #[1,2,256,512]
            pred31= disparityregression(self.maxdisp)(pred31)                                 
            pred3=torch.transpose(pred3.unsqueeze(2), 1, 2)
            pred31=torch.transpose(pred31.unsqueeze(2), 1, 2)
            pre3 = [pred3, pred31]
        else:
            cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')     #[1,1,192,256,512] 
            cost31 = torch.squeeze(cost3,1)[:,0,:,:,:]   
            cost32 = torch.squeeze(cost3,1)[:,1,:,:,:]                                    #[1,192,256,512]
            pred3 = F.softmax(cost31,dim=1)    
            pred31 = F.softmax(cost32,dim=1)                                                              #[1,192,256,512]
            pred3 = disparityregression(self.maxdisp)(pred3)  #[1,2,256,512]
            pred31= disparityregression(self.maxdisp)(pred31)                                 
            pred3=torch.transpose(pred3.unsqueeze(2), 1, 2)
            pred31=torch.transpose(pred31.unsqueeze(2), 1, 2)
            pre3 = [pred3, pred31]    
        if self.training:
            pre1= torch.cat(pre1, dim=1)
            pre2= torch.cat(pre2, dim=1)
            pre3= torch.cat(pre3, dim=1)
            return pre3,pre2,pre1
        else :
          
            pre3= torch.cat(pre3, dim=1)
            return pre3

 

       

class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        if args.model == 'stackhourglass':
                self.model = PSMNet(args.maxdisp)
        self.model = self.model.to(self.device)

        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=3,
                SSIM_w=0.85,
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
                                                  
        else:
            #self.model.load_state_dict(torch.load(args.model_path))
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

    def erro(self,disps,groundtruthl):
        pred_disp = disps.data.cpu()


        true_disp = groundtruthl


        plt.imshow(np.squeeze(np.transpose(pred_disp.cpu().detach().numpy(), (1, 2,
                               0))))
        plt.show()
        plt.imshow(np.squeeze(np.transpose(groundtruthl.cpu().detach().numpy(), (1, 2,
                               0))))
        plt.show()

        
        index = np.argwhere(true_disp>0)
        groundtruthl[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (groundtruthl[index[0][:], index[1][:], index[2][:]] < 3)|(groundtruthl[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

        
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
            # params = list(self.model.parameters())
            # k = 0
            # for i in params:
            #         l = 1
            #         print("该层的结构：" + str(list(i.size())))
            #         for j in i.size():
            #             l *= j
            #         print("该层参数和：" + str(l))
            #         k = k + l
            # print("总参数数量和：" + str(k))
            for data in self.loader:       #(test-for-train)
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']    
                # plt.imshow(np.transpose(left.squeeze().cpu().detach().numpy(), (1, 2,
                #                0)))
                # plt.show()   
                # plt.imshow(np.transpose(right.squeeze().cpu().detach().numpy(), (1, 2,
                #                0)))
                # plt.show()     
                self.optimizer.zero_grad()
                # disps = self.model(torch.cat(left,right),1)
                disps = self.model(left,right)
              
                # plt.imshow(disps[0][:,0,:,:].squeeze().cpu().detach().numpy())
                # plt.show()  
             
                #print(disps[0])
               
                
                loss = self.loss_function(disps,[left,right])
                
           

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                running_loss += loss.item()
            print(' time = %.2f' %(time.time() - c_time))
            #running_loss /=( self.n_img / self.args.batch_size)
           
   
            print('running_loss:', running_loss)
            # running_val_loss = 0.0
            # self.model.eval()
            # start_time = time.time()


            # aa=159
            # total_test_loss=0
            # fileground="/home/a521/wqx/MonoDepth-PyTorch-master2/data_scene_flow/testing/1/"
            # left_fold  = 'image_2/'
            # right_fold = 'image_3/'
            # disp_L = 'disp_occ_0/'
            # disp_R = 'disp_occ_1/'
            # for data in self.val_loader:
            #     # aa+=1
            #     data = to_device(data, self.device)
            #     left = data['left_image']
            #     right = data['right_image']
            #     # plt.imshow(left.squeeze().cpu().detach().numpy())
            #     # plt.show()
            #     disps = self.model(left,right)
            #     #print(torch.min(disps[:,0,:,:].data,2))
            #     # plt.figure()
            #     # plt.imshow(disps[:,0,:,:].squeeze().cpu().detach().numpy())
            #     # #plt.figure()
            #     # # plt.imshow(disps[:,1,:,:].squeeze().cpu().detach().numpy())
            #     # # plt.figure()
            #     # # plt.imshow((disps[:,0,:,:]+disps[:,1,:,:]).squeeze().cpu().detach().numpy())
            #     # plt.show()

            #     # groundtruth=fileground+disp_L+'000'+str(aa)+'_10.png'
            #     # groundtruthl=Image.open(groundtruth)
            #     # w=groundtruthl.size[0]
            #     # h=groundtruthl.size[1]
            #     # groundtruthl = groundtruthl.crop((w-1232, h-368, w, h))
            #     # groundtruthl = np.ascontiguousarray(groundtruthl,dtype=np.float32)/256
               
            #     # size = (int(512), int(256))  
            #     # groundtruthl = cv2.resize(groundtruthl, size, interpolation=cv2.INTER_LINEAR)  

               
              
            #     # groundtruthl=torch.from_numpy(groundtruthl)
             
                
            #     # groundtruthl=(groundtruthl).unsqueeze(0)

                
                
            #     # plt.imshow(np.squeeze(np.transpose(groundtruthl.cpu().detach().numpy(), (1, 2,
            #     #                0))))
            #     # plt.show()
        
            #     # test_loss = self.erro(disps[:,0,:,:].squeeze(1),groundtruthl)
            #     # print('Iter %d 3-px error in val = %.3f' %(aa-160,test_loss*100))

            #     loss = self.loss_function(disps,[left,right])
                
                
               
            #     #val_losses.append(loss.item())
            #     running_val_loss += loss.item()
            # print('allllllllllllllllllllll:',running_val_loss)    
            # print(' time = %.2f' %(time.time() - start_time))
            # Estimate loss per image
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

    def test(self):
        self.model.eval()
        # disparities = np.zeros((self.n_img,
        #                        376, 1241),
        #                        dtype=np.float32)
                             
        # disparities_pp = np.zeros((self.n_img,
        #                           376, 1241),
        #                           dtype=np.float32)
   
        time_start=time.time()
        with torch.no_grad():

            # groundtruthl=Image.open("/home/a521/Downloads/devkit/matlab/data/disp_est.png")
            # groundtruthl = np.ascontiguousarray(groundtruthl,dtype=np.float32)/256
            # groundtruthl=torch.from_numpy(groundtruthl)
        
            for (i, data) in enumerate(self.loader):
               
                
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                # leftflip=torch.flip(left, [3])
                # rightflip=torch.flip(right, [3])
                # plt.imshow(np.transpose(left.squeeze().cpu().detach().numpy(), (1, 2,
                #                0)))
                # plt.show()
                # plt.imshow(np.transpose(right.squeeze().cpu().detach().numpy(), (1, 2,
                #                0)))
                # plt.show()





                
                leftgray=255*(left[:,0,:,:]* 0.3+0.59*left[:,1,:,:]+0.11*left[:,2,:,:]) #1*256*512
                rightgray=255*(right[:,0,:,:]* 0.3+0.59*right[:,1,:,:]+0.11*right[:,2,:,:] )#1*256*512
                leftnumpy = np.array(leftgray)
                rightnumpy = np.array(rightgray)

                left_image1 = np.lib.pad(np.transpose(leftnumpy,(1,2,0)),((4,4),(4,4),(0,0)),mode='edge')
                right_image1 = np.lib.pad(np.transpose(rightnumpy,(1,2,0)),((4,4),(4,4),(0,0)),mode='edge')
                leftpad = Variable(torch.from_numpy(np.transpose(left_image1,(2,0,1))).data.cuda()) #1*264*520
                rightpad = Variable(torch.from_numpy(np.transpose(right_image1,(2,0,1))).data.cuda())
                
                leftdeviation=leftgray #1*256*512
                rightdeviation=rightgray
                for k in range(left.shape[2]):
                    for l in range(left.shape[3]):
                        leftdeviation[:,k,l]=torch.std(leftpad[:,k:k+9,l:l+9])
                        rightdeviation[:,k,l]=torch.std(rightpad[:,k:k+9,l:l+9])   
                leftdeviation=leftdeviation*255
                rightdeviation=rightdeviation*255


           
                xx=leftdeviation.squeeze().cpu().detach().numpy().astype(np.uint16)
                aa=str(i)+"_left.png"
                with open(aa, 'wb') as f:
                    writer = png.Writer(width=xx.shape[1], height=xx.shape[0], bitdepth=16, greyscale=True)
                    zgray2list = xx.tolist()
                    writer.write(f, zgray2list)

                yy=rightdeviation.squeeze().cpu().detach().numpy().astype(np.uint16)
                bb=str(i)+"_right.png"
                with open(bb, 'wb') as f:
                    writer = png.Writer(width=yy.shape[1], height=yy.shape[0], bitdepth=16, greyscale=True)
                    zgray2list = yy.tolist()
                    writer.write(f, zgray2list)    

  

           
        time_end=time.time()
        print('totally cost',time_end-time_start)
        # np.save(self.output_directory + '/disparities.npy', disparities)
        # np.save(self.output_directory + '/disparities_pp.npy',
        #         disparities_pp)
        print('Finished Testing')


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

