from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import matplotlib as mpl

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        #x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        #print(num_in_layers)
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        #print(x)
        x = self.normalize(x)
        #print(x)
        return 0.7 * self.relu(x)


class Resnet50_md(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet50_md, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7, 1)  # H/2  -   64D
        #self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock(64, 64, 3, 1)  # H/8  -  256D
        self.conv3 = resblock(256, 128, 4, 1)  # H/16 -  512D
        self.conv4 = resblock(512, 256, 6, 1)  # H/32 - 1024D
        self.conv5 = resblock(1024, 512, 3, 1)  # H/64 - 2048D

        # decoder
        self.upconv7 = upconv(1024, 512, 3, 2)
        self.iconv7 = conv(1536, 256, 3, 1)

        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(1024 + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(640, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(322, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(98, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)###########################################
        # self.iconv1 = conv(16+2, 64, 3, 1)
        self.disp1_layer = get_disp(16)
        self.diap_out = nn.Conv2d(2,1,1)

        self.convps = nn.Conv2d(16,16*8*8,3,1,padding=1)#############################
        self.ps=nn.PixelShuffle(8)######################################################
        # self.convps = nn.Conv2d(64,16*8*8*3*3,3,1,padding=1)
        # self.ps=nn.PixelShuffle(24)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        #x_pool1 = self.pool1(x1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        #x5 = self.conv5(x4)

        # print("x",x.shape)
        # print("x1",x1.shape)
        # print("x2",x2.shape)
        # print("x3",x3.shape)
        # print("x4",x4.shape)

        upconv5 = self.upconv7(x4)
        concat5 = torch.cat((upconv5, x4), 1)
        iconv5 = self.iconv7(concat5)

        # print("upconv5",upconv5.shape)
        # print("concat5",concat5.shape)
        # print("iconv5",iconv5.shape)


        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, x3), 1)
        iconv4 = self.iconv4(concat4)

        # print("upconv4",upconv4.shape)
        # print("concat4",concat4.shape)
        # print("iconv4",iconv4.shape)


        self.disp4 = self.disp4_layer(iconv4)
        #self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, x2, self.disp4), 1)
        iconv3 = self.iconv3(concat3)

        # print("disp4",self.disp4.shape)
        # print("upconv3",upconv3.shape)
        # print("concat3",concat3.shape)
        # print("iconv3",iconv3.shape)


        self.disp3 = self.disp3_layer(iconv3)
        #self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)
        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, x1, self.disp3), 1)
        iconv2 = self.iconv2(concat2)

        # print("disp3",self.disp3.shape)
        # print("upconv2",upconv2.shape)
        # print("concat2",concat2.shape)
        # print("iconv2",iconv2.shape)        

        self.disp2 = self.disp2_layer(iconv2)
        #self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)
        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.disp2), 1)
        iconv1 = self.iconv1(concat1)

        # print("disp2",self.disp2.shape)
        # print("upconv1",upconv1.shape)
        # print("concat1",concat1.shape)
        # print("iconv1",iconv1.shape)        

        iconv1 = self.convps(iconv1)
        # print("iconv1",iconv1.shape)

        iconv1 = self.ps(iconv1)
        # print("iconv1",iconv1.shape)

        iconv1=F.interpolate(iconv1,size=[256,256],mode='bilinear',align_corners=True)
        # print("disp1",iconv1.shape)
        self.disp1 = self.disp1_layer(iconv1)
        # print("disp1",self.disp1.shape)

        # return self.disp1, self.disp2, self.disp3, self.disp4
        out = self.diap_out(self.disp1)

        # print("out",out.shape)
      
        # print(zzz.shape)
        return out

class Resnet18_md(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet18_md, self).__init__()
        # encoder
        #print('dfcsdfsd:',num_in_layers)
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        #print(self.conv1.shape)
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # decoder
        self.upconv6 = upconv(512, 512, 3, 2)
        self.iconv6 = conv(256+512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(64+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)
        self.diap_out = nn.Conv2d(2,1,1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        print(x.shape)
        x1 = self.conv1(x)
        #print(x.shape)
        print(x1.shape)
        x_pool1 = self.pool1(x1)
       
        x2 = self.conv2(x_pool1)
       
        x3 = self.conv3(x2)
     
        x4 = self.conv4(x3)
        print(x4.shape)
        x5 = self.conv5(x4)
        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4


        # decoder
        upconv6 = self.upconv6(x5)
        print(upconv6.shape)
        print(skip5.shape)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)
        #print(iconv6.shape)

        upconv5 = self.upconv5(iconv6)
        #print(upconv5.shape)
        concat5 = torch.cat((upconv5, skip4), 1)
        #print(concat5.shape)
        iconv5 = self.iconv5(concat5)
        #print(iconv5.shape)

        upconv4 = self.upconv4(iconv5)
        #print(upconv4.shape)
        concat4 = torch.cat((upconv4, skip3), 1)
        #print(concat4.shape)
        iconv4 = self.iconv4(concat4)
        #print(iconv4.shape)
        
        self.disp4 = self.disp4_layer(iconv4)
       
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)

        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        # print(upconv2.shape)
        # print(skip1.shape)
        # print(self.udisp3.shape)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        #print(concat2.shape)
        iconv2 = self.iconv2(concat2)

        self.disp2 = self.disp2_layer(iconv2)

        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        # print('iconv1 shape:',iconv1.shape)

        self.disp1 = self.disp1_layer(iconv1)
        #disp1 shape:torch.Size([8, 2, 512, 512])
        # print('disp1 shape:{},disp2 shape:{}, disp3 shape: {}, disp4 shape: {}'.format(self.disp1.shape,self.disp2.shape,self.disp3.shape,self.disp4.shape))
        out = self.diap_out(self.disp1)
        return out
        # return self.disp1, self.disp2, self.disp3, self.disp4


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class ResnetModel(nn.Module):
    def __init__(self, num_in_layers, encoder='resnet18', pretrained=False):
        super(ResnetModel, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50',\
                           'resnet101', 'resnet152'],\
                           "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)\
                                (pretrained=pretrained)
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                              kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1 # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool # H/4

        # encoder
        self.encoder1 = resnet.layer1 # H/4
        self.encoder2 = resnet.layer2 # H/8
        self.encoder3 = resnet.layer3 # H/16
        self.encoder4 = resnet.layer4 # H/32

        # decoder
        self.upconv6 = upconv(filters[3], 512, 3, 2)
        self.iconv6 = conv(filters[2] + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(filters[1] + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(filters[0] + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 1) #
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x_first_conv = self.firstconv(x)
        x = self.firstbn(x_first_conv)
        x = self.firstrelu(x)
        x_pool1 = self.firstmaxpool(x)
        x1 = self.encoder1(x_pool1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        # skips
        skip1 = x_first_conv
        skip2 = x_pool1
        skip3 = x1
        skip4 = x2
        skip5 = x3

        # decoder
        upconv6 = self.upconv6(x4)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=1, mode='bilinear', align_corners=True)
        self.disp4 = nn.functional.interpolate(self.disp4, scale_factor=0.5, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4
