
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class UpsamplerBlock11 (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 5, stride=4, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()
        self.initial_block1 = DownsamplerBlock(1,16)

        self.initial_block2 =(DownsamplerBlock(16,64))

        self.layers1 = nn.ModuleList()

        for x in range(0, 5):    #5 times
           self.layers1.append(non_bottleneck_1d(64, 0.3, 1))  

        self.initial_block3 = (DownsamplerBlock(64,128))

        self.layers2 = nn.ModuleList()

        for x in range(0, 2):    #2 times
            self.layers2.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 16))

        #only for encoder mode:
        self.output_conv1 = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)


        self.initial_block4 = (UpsamplerBlock(128,64))

        self.layers3 = nn.ModuleList()
        self.layers3.append(non_bottleneck_1d(64, 0.3, 1))
        self.layers3.append(non_bottleneck_1d(64, 0.3, 1))

        self.initial_block5 = (UpsamplerBlock(64,16))
        self.initial_block6 = (UpsamplerBlock11(128,32))

        self.layers4 = nn.ModuleList()
        self.layers4.append(non_bottleneck_1d(16, 0.3, 1))
        self.layers4.append(non_bottleneck_1d(16, 0.3, 1))

        self.output_conv2 = nn.ConvTranspose2d( 16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.concat_x3 = nn.Conv2d(192, 64, 1,)
        self.concat_x4 = nn.Conv2d(144, 16, 1,)
        self.concat_x1 = nn.Conv2d(32, 16, 1,)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
    def forward(self, input):
        output = self.initial_block1(input)

        # x1=output#128
        
        output = self.initial_block2(output)
        
        x2=output#64
        for layer in self.layers1:
            output = layer(output)
        # x3=output#64

        output = self.initial_block3(output)#32
   
        #x4=output
        for layer in self.layers2:
            output = layer(output)  

        x11=output
        # x12 = self.initial_block4(x11)#32
        # print("x1212222222",x12.shape)  

        output = self.initial_block4(output)#up

        # output=torch.cat((output,output),1)
        # output = self.concat_x3(output)
        out111=self.Upsample(x11)
        # print("out11111111111",out111.shape)  


        # output=torch.cat((output,x3),1)#64
        # output = self.concat_x3(output)
        output=torch.cat((output,out111),1)
        # print("outputccccccccat",output.shape)  
        output = self.concat_x3(output)
        for layer in self.layers3:
            
            output = layer(output)
 
        output = self.initial_block5(output)
        # print(output.shape)
        # x11 = self.initial_block6(x11)#up
        out111=self.Upsample1(x11)

        output=torch.cat((output,out111),1)
        output = self.concat_x4(output)

        #output=output+x1
        for layer in self.layers4:
            output = layer(output)
        output = self.output_conv2(output)

        return output