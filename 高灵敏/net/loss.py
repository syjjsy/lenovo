import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.98, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w

        self.mse_loss = nn.MSELoss(size_average=True)
    

    def forward(self, disps1,bg):

        
        bg_image=bg
        left=disps1
        loss = self.mse_loss(left,bg_image)
        # loss1=bg_image-left
        # loss=loss1*loss1
        # loss=torch.mean(loss)
        
        return loss