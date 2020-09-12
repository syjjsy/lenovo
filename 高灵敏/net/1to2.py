import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class OneToTwo:
    def __init__(self,syj):
        self.syj=syj
    

    def Ott(self,left):
        disps1=self.model(left).np()
        


