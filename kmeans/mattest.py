import torch
import numpy as np
from itertools import combinations
from PIL import Image
from numpy import *
from pylab import *
from PIL import Image,ImageDraw
from PIL import Image
from numpy import *
from pylab import *
import os
import scipy.io as io
# a=torch.ones(256,256)
# b=torch.zeros(256,256)
# c=b-a
# d=c.data.numpy()
# print(d)
mat_path="C:\\Users\\20157\\Pictures\\1.bmp"
# mat = np.zeros([4, 20])
pic=Image.open(mat_path)
# pic.label=1
c=pic.shape()

# io.savemat(mat_path, {'d':d})