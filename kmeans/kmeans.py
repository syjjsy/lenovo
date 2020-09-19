from itertools import combinations
from PIL import Image
from numpy import *
from pylab import *
from PIL import Image,ImageDraw
from PIL import Image
from numpy import *
from pylab import *
import os
 
class ClusterNode(object):
    def __init__(self,vec,left,right,distance=0.0,count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.distance = distance
        self.count = count # 只用于加权平均
    def extract_clusters(self,dist): 
        if self.distance < dist:
            return [self]
        return self.left.extract_clusters(dist) + self.right.extract_clusters(dist)
    def get_cluster_elements(self):  
        return self.left.get_cluster_elements() + self.right.get_cluster_elements()
    def get_height(self):    
        return self.left.get_height() + self.right.get_height()
    def get_depth(self):
        return max(self.left.get_depth(), self.right.get_depth()) + self.distance
    
    def draw(self,draw,x,y,s,imlist,im):
        """ 用图像缩略图递归地画出叶节点 """
        h1 = int(self.left.get_height()*20 / 2)
        h2 = int(self.right.get_height()*20 /2)
        top = y-(h1+h2)
        bottom = y+(h1+h2)
# 子节点垂直线
        draw.line((x,top+h1,x,bottom-h2),fill=(0,0,0))
    # 水平线
        ll = self.distance*s
        draw.line((x,top+h1,x+ll,top+h1),fill=(0,0,0))
        draw.line((x,bottom-h2,x+ll,bottom-h2),fill=(0,0,0))
    # 递归地画左边和右边的子节点
        self.left.draw(draw,x+ll,top+h1,s,imlist,im)
        self.right.draw(draw,x+ll,bottom-h2,s,imlist,im)
 
class ClusterLeafNode(object):
    
    def __init__(self,vec,id):
        self.vec = vec
        self.id = id
    def extract_clusters(self,dist):
        return [self]
    def get_cluster_elements(self):
        return [self.id]
    def draw(self,draw,x,y,s,imlist,im):
        nodeim = Image.open(imlist[self.id])
        nodeim.thumbnail([20,20])
        ns = nodeim.size
        im.paste(nodeim,[int(x),int(y-ns[1]//2),int(x+ns[0]),int(y+ns[1]-ns[1]//2)])
 
    def get_height(self):
        return 1
    def get_depth(self):
        return 0
    
def L2dist(v1,v2):
    return sqrt(sum((v1-v2)**2))
def L1dist(v1,v2):
    return sum(abs(v1-v2))
    
def hcluster(features,distfcn=L2dist):
    """ 用层次聚类对行特征进行聚类 """
    # 用于保存计算出的距离
    distances = {}
    # 每行初始化为一个簇
    node = [ClusterLeafNode(array(f),id=i) for i,f in enumerate(features)]
    while len(node)>1:
        closest = float('Inf')
# 遍历每对，寻找最小距离
        for ni,nj in combinations(node,2):
            if (ni,nj) not in distances:
                distances[ni,nj] = distfcn(ni.vec,nj.vec)
            d = distances[ni,nj]
            if d<closest:
                closest = d
                lowestpair = (ni,nj)
        ni,nj = lowestpair
    #  对两个簇求平均
        new_vec = (ni.vec + nj.vec) / 2.0
    #  创建新的节点
        new_node = ClusterNode(new_vec,left=ni,right=nj,distance=closest)
        node.remove(ni)
        node.remove(nj)
        node.append(new_node)
    return node[0]
    
    
def draw_dendrogram(node,imlist,filename='clusters.jpg'):
    rows = node.get_height()*20
    cols = 1200
# 距离缩放因子，以便适应图像宽度
    s = float(cols-150)/node.get_depth()
# 创建图像，并绘制对象
    im = Image.new('RGB',(cols,rows),(255,255,255))
    draw = ImageDraw.Draw(im)
# 初始化树开始的线条
    draw.line((0,rows/2,20,rows/2),fill=(0,0,0))
# 递归地画出节点
    node.draw(draw,20,(rows/2),s,imlist,im)
    im.save(filename)


path = './train'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
 
# 提取特征向量，每个颜色通道量化成 8 个小区间
features = zeros([len(imlist), 512])
for i,f in enumerate(imlist):
    im = array(Image.open(f))
# 多维直方图
    h,edges = histogramdd(im.reshape(-1,3),8,normed=True,range=[(0,255),(0,255),(0,255)])
    features[i] = h.flatten()
tree = hcluster.hcluster(features)
 
 
hcluster.draw_dendrogram(tree,imlist,filename='sunset.pdf')
