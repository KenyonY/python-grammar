
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
# from sklearn.cluster import k_means 
from PIL import Image
import colorsys
import glob
from tqdm import tqdm
# from ../utils import *

class MajorTones:
    
    @staticmethod
    def preProces(imgName):
        ''' 简单预处理下'''
        feature = Image.open(imgName)
        # resize一下,提高速度
        feature = feature.resize((100,100))
        feature = np.array(feature)
        # 因为有的图片色彩通道是4个
        feature = feature[:,:,:3]

        if feature.dtype == 'uint8':
            feature = feature/255.
        return feature
    
    @staticmethod
    def plot_simgle_sample(img_path, n_cls=2):
        '''测试k_means效果'''
        km = KMeans(n_clusters=n_cls)
        plt.subplot(1,3,1)
        plt.imshow(Image.open(img_path))
        # 主基调图片
        keynote  = preProces(img_path)
        plt.subplot(1,3,2)
        plt.imshow(keynote)
        h,w,channel = keynote.shape
        keynote = km.fit(keynote.reshape(h*w, channel))

        plt.subplot(1,3,3)
        plt.imshow(keynote.cluster_centers_.reshape(n_cls,1,3))
        plt.show()
        print(keynote.cluster_centers_)

    @staticmethod
    def getCenters(feature, n_cls):
        '''获得多个色彩聚类中心'''
        (h, w, channel) = feature.shape
        feature = feature.reshape(h*w, channel)
        res = KMeans(n_clusters=n_cls).fit(feature)
        return res.cluster_centers_  
        
    @staticmethod
    def getCentroid(feature, n_cls = 2):
        '''获取主色调的RGB值
        inputParam: 
        feature: 3D rgb array
        n_cls: Please keep it unchanged
        '''
        (h, w, channel) = feature.shape
        feature = feature.reshape(h*w, channel)
        res = KMeans(n_clusters=n_cls).fit(feature)
        L = len(res.labels_)

        #　选取较多的那个簇
        if L - np.count_nonzero(res.labels_) > L/2: flag = 0 
        else: flag = 1 
        feature = feature[res.labels_==flag]   
        # 进行第二次聚类
        res = KMeans(n_clusters=n_cls).fit(feature)
        L = len(res.labels_)
        if L - np.count_nonzero(res.labels_) > L/2: flag = 0
        else: flag = 1

        return res.cluster_centers_[flag]
    
    
    @staticmethod
    def get_dominant_color(image):
        '''从网上down来的,运算量较小,可能适用于某些场景,
        至于适用于哪些,我咋知道
        
        '''
        image = Image.open(image)
        #颜色模式转换，以便输出rgb颜色值
        image = image.convert('RGBA')

        #生成缩略图，减少计算量，减小cpu压力
        image.thumbnail((200, 200))

        max_score = 0
        dominant_color = 0

        for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
            # 跳过纯黑色
            if a == 0:
                continue

            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

            y = (y - 16.0) / (235 - 16)

            # 忽略高亮色
            if y > 0.9:
                continue

            # Calculate the score, preferring highly saturated colors.
            # Add 0.1 to the saturation so we don't completely ignore grayscale
            # colors by multiplying the count by zero, but still give them a low
            # weight.
            score = (saturation + 0.1) * count

            if score > max_score:
                max_score = score
                dominant_color = (r, g, b)

        return np.array(dominant_color)
    
    @staticmethod
    def process_batch_image(img_list, show_plot=0):
        '''
        input: 
        output: img_list的聚类中心列表
        '''
        Center_batch_image = []
        for i,img in enumerate(tqdm(img_list)):
            feature = MajorTones.preProces(img)
            centroid = MajorTones.getCentroid(feature)  
        #     centroid = MajorTones.get_dominant_color(img) # other
            Center_batch_image.append(centroid)
            if show_plot:
                plt.subplot(1,2,1)
                h,w,c = feature.shape
                plt.imshow(feature.reshape(h,w,c))
                plt.subplot(1,2,2)
                plt.imshow(centroid.reshape(1,1,3))
                plt.title(f'{i}')
                plt.show()
                print(60*'*')
        
        # print('processing complete')
        return Center_batch_image