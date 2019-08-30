import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
# from sklearn.cluster import k_means 
from PIL import Image
import colorsys
import glob

'''
vec1 = np.array([1,1,1]).reshape(3,1)
vec2 = np.array([2,3,4]).reshape(3,1)
'''

class Dist:
    
    '''input: vec1 , vec2 均为列向量'''
    
    @staticmethod
    def euclidean_dist(vec1, vec2):
        """欧氏距离:
        我们现实所说的直线距离"""
        assert vec1.shape == vec2.shape 
        return np.sqrt((vec2-vec1).T @ (vec2-vec1))
    
    @staticmethod
    def manhattan_dist(vec1, vec2):
        """曼哈顿距离:
        城市距离"""
        return sum(abs(vec1 - vec2))
    
    @staticmethod
    def chebyshev_dist(vec1, vec2):
        """切比雪夫距离:
        国际象棋距离
        """
        return abs(vec1 - vec2).max()
    @staticmethod
    def minkowski_dist(vec1, vec2, p=2):
        """闵可夫斯基距离:
        应该说它其实是一组距离的定义: 
        inputParam: p
        return distance,
        while p=1: dist = manhattan_dist
        while p=2: dist = euclidean_dist
        while p=inf: dist = chebyshev_dist
        """
        s = np.sum(np.power(vec2 - vec1, p))
        return np.power(s,1/p)
    
    @staticmethod
    def cosine_dist(vec1, vec2):
        """夹角余弦"""
        # np.linalg.norm(vec, ord=1) 计算p=1范数,默认p=2
        return (vec1.T @ vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    @staticmethod
    def hamming_dist():
        pass
    
    @staticmethod
    def jaccard_simil_coef():
        pass
        


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

#         print('label总数是',len(res.labels_))
#         print('label = 1的数目是',np.count_nonzero(res.labels_))
#         print('flag是',flag)
        # 返回数目角度的那个簇的centroid
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
    

        
