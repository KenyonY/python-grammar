import numpy as np 
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
        
def is_similarity(vec1, vec2, dist_choise=2, tolerance=0.3):
    '''inputParam:
    dist_choise: 选择哪种距离, 1:P1范数(曼哈顿距离), 2: P2范数(欧氏距离), 3: P_inf范数(切比雪夫距离)
    tolerance: 相似度小于该值时会被判断为同类别,并 return True
    '''
    if dist_choise == 1:
        distance = Dist.manhattan_dist
    elif dist_choise == 2:
        distance = Dist.euclidean_dist
    elif dist_choise == 3:
        distance = Dist.chebyshev_dist
    else:
        raise 'dist_choise is a bad number'
        
    # 判断相似度:
    if distance(vec1, vec2) < tolerance:
        return True
    else:
        return False

def find_simil_idx(Vec_list, VEC,dist_choise=2, tolerance=0.3):
    '''
    Vec_list中每一个向量将与VEC向量比较, 最后返回所有与VEC相似向量下标列表
    '''
    IDX = []
    for idx, vec in enumerate(Vec_list):
        if is_similarity(vec, VEC,dist_choise, tolerance):
            IDX.append(idx)
    return IDX   

        
