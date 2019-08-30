import numpy as np 

def gaussian(x,sigma,miu):
    """
    标准高斯函数
    x可以是标量或者矢量
    """
    a = 1/np.sqrt(2*np.pi*sigma**2)
    return a* np.exp(-(x-miu)**2/(2*sigma**2))

def gauss2d(U, V, sigma, miu):
	'''
	This function will map x to Gaussian curve.
	input: 
		x is a 2D numpy Array. 
		sigma and miu are be used to discribe the shape of Normal distribution.
	output:
		The map of x
	'''
	a = 1/np.sqrt(2*np.pi*sigma**2) 
	res_max = a* np.exp(-((0-miu)**2+(0-miu)**2)/(2*sigma**2))#For normalization
	res = a* np.exp(-((U-miu)**2+(V-miu)**2)/(2*sigma**2))
	return res/res_max # Normalized

def butter(U,V,n,m):
	'''
	input:
		n the order of butterwith. range: 1~10
		m range: 1~100
	'''
	return 1/(1+(np.sqrt(U**2+V**2)/m)**(2*n))

def sawtooth(n,x):
    '''
    锯齿函数sawtooth的傅里叶级数为 sin(nx)/n ,n从一累加到无穷 
    '''
    S=np.zeros(x.shape)
    for i in range(1,n+1):
        s=np.sin(i*x)/i
        S+=s

    return S

def MaxMinNormal(I,out_min, out_max):
    '''
    input: 
    I: this vector to be scaled
    out_min : the minimun of out vector
    out_max : the maximun of out vector
    '''
    Imax = I.max()
    Imin = I.min()
    out = out_min + (out_max - out_min)/(Imax - Imin) * (I-Imin)
    return out.astype('uint8')

# 伽马变换
def Gamma_trans(I, I_max, gamma):
    '''
    gamma: if your intersted region is too bright, set gamma > 1 decreasing contrast.
    and if your intersted region is too bright dark, set 1> gamma > 0 to increase contrast.
    I_max: is the maximun of the channel of I.
    '''
    fI = I/I_max
    out = np.power(fI, gamma)
    out = out*I_max
    return out.astype("uint8")