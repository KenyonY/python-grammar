# 指定CPU跑
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


r = 10
theta = tf.linspace(0, 2*np.pi, 100)
y = tf.sin(theta)
x = tf.cos(theta)
plt.plot(x,y)
plt.show()