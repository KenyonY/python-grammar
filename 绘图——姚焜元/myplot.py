import matplotlib.pyplot as plt
import numpy as np


class MyPlot:
    @staticmethod
    def yuan(R):
        theta = np.linspace(0,2*np.pi,100)
        x = R*np.cos(theta)
        y = R*np.sin(theta)
        plt.figure(figsize=(5,5))
        plt.plot(x,y,'-o')
        plt.axis('equal')
        plt.axis('off')
    
    @staticmethod
    def fang(L):
        plt.figure(figsize=(4,4))
        x = [0, 0, L, L, 0]
        y = [0, L, L, 0, 0]
        plt.plot(x, y, c='y')
        plt.axis('equal')
        plt.axis('off')

    
if __name__ == "__main__":
    myplot = MyPlot()
    myplot.yuan(5)
    myplot.fang(5)
    plt.show()
