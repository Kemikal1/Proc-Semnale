import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wv
import scipy.signal as sig

if __name__ == '__main__':
    n=0.03
    x=np.linspace(0,n,300)
    x2=np.linspace(0,1,300)

    y1=np.cos(520*np.pi*x+np.pi/3)
    y2=np.cos(280*np.pi*x-np.pi/3)
    y3=np.cos(120*np.pi*x+np.pi/3)
    y4 = np.cos(200 * np.pi * x  + np.pi / 3)
    fig,axs =plt.subplots(6)
    axs[0].plot(x,y1)
    axs[1].plot(x, y2)
    axs[2].plot(x, y3)
    #axs[3].plot(x, y4)
    #t=(2*np.pi)/200
    #print(t)
    fn=200
    m=math.floor(np.floor(fn*n))
    print(m)
    x1 = np.linspace(0, n, m+1)
    x=x1
    y1 = np.cos(520 * np.pi * x  + np.pi / 3)
    y2 = np.cos(280 * np.pi * x  - np.pi / 3)
    y3 = np.cos(120 * np.pi * x  + np.pi / 3)
    axs[3].plot(x1, y1)
    axs[4].plot(x1, y2)
    axs[5].plot(x1, y3)
    plt.show()



#Exercitiul 1
'''
a)1/2000=0.0005s
b)4biti=0.5 bytes=>0.5*2000*3600=3.6MB


'''
