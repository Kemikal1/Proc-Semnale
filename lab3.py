"""1.Frecventa minima de esantionare  pentru contrabass este de 400hz
   2.Folosim  (2fc + B)/(m + 1)
    a)95hz
    b)63,3Hz
    c)38Hz


"""
"""ex 3"""
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
if __name__ == '__main__':
    with open('trafic.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    x = np.linspace(0, 1, 1000)
    x1 = np.linspace(0,1000,1000)
    x2 = np.linspace(24,96,72)
    y= np.sin(20*x*np.pi)
    X=np.fft.fft(y)
    print(len(X))
    X=X[:int(len(X)/2)]
    x1=x1[:int(len(x1)/2)]
    fig, axs = plt.subplots(4)
    axs[0].plot(x,y)
    axs[1].plot(x1,X)



    """ex4"""
    data.pop(0)
    data=[ int(i[0]) for i in data]
    dataT = data
    data = data[24:96]

    w1=3
    w2=7
    data=np.array(data)
    data1=np.convolve(data,np.ones(w1),'valid')/w1
    data2=np.convolve(data,np.ones(w2),'valid')/w2
    X = np.fft.fft(dataT)


    axs[2].plot(x2,data,label='semnal original')
    axs[2].plot(x2[int(w1/2):len(data1)+int(w1/2)], data1,label='medie alunecatoare de marime 3')
    axs[2].plot(x2[int(w2/2):len(data2)+int(w2/2)], data2,label='medie alunecatoare de marime 7')
    axs[2].legend()
    X = X[:int(len(X) / 2)]
    x3=np.linspace(0,int(len(dataT)),int(len(dataT)))
    x3=x3[:int(len(x3)/2)]
    dataT=dataT[:int(len(dataT)/2)]
    axs[3].plot(x3,X)
    print(len(dataT))
    plt.show()

