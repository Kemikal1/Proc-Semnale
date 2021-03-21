"""Ex1
Pentru calcularea DFT a unui semnal esantionat cu 44.1kHz si cu distantarea binurilor de 1 HZ este nevoie
de 22.100 sample-uri
"""

import numpy as np
import matplotlib.pyplot as plt
def Han(dim):
    x=[]
    for i in range(dim):
       x.append(0.5*(1-np.cos(2*np.pi*(i+1)/dim)))
    x=np.array(x)
    return x
def Drept(dim):
    x=[]
    for i in range(dim):
       x.append(1)
    x=np.array(x)
    return x

def Hamming(dim):
    x = []
    for i in range(dim):
        x.append(0.54-0.46*np.cos(2*np.pi*i/dim))
    x = np.array(x)
    return x

def Blackman(dim):
    x = []
    for i in range(dim):
        x.append(0.42 - 0.5 * np.cos(2 * np.pi * i / dim) +0.08*np.cos(4 * np.pi * i / dim))
    x = np.array(x)
    return x

def FlatTop(dim):
    x = []
    for i in range(dim):
        x.append(0.22 - 0.42 * np.cos(2 * np.pi * i / dim) + 0.28 * np.cos(4 * np.pi * i / dim)-0.08*np.cos(6 * np.pi * i / dim) +0.07*np.cos(8 * np.pi * i / dim))
    x = np.array(x)
    return x


if __name__ == '__main__':
    """Ex2 a +bonus """

    x = np.linspace(0, 1, 200)
    y = np.sin(200*x*np.pi)
    fig, axs = plt.subplots(6)
    print(y)
    axs[0].plot(x,y*Drept(len(y)))
    print(y)
    axs[0].plot(x,y*Han(len(y)),label='Frecventa cu Hanning')
    axs[1].plot(x,y*Hamming(len(y)),label='Frecventa cu Hamming')
    axs[2].plot(x, y * Blackman(len(y)),label='Frecventa cu Blackman')
    axs[3].plot(x, y * FlatTop(len(y)),label='Frecventa cu Flat-top')

    """Ex2 b"""
    x = np.linspace(0,8000,1000)
    x1 = np.linspace(0,1, 8000)
    y1 = np.sin(2000*np.pi*x1)
    y2 = np.sin(2200*np.pi*x1)
    X1 = abs(np.fft.fft(y1[1000:2000]*Drept(1000)))
    X2 = abs(np.fft.fft(y2[1000:2000]*Drept(1000)))
    axs[4].plot(x[:int(len(x)/2)],X1[:int(len(x)/2)],label='1000Hz')
    axs[5].plot(x[:int(len(x)/2)],X2[:int(len(x)/2)],label='1100Hz')
    for i in range(6):
        axs[i].legend()
    plt.show()
    """Pentru prima sinusoida marimea ferestrei nu este indeajuns de mare pentru a analiza componenta spectrala bine
    cee a ce duce la un varf plat al DFT,fiind nevoi de de mai mai multe cicluri ale frecventei(fereastra mia mare) sau
    analiza unei frecvente mai inalte(cazul 2)
    Frecventa 2 este un candidat mai bun pentru aplicarea acestei ferestre
    """
    """Bonusul a fost facut in lab 3"""