import numpy as np
import matplotlib.pyplot as plt
import csv
from copy import deepcopy
import scipy.signal

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



with open('trafic.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
"""EX 1"""
x = np.linspace(0, 100, 100)
fig, axs = plt.subplots(5)
axs[0].plot(x[:int(len(x)/2)],20*np.log10 (np.fft.fft(Drept(len(x))))[:int(len(x)/2)],label='DFT Drept')
axs[1].plot(x[:int(len(x)/2)], np.fft.fft(Han(len(x)))[:int(len(x)/2)],label='DFT Hanning')
axs[2].plot(x[:int(len(x)/2)], np.fft.fft(Hamming(len(x)))[:int(len(x)/2)],label='DFT  Hamming')
axs[3].plot(x[:int(len(x)/2)], np.fft.fft(Blackman(len(x)))[:int(len(x)/2)],label='DFT  Blackman')
axs[4].plot(x[:int(len(x)/2)], np.fft.fft(FlatTop(len(x)))[:int(len(x)/2)],label='DFT Flat-top')
for i in range(5):
    axs[i].legend()
print(20*np.log10 (np.fft.fft(Drept(len(x))))[:int(len(x)/2)])

"""EX 2"""

#A
fig1,axs=plt.subplots(1)
data.pop(0)
data=[ int(i[0]) for i in data]
data=np.array(data)
x=np.linspace(0,int(len(data)),int(len(data)))
dataT=abs(np.fft.fft(data))
axs.plot(x[:int(len(x)/2)],dataT[:int(len(x)/2)])
axs.legend()

#B
print(len(data))
"""Alegem frecventa de 50 deoarece majoritatea zgomotului este de peste 50Hz si este 0.2 din frecventa Nyquist """


#C
fig2,axs=plt.subplots(2)
b_but,a_but=scipy.signal.butter(5,0.2,btype='low',output='ba')
b_cheb,a_cheb=scipy.signal.cheby1(5,5,0.2,btype='low',output='ba')
w_but,h_but=scipy.signal.freqz(b_but,a_but)
w_cheb,h_cheb=scipy.signal.freqz(b_cheb,a_cheb)
#D
axs[0].plot(w_but,20*np.log10(h_but),label='Butterworth')
axs[1].plot(w_cheb,20*np.log10(h_cheb),label='Chebyshev')
for i in range(2):
    axs[i].legend()
#E
fig3,axs=plt.subplots(3)

x_but=scipy.signal.filtfilt(b_but,a_but,data)
x_cheb=scipy.signal.filtfilt(b_cheb,a_cheb,data)
axs[0].plot(x,x_but,label='Filtru Butterworth')
axs[1].plot(x,x_cheb,label=' Filtru Chebyshev')
axs[2].plot(x,data,label='Brut')
plt.legend()

for i in range(3):
    axs[i].legend()
#Butterworth e mai bun in cazul acesta deoarece pastreaza amplitudinile mai bine fata de Chebyshev
#F

fig4,axs=plt.subplots(3)
wn=0.2
N=3
rp=5
plt.title('Butterworth cu diferite ordine')

for i in range(3):
    b_but, a_but = scipy.signal.butter(N+i*2, wn, btype='low', output='ba')
    x_but = scipy.signal.filtfilt(b_but, a_but, data)
    axs[i].plot(x,x_but,label='Filtru Butterworth N='+str(N+i*2))
    axs[i].legend()
plt.legend()



fig5,axs=plt.subplots(3)
plt.title('Chebyshev cu diferite ordine')
for i in range(3):
    b_cheb, a_cheb = scipy.signal.cheby1(N+i*2, rp, wn, btype='low', output='ba')
    x_cheb = scipy.signal.filtfilt(b_cheb, a_cheb, data)
    axs[i].plot(x,x_cheb,label=' Filtru Chebyshev  N='+str(N+i*2))
    axs[i].legend()

N=5
rp=3
fig6,axs=plt.subplots(3)
plt.title('Chebyshev cu atenuarea diferita')
for i in range(3):
    b_cheb, a_cheb = scipy.signal.cheby1(N, rp+i*2, wn, btype='low', output='ba')
    x_cheb = scipy.signal.filtfilt(b_cheb, a_cheb, data)
    axs[i].plot(x,x_cheb,label=' Filtru Chebyshev rp='+str(rp+i*2))
    axs[i].legend()

plt.show()

#Cei mai buni parametri pentru chebyshev sunt rp=5 si N=7 deoarece atenuarea este mai putin evidenta
#Cel mai bun parametru epntru butterworth este N=5