import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from celluloid import Camera

def func(k):
    a=2
    t=np.pi
    k=k-int(k/t)*t
    return a/t*k

def fourier_aprox(c,x,per):
    s=c[0]/2
    for i in range(1,len(c)):
       s+=2*c[i]*np.exp((1j)*2*np.pi*x*(i)/per)#+np.conj(c[i])*np.exp((1j)*2*np.pi*x*(-i)/per)
    return s

def fourier(index_n,dft,dom,per):
    c=[]
    for i in range(0,index_n+1):
       c.append(dft[i]/dom.size)
    fou_v=[]
    print(c)
    for i in dom:
        fou_v.append(fourier_aprox(c,i,per))
    return fou_v

#Obtinem coeficientii aproximarii fourier prin suma reimann

def coef(index_n,y_s,dom,per):
    c=[]
    for i in range(index_n+1):
        s=y_s*np.exp(-1j*2*np.pi*dom*i/per)
        s=np.array(s)
        c.append(s.sum()/s.size)
    return c
#Obtinem aproximarea fourier intrun  punct
def fourier1(x,co,per):
    s=0
    for i in range(len(co)):
        s+=2*co[i]*np.exp(1j*2*np.pi*(i+1)*x/per)
    return s

fig,axs=plt.subplots(2)
x=np.linspace(0,4*np.pi,1000)
f_v=np.vectorize(func)
y=f_v(x)

"""
x_d=np.linspace(0,100,1000)
y_d=np.fft.fft(y)
y_f=fourier(2,(y_d),x,np.pi)
y_f=np.array(y_f)
"""


#Coeficientii aproximarii
#N=10
N=10
coefic=np.array(coef(N,y,x,np.pi))
print(coefic)
#Obtinem aproximarea pentru tot domeniul
y333=fourier1(x,coefic,np.pi)





axs[0].plot(x,y)
axs[1].plot(x,abs(y333))
plt.show()

t=np.linspace(0,np.pi,150)
fig,axs=plt.subplots(2)
camera=Camera(fig)
for i in t:
    #plotul pentru functia normala
    f_v = np.vectorize(func)
    y = f_v(x+i)
    axs[0].plot(x,y,color='red')

    #plotul pentru functia aproximata fourier
    coefic = np.array(coef(10, y, x+i, np.pi))
    y333 = fourier1(x+i, coefic, np.pi)
    axs[1].plot(x,abs(y333),color='red')
    camera.snap()
anim=camera.animate(interval=20, blit=True)
plt.show()