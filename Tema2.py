from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

"""Functie de calcularea a deviatiei standard si mediei a unei imagini in cazult transformatelor cu valori extreme"""
def med_std(z):
    n=(z.shape[0]*z.shape[1])
    med_c=0
    for j in z:
        for k in j:
            med_c+=k
    med_c/=n

    std_x=0
    for j in z:
        for k in j:
            std_x+=(k-med_c)*(k-med_c)
    std_x=np.sqrt(std_x/n)
    return med_c,std_x


"""Functie ce calculeaza mean squared error"""
def mse(o,t):
    s=0
    for i in range(len(o)):
        for j in range(len(o[0])):
            s+=(o[i][j]-t[i][j])*(o[i][j]-t[i][j])

    return s/len(o)/len(o[0])


"""Functie de afisarea a maginilor si transformata ei normalizata cu log10  cu outlierii scosi"""
def imspec(i):

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(i)
    Y = np.fft.fft2(i)
    Y=abs(Y)
    n=(i.shape[0]*i.shape[1])

    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if(Y[i][j]==0):
                Y[i][j]=1
    freq_db = 20 * np.log10(Y)
    med_a, std_a = med_std(freq_db)
    print(med_a,std_a)
    # Scoaterea valorilor extreme
    err=1.5*std_a # erroare cu care acceptam valorile extreme
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if freq_db[i][j]>med_a+err:
                freq_db[i][j]=med_a+err
            if freq_db[i][j] < med_a - err:
                freq_db[i][j] = med_a - err
    a=axs[1].imshow(freq_db[20:][20:])
    fig.colorbar(a)

"""Functie de afisarea a imaginilor si transformatei sale normalizata"""
def imspec1(i):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(i)
    Y = np.fft.fft2(i)
    Y = abs(Y)
    med = 0
    n = (i.shape[0] * i.shape[1])
    freq_db = (Y)
    a = axs[1].imshow(freq_db)
    fig.colorbar(a)


def modifi_cutoff(o,t,snr):
    fig, axs = plt.subplots(1, 2)
    y = np.fft.fft2(t)
    fre_db = 20 * np.log10(abs(y) + 1)
    snr_cur=20*np.log10(255)-10*np.log10(1)
    snr_cur=abs(snr_cur)
    m=0
    for k in fre_db:
        for j in k:
            if m< j:
                m=j  #Frecventa de cutoff porneste de la cea mai amre frecventa
    m-=10       #Timpul de cautare ia prea mult... iar de la max la max-10 cutoff SNR nus e modifica
    x=[]
    #Scadem din frecventa de cutoff pana obtiner SNR aproximativ dorit,in unele cazuri nu exista un anume SNR cautat asa ca
    #Ne multumim cu ca mia apropiata valoare mai mica decat cea ceruta....
    while(snr_cur>snr):
        y_c=y.copy()
        y_c[fre_db>m]=0
        x=np.fft.ifft2(y_c)
        x=np.real(x)
        x-=np.amin(x)
        max = np.amax(x)
        x=x/max*255
        mse_t=mse(o,x)
        snr_cur = 20*np.log10(255)-10*np.log10(mse_t+1)
        snr_cur = np.floor((snr_cur))
        m-=1
    print(snr_cur)
    axs[0].imshow(x,cmap=plt.cm.gray)
    a= axs[1].imshow(fre_db)
    fig.colorbar(a)
    plt.show()







fig, axs = plt.subplots(1,2)
X = misc.face(gray=True)
axs[0].imshow(X, cmap=plt.cm.gray)
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y)+1)
a= axs[1].imshow(freq_db)
fig.colorbar(a)
Y_c=Y.copy()
Y_c[freq_db>90]=0
plt.show()





"""
freq_cutoff = 120
Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2

x=X_cutoff
x-=np.amin(x)
x=x/np.amax(x)*255
mse_j=mse(X,x)
print(20*np.log10(255)-10*np.log10(mse_j),mse_j)
plt.imshow(x, cmap=plt.cm.gray)
plt.show()
X_c=np.fft.ifft2(Y_c)
X_c=np.real(X_c)

"""

"""Sarcina 1"""
"""subpunctul a"""
print(X.shape)
Xc=[]
for i in range(5):
    Xc.append(deepcopy(X))
    Xc[i]=np.ndarray(shape=(X.shape[0],X.shape[1]),dtype=float)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x=(((np.sin(2*np.pi*i/15+2*np.pi*j/15))))
        Xc[0][i][j]=x


imspec(Xc[0])
""" subpunctul b"""

Xc[1]=np.ndarray(shape=(X.shape[0],X.shape[1]),dtype=float)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x=(np.sin(4*np.pi*i/40)+np.cos(6*np.pi*j/40))
        Xc[1][i][j]=x


imspec(Xc[1])
n=X.shape[1]*X.shape[0]
"""subpunctul c"""
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if (i==0 and j==5) or (i==0 and j==X.shape[1]-5):
            x=1
        else:
            x=0
        Xc[2][i][j]=x
Xc[2]=np.fft.ifft2(Xc[2])
imspec(abs(Xc[2]))


"""subpunctul d"""
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if (i==5 and j==0) or (j==0 and i==X.shape[0]-5):
            x=1
        else:
            x=0
        Xc[3][i][j]=x
Xc[3]=np.fft.ifft2(Xc[3])
imspec(abs(Xc[3]))

"""subpunctul e"""

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if (i==5 and j==5) or (i==X.shape[0]-5 and j==X.shape[1]-5):
            x=1
        else:
            x=0
        Xc[4][i][j]=x

Xct=np.fft.ifft2(Xc[4])

imspec1(abs(Xct))
plt.show()

"""Sarcina 2"""

modifi_cutoff(X,X,23)

"""Sarcina 3"""


med,std=med_std(X)

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

X_n=[]
X_avg=X_noisy.copy()
for i in range(100):                      #Signal averaging pentru a scapa de zgomot in cazul asta cu 101 de imagini cu zgomot aleator
    noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=X.shape)
    X_n.append(X+noise)

for i in X_n:
    X_avg+=i

X_avg=X_avg/(len(X_n)+1)    #Facem media celor 101 de imagini cu zgomot


X_noisy-=np.amin(X_noisy)
X_avg-=np.amin(X_avg)
X_noisy=X_noisy/np.amax(X_noisy)*255
X_avg=X_avg/np.amax(X_avg)*255
#Normalizam imaginile la intervalul [0,255] Pentru a calcula SNR fata de imaginea oriignala


Y_noisy= 20 * np.log10(abs(np.fft.fft2(X_noisy))+1)
Y_avg=20 * np.log10(abs(np.fft.fft2(X_avg))+1)


#Pentru SNR am folosit formula de 20*log10(VAl max pixel)-10*log10(MSE)
#MSE = mean square error dintre imaginea originala(fara zgomot) si cea comparata

med_noisy,std_noisy=med_std(X_noisy)
mse_noisy=mse(X,X_noisy)
snr=20*np.log10(255)-10*np.log10(mse_noisy)
print(med_noisy,std_noisy,snr)      #Snr pentru imaginea zgomotaosa

mse_avg=mse(X,X_avg)
med_avg,std_avg=med_std(X_avg)
snr1=20*np.log10(255)-10*np.log10(mse_avg)
print(med_avg,std_avg,snr)      #Snr pentru imaginea cu zgomotul redus


plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy, Snr='+str(np.floor(snr)))
plt.figure()
plt.imshow(X_avg, cmap=plt.cm.gray)
plt.title('Denoised , Snr='+str(np.floor(snr1)))
plt.show()

"""Sarcina 4"""


