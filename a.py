from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def med_std(z):
    print(z)
    n=(z.shape[0]*z.shape[1])
    med=0
    for j in z:
        for k in j:
            med+=k
    med/=n
    std=0
    for j in z:
        for k in j:
            std+=(k-med)*(k-med)
    std=np.sqrt(std/n)
    return med,std


def modifi_cutoff(t,snr):
    fig, axs = plt.subplots(1, 2)
    y = np.fft.fft2(t)
    y1=abs(y)
    fre_db = deepcopy(20 * np.log10(y1 + 1))
    print(fre_db.dtype)
    med,std=med_std(fre_db)



fig, axs = plt.subplots(1,2)
X = misc.face(gray=True)
axs[0].imshow(X, cmap=plt.cm.gray)
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y)+1)
a= axs[1].imshow(freq_db)
fig.colorbar(a)
Y_c=Y.copy()
Y_c[freq_db>90]=0
print(med_std(freq_db))
print(med_std(20*np.log10(abs(Y_c)+1)))
X_c=np.fft.ifft2(Y_c)
X_c=np.real(X_c)

modifi_cutoff(X,16)