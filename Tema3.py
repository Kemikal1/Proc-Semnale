import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn




X = misc.ascent()
plt.imshow(X, cmap=plt.cm.gray)
plt.show()


Y1 = dctn(X, type=1)
Y2 = dctn(X, type=2)
Y3 = dctn(X, type=3)
Y4 = dctn(X, type=4)
freq_db_1 = 20*np.log10(abs(Y1))
freq_db_2 = 20*np.log10(abs(Y2))
freq_db_3 = 20*np.log10(abs(Y3))
freq_db_4 = 20*np.log10(abs(Y4))

plt.subplot(221).imshow(freq_db_1)
plt.subplot(222).imshow(freq_db_2)
plt.subplot(223).imshow(freq_db_3)
plt.subplot(224).imshow(freq_db_4)
plt.show()


k = 120

Y_ziped = Y2.copy()
Y_ziped[k:] = 0
X_ziped = idctn(Y_ziped)

plt.imshow(X_ziped, cmap=plt.cm.gray)
plt.show()


Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down);

plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('Down-sampled')
plt.show()



Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# Encoding
x = X[:8, :8]
y = dctn(x)
y_jpeg = Q_jpeg*np.round(y/Q_jpeg)

# Decoding
x_jpeg = idctn(y_jpeg)

# Results
y_nnz = np.count_nonzero(y)
y_jpeg_nnz = np.count_nonzero(y_jpeg)

plt.subplot(121).imshow(x, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
plt.title('JPEG')
#plt.show()

print('Componente în frecvență:' + str(y_nnz) +
      '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

"""
Sarcina 1
"""


def RGBtoYCbCr(R,G,B):
    Y=0.299*R+0.587*G+0.114*B
    Cb=128-0.168736*R - 0.331264*G + 0.5*B
    Cr=128+0.5*R - 0.418688*G-0.081312*B
    return Y,Cb,Cr

def YCbCrtoRGB(y,cb,cr):
    cb -= 128
    cr -= 128
    r = y + 1.402 * (cr)
    g = y - 0.344136 * (cb) - 0.714136 * cr
    b = y + 1.772 * cb
    return r,g,b

"""Functie de conversie in Jpg"""
def convJPG(im,Q):
    # Separarea canaleleor RGB
    r=im[:,:,0]
    g=im[:,:,1]
    b=im[:,:,2]

    #Trecerea din RGB in YCbCr
    y,cb,cr=RGBtoYCbCr(r,g,b)
    for i in range(1,im.shape[0]):
        for j in range(1,im.shape[1]):
            if i%8==0 and j%8==0:
                xy = y[i - 8:i, j - 8:j]
                xcb = cb[i - 8:i, j - 8:j]
                xcr = cr[i - 8:i, j - 8:j]
                x = [xy, xcb, xcr]
                for k in range(len(x)):
                    yt=dctn(x[k])
                    yjpeg=Q*(np.round(yt/Q))
                    xjpeg=idctn(yjpeg)
                    x[k]=xjpeg
                y[i - 8:i, j - 8:j] = x[0]
                cb[i - 8:i, j - 8:j] = x[1]
                cr[i - 8:i, j - 8:j] = x[2]
    imr=im.copy()

    # Trecerea din YCrCb in RGB
    r,g,b=YCbCrtoRGB(y,cb,cr)

    imr[:, :, 0]=r
    imr[:, :, 1]=g
    imr[:, :, 2]=b
    return imr
#compresia canalului
def comp_ch(ch,f):
    xch=dctn(ch)
    xch[f:]=0
    return idctn(xch)

#
def compress(im,f):
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    y, cb, cr = RGBtoYCbCr(r, g, b)

    r=comp_ch(r,f)
    g=comp_ch(g,f)
    b=comp_ch(b,f)


    y=comp_ch(y,f)
    cb=comp_ch(cb,f)
    cr=comp_ch(cr,f)

    #r,g,b=YCbCrtoRGB(y,cb,cr)
    imr=im.copy()
    imr[:, :, 0] = r
    imr[:, :, 1] = g
    imr[:, :, 2] = b
    return imr


plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')

for i in range(1,(X.shape[0])):
    for j in range (1,X.shape[1]):
        if i%8==0 and j%8==0:
            x=X[i-8:i,j-8:j]
            y=dctn(x)
            y_jpeg=Q_jpeg*np.round(y/Q_jpeg)
            x_jpeg=idctn(y_jpeg)
            X[i-8:i,j-8:j]=x_jpeg

plt.subplot(122).imshow(X, cmap=plt.cm.gray)
plt.title('Jpeg')
#plt.show()

"""

Sarcina 2

"""
X=misc.face()
X=X.copy()

#plt.subplot(121).imshow(X)

#Factorul de multiplicare a matricei jpg pentru compresie mai mare
F=10
Q_jpeg=F*np.array(Q_jpeg)
Q_jpeg=Q_jpeg.tolist()

#convertim la JPG
X_jpg=convJPG(X,Q_jpeg)
#plt.subplot(122).imshow(X_jpg)
#plt.show()

"""
Sarcina 3
"""



"""
Calculam mse-ul intre canalul Y ale celor doua imagini 
"""
def mse(im1,im2):
    s=0
    im1_y=RGBtoYCbCr(im1[:,:,0],im1[:,:,1],im1[:,:,2])[0]
    im2_y=RGBtoYCbCr(im2[:,:,0],im2[:,:,1],im2[:,:,2])[0]
    for i in range(im1.shape[0]):
        for j in range (im1.shape[1]):
            s+=(im1_y[i][j]-im2_y[i][j])**2
    s/=im1.shape[0]*im1.shape[1]
    return s

X_jpg=compress(X,60)
plt.imshow(X_jpg)
plt.show()

F=1
Q_jpeg=F*np.array(Q_jpeg)
Q_jpeg=Q_jpeg.tolist()
MSE=0
print(MSE)
print("Mse dorit")
MSE_input=input()
MSE_input=int(MSE_input)
F=255

while MSE<MSE_input:
    F-=10
    Q_jpeg1 = F * np.array(Q_jpeg)
    Q_jpeg1 = Q_jpeg1.tolist()
    X_jpg=compress(X,F)
    print(MSE)
    MSE = mse(X, X_jpg)
    MSE = np.round(MSE)
fig,axs=plt.subplots(1)
axs.imshow(X_jpg)
plt.title("MSE "+str(MSE))
plt.show()

"""
Sarcina 4
"""