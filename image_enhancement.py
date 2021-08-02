
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
########## basic grayscale transformation #############
def grayscale(im):
    linear = 2*im
    linear[linear>255] = 255
    cv2.imshow('linear transform',linear)
 
    im = im / 255
    gamma = 0.4
    power = np.power(im,gamma)
    cv2.imshow('power transformation',power)
 
########### basic HSV transformation ############
def HSVtrans(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    turn_green = im_hsv.copy()
    turn_green[:,:,0] = (turn_green[:,:,0] + 30) % 180
    turn_green = cv2.cvtColor(turn_green, cv2.COLOR_HSV2BGR)
    cv2.imshow('change hue', turn_green)
 
    turn_gray = im_hsv.copy()
    turn_gray[:,:,1] = 1/2 * turn_gray[:,:,1]
    turn_gray = cv2.cvtColor(turn_gray, cv2.COLOR_HSV2BGR)
    cv2.imshow('change saturation', turn_gray)
 
    turn_dark = im_hsv.copy()
    turn_dark[:,:,2] = 1/2 * turn_dark[:,:,2]
    turn_dark = cv2.cvtColor(turn_dark, cv2.COLOR_HSV2BGR)
    cv2.imshow('change value', turn_dark)
 
 
############# histogram ##############
def calchist(im):
    h,w = im.shape[:2]
    im2 = im.reshape(h*w,-1)[:,0]
    histogram, _, _ = plt.hist(im2,256,facecolor='black')
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
 
def histreg(im):
    im = im[:,:,0] #Select Red Channel
    Imin,Imax = cv2.minMaxLoc(im)[:2]
    Omin,Omax = 0,255
    a = (Omax-Omin)/(Imax-Imin)
    b = Omin - a*Imin
    out = a*im + b
    out = out.astype(np.uint8)
    plt.figure(1)
    calchist(im)
    plt.figure(2)
    calchist(out)
 
########## mean filter #################
def mean_fil(im, x, y, k_size):
    average = 0 
    for m in range(-int(k_size/2), int(k_size/2)):
        for n in range(-int(k_size/2), int(k_size/2)):
            average += im[x+m][y+n]/(k_size*k_size)
    return average
 
########## median filter #################
def median_fil(im, x, y, k_size):
    median = []
    for m in range(-int(k_size/2), int(k_size/2)+1):
        for n in range(-int(k_size/2), int(k_size/2)+1):
            median.append(im[x+m][y+n])
    median.sort()
    return median[int(k_size*k_size/2)]
 
############ laplace filter ###############
def laplace(im,c):
    filter = np.array([
        [1,1,1],
        [1,-8,1],
        [1,1,1],
    ])
 
    h,w = im.shape[:2]
    imout = im.copy()
    for m in range(1,h-1):
        for n in range(1,w-1):
            grad = np.sum(filter*im[m-1:m+2, n-1:n+2])
            imout[m,n] = im[m,n] + c*grad
    return imout[1:h-1,1:w-1]
 
im = cv2.imread('lena.jpg')
if im is None:
    print('Fail to open the image.')
    exit()
 
 
## basic image processing ##
grayscale(im)          
histreg(im)            
HSVtrans(im)           
 
 
############### mean filter & median filter ##############
imcopy1 = im.copy()
ksize1 = 3  #kernel size
for i in range(int(ksize1/2), im.shape[0]-int(ksize1/2)):
    for j in range(int(ksize1/2), im.shape[1]-int(ksize1/2)):
        imcopy1[i,j] = mean_fil(im,i,j,ksize1)
cv2.imshow('mean filter',imcopy1)
 
imcopy2 = im.copy()
ksize2 = 3  #kernel size
for k in range(3):
    for i in range(int(ksize2/2), im.shape[0]-int(ksize2/2)):
        for j in range(int(ksize2/2), im.shape[1]-int(ksize2/2)):
            imcopy2[i,j,k] = median_fil(im[:,:,k],i,j,ksize2)
cv2.imshow('median filter',imcopy2)
 
 
################ laplace filter ###################
cv2.imshow('laplace filter', laplace(im,0.1))
 
 
################ morphological transformation #################
open = cv2.morphologyEx(im,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
cv2.imshow('opening operation', open)
 
 
################ corner detection ################
imcopy3 = im.copy()
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
corner = cv2.cornerHarris(imgray, 2, 3, 0.04)
 
imcopy3[corner>0.01*np.max(corner)] = [0,0,255]
cv2.imshow('corner detection', imcopy3)
 
 
############## image pyramid ###############
imup = cv2.pyrUp(im)
imdown = cv2.pyrDown(im)
imdd = cv2.pyrDown(imdown)
cv2.imshow('image upsampling', imup)
cv2.imshow('image downsampling', imdown)
cv2.imshow('image down sampling again', imdd)
cv2.waitKey()
 
cv2.destroyAllWindows()
