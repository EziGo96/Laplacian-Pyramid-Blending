'''
Created on 11-Dec-2022

@author: EZIGO
'''
import cv2
import numpy as np
from sroy22_proj03.Conv2 import conv2 as conv

def lPyr_upsampler(img,n_img):
    old_r = img.shape[0]
    old_c = img.shape[1]
    new_r = n_img.shape[0]
    new_c = n_img.shape[1]
    r_r = new_r/old_r
    r_c = new_c/old_c
    R = (np.floor((np.arange(0,new_r,1))/r_r)).astype('int16')
    C = (np.floor((np.arange(0,new_c,1))/r_c)).astype('int16')
    uimg = img[R,:]
    uimg = uimg[:,C]
    c_lap = conv(uimg,k,"zero_pad")
    return c_lap


def gPyr_downsampler(img):
    c_gaus = conv(img,k,"zero_pad")
    old_r = c_gaus.shape[0]
    old_c = c_gaus.shape[1]
    new_r = (np.floor(c_gaus.shape[0]/ 2)).astype('int16')
    new_c = (np.floor(c_gaus.shape[1]/ 2)).astype('int16')
    
    r_r = new_r/old_r
    r_c = new_c/old_c
    
    R = (np.floor((np.arange(0,new_r,1))/r_r)).astype('int16')
    C = (np.floor((np.arange(0,new_c,1))/r_c)).astype('int16')
    
    dimg = c_gaus[R,:]
    dimg = dimg[:,C]
    return dimg

def ComputePyr(img,num_layers):
    global n_l, k
    n_l = 1
    shape = img.shape[0]
    for i in range(num_layers-1):
        if (shape//2 < 5):
            print("Warning! Max layers for given input is  %d " %n_l)
            num_layers = n_l
            break
        else:
            shape = shape //2
            n_l += 1
     
    #Gaussian Kernel
    x = cv2.getGaussianKernel(5,2)
    k = x*x.T
     
    #Gaussian Pyramid
    G = img.copy().astype('float32')
    gPyr = [G]
    for i in range(num_layers-1):
        G =  gPyr_downsampler(G)
        gPyr.append(G)

    #Laplacian Pyramid
    lPyr = [gPyr[num_layers-1]]
    for i in range(num_layers-1,0,-1):
        GE = lPyr_upsampler(gPyr[i],gPyr[i-1])
        L = np.subtract(gPyr[i-1],GE)
        lPyr.append(L)
     
    return gPyr,lPyr
