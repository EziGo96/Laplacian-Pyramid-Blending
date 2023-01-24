'''
Created on 11-Dec-2022

@author: EZIGO
'''
import cv2
import numpy as np 
from sroy22_proj03.ComputePyr import ComputePyr,gPyr_downsampler
from sroy22_proj03.Laplacian_blend import laplacian_blend
import time 

def draw(event,x,y,flags,param):
    global x1,y1,x2,y2,drawing,m_shape
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1,y1 = x,y
        
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode == False:
                cv2.ellipse(fore_imgcp,((x1+(x-x1)//2),(y1+(y-y1)//2)),(int((x-x1)//np.sqrt(2)),int((y-y1)//np.sqrt(2))),0,0,360,(0,255,0),2)
            else:
                cv2.rectangle(fore_imgcp,(x1,y1),(x,y),(0,255,0),2)           
            x2 = x;
            y2 = y;
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == False:
            cv2.ellipse(fore_imgcp,((x1+(x-x1)//2),(y1+(y-y1)//2)),(int((x-x1)//np.sqrt(2)),int((y-y1)//np.sqrt(2))),0,0,360,(0,255,0),2)
        else:
            cv2.rectangle(fore_imgcp,(x1,y1),(x,y),(0,255,0),2)           
        x2 = x;
        y2 = y;

def alignment(b_img,f_img,r,c):
    global s_flag
    s_flag = False
    a_img = np.zeros(b_img.shape)
    a_img[r:f_img.shape[0]+r,c:f_img.shape[1]+c] = f_img
    return a_img.astype('uint8')

def mask_gpyr(mask,num_layers):
    G_M = np.copy(mask).astype('float32')
    gpM = [G_M]
    for i in range(num_layers-1):
        G_M = gPyr_downsampler(G_M)
        gpM.append(G_M)
    gpM.reverse()
    return gpM


def mask_gen(img,x1,y1,x2,y2):
    mask = np.zeros((img.shape)).astype('float32')
    if mode == False:
        mask=cv2.ellipse(mask,((x1+(x2-x1)//2),(y1+(y2-y1)//2)),(int((x2-x1)//np.sqrt(2)),int((y2-y1)//np.sqrt(2))),0,0,360,(1,1,1),-1)
    else:
        mask=cv2.rectangle(mask,(x1,y1),(x2,y2),(1,1,1),-1)
    return mask

drawing = False # true if mouse l_burron clicked
mode = True #  True, draw rectangle. Press 'f' to toggle to ellipse
x1,y1 = -1,-1
s_flag = True # if size of fore_img & back_img is not same, s_flag sets to False

#read image
fore_img = cv2.imread('lena.png')
back_img = cv2.imread('P3_BG_2.png')
dsize = back_img.shape
print(dsize)
fore_img = cv2.resize(fore_img,dsize=(dsize[0],dsize[1]),interpolation=cv2.INTER_CUBIC)

# validation for aligning
if (back_img.shape[0] > fore_img.shape[0]):
    s_flag = False
    n_fore_img = alignment(back_img,fore_img,10,100)
    fore_img = np.copy(n_fore_img)
    fore_imgcp = np.copy(n_fore_img)
else:
    fore_imgcp = np.copy(fore_img)

# display for GUI
cv2.namedWindow('Foreground Image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Foreground Image',draw)

while(1):
    cv2.imshow('Foreground Image',fore_imgcp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('f'): # ellipse
        mode = not mode
    elif k == 27:
        cv2.destroyAllWindows()
        break

# mask creation
mask = mask_gen(fore_imgcp,x1,y1,x2,y2)

# display images
cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image',fore_img)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask',mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image',back_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

start_time = time.time()
# compute pyramids- here you can enter layers as per requirement
gPyr_F,l_Pyr_F = ComputePyr(fore_img,10)
gPyr_B,l_Pyr_B = ComputePyr(back_img,10)
gPyr_M = mask_gpyr(mask,len(gPyr_F)) 
blended_img = laplacian_blend(l_Pyr_F,l_Pyr_B,gPyr_M,len(gPyr_F)) 

endtime=time.time()
runtime_s=(endtime - start_time)

print("Runtime: "+str(np.round(runtime_s,2))+"s")
# display images
cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image',fore_imgcp)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask',mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image',back_img)
cv2.namedWindow('Blended Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blended Image',blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()