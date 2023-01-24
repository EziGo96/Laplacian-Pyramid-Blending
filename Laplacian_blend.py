'''
Created on 11-Dec-2022

@author: EZIGO
'''
import numpy as np
import cv2
from sroy22_proj03.ComputePyr import lPyr_upsampler
def laplacian_blend(l_Pyr_F,l_Pyr_B,gPyr_M,num_layers):
    l_Pyr_blend = []
    for l_pyr_f,l_pyr_b,m in zip(l_Pyr_F,l_Pyr_B,gPyr_M):
        # print(l_pyr_f.shape,m.shape,l_pyr_b.shape)
        l_pyr_blend = l_pyr_f* m + l_pyr_b * (1 - m)
        l_Pyr_blend.append(np.float32(l_pyr_blend ))
    
    lap_bl = l_Pyr_blend[0]
    for i in range(1,num_layers):
        lap_bl = lPyr_upsampler(lap_bl,l_Pyr_blend[i])
        lap_bl = cv2.add(lap_bl,l_Pyr_blend[i])
    return np.clip(lap_bl,0,255).astype('uint8')