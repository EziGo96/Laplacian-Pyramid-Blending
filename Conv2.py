'''
Created on 11-Dec-2022

@author: EZIGO
'''
import numpy as np

def zero_pad(f,padding_width):
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2),dtype=float)
    padded_f[padding_width:-padding_width, padding_width:-padding_width] = f
    return padded_f

def zero_pad_3(f,padding_width):
    r = f[:,:,0]
    g = f[:,:,1]
    b = f[:,:,2]
    
    padded_r = zero_pad(r,padding_width)
    padded_g = zero_pad(g,padding_width)
    padded_b = zero_pad(b,padding_width)
    
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2,f.shape[2]),dtype=int)
    padded_f[:,:,0] = padded_r
    padded_f[:,:,1] = padded_g
    padded_f[:,:,2] = padded_b
    return padded_f

def wrap_around(f,padding_width):
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2),dtype=int)
    padded_f[padding_width:-padding_width, padding_width:-padding_width] = f
    '''corner cases'''
    padded_f[0:padding_width,0:padding_width] = f[-(padding_width+1):-1,-(padding_width+1):-1]
    padded_f[0:padding_width,-(padding_width+1):-1] = f[-(padding_width+1):-1,0:padding_width]
    padded_f[-(padding_width+1):-1,0:padding_width] = f[0:padding_width,-(padding_width+1):-1]
    padded_f[-(padding_width+1):-1,-(padding_width+1):-1] = f[0:padding_width,0:padding_width]
    '''edge cases'''
    padded_f[padding_width:-padding_width,0:padding_width] = f[:,-(padding_width+1):-1]
    padded_f[0:padding_width,padding_width:-padding_width] = f[-(padding_width+1):-1,:]
    padded_f[padding_width:-padding_width,-(padding_width+1):-1] = f[:,0:padding_width]
    padded_f[-(padding_width+1):-1,padding_width:-padding_width] = f[0:padding_width,:]
    return padded_f

def wrap_around_3(f,padding_width):
    r = f[:,:,0]
    g = f[:,:,1]
    b = f[:,:,2]
    
    padded_r = wrap_around(r,padding_width)
    padded_g = wrap_around(g,padding_width)
    padded_b = wrap_around(b,padding_width)
    
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2,f.shape[2]),dtype=int)
    padded_f[:,:,0] = padded_r
    padded_f[:,:,1] = padded_g
    padded_f[:,:,2] = padded_b
    return padded_f    

def copy_pad(f,padding_width):
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2),dtype=int)
    padded_f[padding_width:-padding_width, padding_width:-padding_width] = f
    '''corner cases'''
    padded_f[0:padding_width,0:padding_width].fill(f[0,0])
    padded_f[0:padding_width,-(padding_width+1):-1].fill(f[0,-1])
    padded_f[-(padding_width+1):-1,0:padding_width].fill(f[-1,0])
    padded_f[-(padding_width+1):-1,-(padding_width+1):-1].fill(f[0,0])
    '''edge cases'''
    padded_f[padding_width:-padding_width,0:padding_width] = f[:,0][:, np.newaxis]
    padded_f[0:padding_width,padding_width:-padding_width] = f[0,:][:, np.newaxis].T
    padded_f[padding_width:-padding_width,-(padding_width+1):-1] = f[:,-1][:, np.newaxis]
    padded_f[-(padding_width+1):-1,padding_width:-padding_width] = f[-1,:][:, np.newaxis].T
    return padded_f 

def copy_pad_3(f,padding_width):
    r = f[:,:,0]
    g = f[:,:,1]
    b = f[:,:,2]
    
    padded_r = copy_pad(r,padding_width)
    padded_g = copy_pad(g,padding_width)
    padded_b = copy_pad(b,padding_width)
    
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2,f.shape[2]),dtype=int)
    padded_f[:,:,0] = padded_r
    padded_f[:,:,1] = padded_g
    padded_f[:,:,2] = padded_b
    return padded_f  

def reflect_pad(f,padding_width):
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2),dtype=int)
    padded_f[padding_width:-padding_width, padding_width:-padding_width] = f
    '''corner cases'''
    padded_f[0:padding_width,0:padding_width] = np.fliplr(np.flipud(f[0:padding_width,0:padding_width]))
    padded_f[0:padding_width,-(padding_width+1):-1] = np.fliplr(np.flipud(f[0:padding_width,-(padding_width+1):-1]))
    padded_f[-(padding_width+1):-1,0:padding_width] = np.fliplr(np.flipud(f[-(padding_width+1):-1,0:padding_width]))
    padded_f[-(padding_width+1):-1,-(padding_width+1):-1] = np.fliplr(np.flipud(f[-(padding_width+1):-1,-(padding_width+1):-1]))
    '''edge cases'''
    padded_f[padding_width:-padding_width,0:padding_width] = np.fliplr(f[:,0:padding_width])
    padded_f[0:padding_width,padding_width:-padding_width] = np.flipud(f[0:padding_width,:])
    padded_f[padding_width:-padding_width,-(padding_width+1):-1] = np.fliplr(f[:,-(padding_width+1):-1])
    padded_f[-(padding_width+1):-1,padding_width:-padding_width] = np.flipud(f[-(padding_width+1):-1,:])
    return padded_f

def reflect_pad_3(f,padding_width):
    r = f[:,:,0]
    g = f[:,:,1]
    b = f[:,:,2]
    
    padded_r = reflect_pad(r,padding_width)
    padded_g = reflect_pad(g,padding_width)
    padded_b = reflect_pad(b,padding_width)
    
    padded_f = np.zeros(shape=(f.shape[0] + padding_width * 2,f.shape[1] + padding_width * 2,f.shape[2]),dtype=int)
    padded_f[:,:,0] = padded_r
    padded_f[:,:,1] = padded_g
    padded_f[:,:,2] = padded_b
    return padded_f  
    
def target_size(img_size, kernel_size):
    num_pixels = 0
    for i in range(img_size):
        a = i + kernel_size
        if a <= img_size:
            num_pixels += 1
    return num_pixels

def convolve(f, w, stride):
    target_size_i = target_size(img_size=f.shape[0],kernel_size=w.shape[0])
    target_size_j = target_size(img_size=f.shape[1],kernel_size=w.shape[1])
    k_i = w.shape[0]
    k_j = w.shape[1]
    convolved_f = np.zeros(shape=(target_size_i, target_size_j),dtype=float)
    
    for i in range(0,target_size_i,stride[0]):
        for j in range(0,target_size_j,stride[1]):
            sub = f[i:i+k_i, j:j+k_j]
            convolved_f[i, j] = np.sum(np.multiply(sub, w))
    return convolved_f

def conv2(f,w,pad):
    stride = [1,1]
    if len(f.shape) == 3:
        n_channel = f.shape[-1]
    else:
        n_channel = 1
    w_size=w.shape[0]
    padding_width=max((1,(w_size-1)//2))
    if n_channel==1:
        if pad=="zero_pad":
            padded_f = zero_pad(f,padding_width)
        elif pad=="wrap_around":
            padded_f = wrap_around(f,padding_width)
        elif pad=="copy_pad":
            padded_f = copy_pad(f,padding_width)
        elif pad=="reflect_pad":
            padded_f = reflect_pad(f,padding_width)
        else:
            print("no such padding")
        g = convolve(padded_f, w, stride)
    
    elif n_channel==3:
        if pad=="zero_pad":
            padded_f = zero_pad_3(f,padding_width)
        elif pad=="wrap_around":
            padded_f = wrap_around_3(f,padding_width)
        elif pad=="copy_pad":
            padded_f = copy_pad_3(f,padding_width)
        elif pad=="reflect_pad":
            padded_f = reflect_pad_3(f,padding_width)
        else:
            print("no such padding")
        r = padded_f[:,:,0]
        g = padded_f[:,:,1]
        b = padded_f[:,:,2]
    
        g_r = convolve(r, w, stride)
        g_g = convolve(g, w, stride)
        g_b = convolve(b, w, stride)
    
        g = np.zeros(shape=(g_r.shape[0], g_r.shape[1], 3),dtype=int)
        g[:,:,0] = g_r
        g[:,:,1] = g_g
        g[:,:,2] = g_b
    else:
        print("Wrong Image format")
        
    return g.astype('float32')
