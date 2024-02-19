import os
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage import zoom
import nibabel as nib

def clipping_intensity(x, clipping):
    #Remove outliers, cliping% max/min  
    upper = np.percentile(x, 100 - clipping)
    lower = np.percentile(x, clipping)
    #print(upper)
    #print(lower)
    x[x > upper] = upper
    x[x < lower] = lower
    return x

def normalize(x):
    for i in range(x.shape[-1]):
        x[:,:,:,i] = (x[:,:,:,i]-np.mean(x[:,:,:,i]))/np.std(x[:,:,:,i])
    return x
def padXY(x, y, padSize):
    if x.shape[0]<=padSize and x.shape[1]<=padSize:
        x_new = np.zeros((padSize, padSize, x.shape[2]))
        #y_new_s = np.max(y.shape[0:2])        
        y_new = np.zeros((padSize, padSize, y.shape[2]))        
        x_new[int((padSize-x.shape[0])/2):int((padSize-x.shape[0])/2)+x.shape[0],
             int((padSize-x.shape[1])/2):int((padSize-x.shape[1])/2)+x.shape[1],
             :] = x
        y_new[int((padSize-y.shape[0])/2):int((padSize-y.shape[0])/2)+y.shape[0],
             int((padSize-y.shape[1])/2):int((padSize-y.shape[1])/2)+y.shape[1],
             :] = y
    elif x.shape[0]>padSize and x.shape[1]<=padSize:
        x_new = np.zeros((x.shape[0], padSize, x.shape[2]))
        #y_new_s = np.max(y.shape[0:2])        
        y_new = np.zeros((y.shape[0], padSize, y.shape[2]))
        x_new[:,
             int((padSize-x.shape[1])/2):int((padSize-x.shape[1])/2)+x.shape[1],
             :] = x
        y_new[:,
             int((padSize-y.shape[1])/2):int((padSize-y.shape[1])/2)+y.shape[1],
             :] = y
    elif x.shape[0]<=padSize and x.shape[1]>padSize:
        x_new = np.zeros((padSize, x.shape[1], x.shape[2]))
        #y_new_s = np.max(y.shape[0:2])        
        y_new = np.zeros((padSize, y.shape[1], y.shape[2]))            
        x_new[int((padSize-x.shape[0])/2):int((padSize-x.shape[0])/2)+x.shape[0],
             :,
             :] = x
        y_new[int((padSize-y.shape[0])/2):int((padSize-y.shape[0])/2)+y.shape[0],
             :,
             :] = y      
    return x_new, y_new
def tprocess(data_dir, biparametric=False):
    data = np.load(data_dir)
    x = data['x_standard']
    y = data['y_standard']
    
    if x.shape[0]<256 or x.shape[1]<256:
        x, y = padXY(x, y, padSize=256)
    #if x.shape[0] != x.shape[1]:
        #max_size = np.max(x.shape[:2])
        #x, y = padXY(x, y, padSize=max_size)
    if len(x.shape)<4:
        x = np.expand_dims(x, -1)
    inference_size = 256
    
    #if x.shape[0] != inference_size:
        #x = zoom(x, (inference_size/x.shape[0], inference_size/x.shape[1], 1, 1), order = 1)
    
    t1c_mean = np.mean(x[:,:,:,0])
    t1c_std = np.std(x[:,:,:,0])
    if biparametric:
        t2_mean = np.mean(x[:,:,:,1])
        t2_std = np.std(x[:,:,:,1])
    
    t1c = x[:,:,:,0]-t1c_mean
    t1c = t1c/t1c_std
    if biparametric:
        t2 = x[:,:,:,1]-t2_mean
        t2 = t2/t2_std
    t1c = np.expand_dims(t1c, -1)
    if biparametric:
        t2 = np.expand_dims(t2, -1)
        x = np.concatenate((t1c, t2),-1)
    else:
        x = t1c
    x = np.expand_dims(x, 0)
    return np.array(x,dtype=np.float32), y

def tprocess_nii(data_dir, biparametric=False):
    #print(data_dir)
    pid = data_dir.split('/')[-1]    
    t1c = nib.load(os.path.join(data_dir, pid + '_t1ce.nii.gz')).get_fdata()
    y = nib.load(os.path.join(data_dir, pid + '_seg.nii.gz')).get_fdata()
    t1c[np.where(np.isnan(t1c))] = 0
    t1c = np.rot90(t1c, 3)
    t1c = clipping_intensity(t1c, 1)
    y[np.where(np.isnan(y))] = 0
    y[np.where(y==2)] = 0
    y = np.rot90(y, 3)
    
    if t1c.shape[0]<256 or t1c.shape[1]<256:
        t1c, y = padXY(t1c, y, padSize=256)
    #if t1c.shape[0] != t1c.shape[1]:
        #max_size = np.max(t1c.shape[:2])
        #t1c, y = padXY(t1c, y, padSize=max_size)
    
    inplane_size = 256
    slice_num = 64
    
    t1c_mean = np.mean(t1c)
    t1c_std = np.std(t1c)
    t1c = t1c-t1c_mean
    t1c = t1c/t1c_std
    if biparametric:
        t2 = nib.load(os.path.join(data_dir, pid + '_t2.nii.gz')).get_fdata()
        t2[np.where(np.isnan(t2))] = 0
        t2 = np.rot90(t2, 3)
        t2 = clipping_intensity(t2, 1)
        
        if t2.shape[0]<256 or t2.shape[1]<256:
            t2, y = padXY(t2, y, padSize=256)
        if t2.shape[0] != t2.shape[1]:
            max_size = np.max(t2.shape[:2])
            t2, y = padXY(t2, y, padSize=max_size)
        
        t2_mean = np.mean(t2)
        t2_std = np.std(t2)
        t2 = t2-t2_mean
        t2 = t2/t2_std
    
    t1c = np.expand_dims(t1c, -1)
    if biparametric:
        t2 = np.expand_dims(t2, -1)
        x = np.concatenate((t1c, t2),-1)
    else:
        x = t1c
        
    y[y>=0.5] = 1
    y[y<0.5] = 0
    
    x = np.expand_dims(x, 0)
    return np.array(x,dtype=np.float32), np.array(y,dtype=np.float32)

def sliding_inference(img, model, inplane_size, slice_num):
    stride = 64
    step = int((img.shape[1]-inplane_size)/stride)
    cer = np.zeros((img.shape[1], img.shape[2], img.shape[3]))
    pre = np.zeros_like(cer)
    
    if img.shape[3] > slice_num:
        slice_stride = int(slice_num/4)
        slice_step = int((img.shape[3]-slice_num)/slice_stride)
        for si in range(slice_step+1):
            if int(si*slice_stride)+slice_num>img.shape[3]:
                slice_start = int(img.shape[3]-slice_num)
                slice_end = int(img.shape[3])
            else:
                slice_start = int(si*slice_stride)            
                slice_end = int(si*slice_stride)+slice_num
            
            for ix in range(step+1):
                if int(ix*stride)+inplane_size<=img.shape[1]:
                    startx = int(ix*stride)
                    endx = int(ix*stride)+inplane_size
                else:
                    startx = int(img.shape[1])-inplane_size
                    endx = int(img.shape[1])
                for iy in range(step+1):
                    if int(iy*stride)+inplane_size<=img.shape[2]:
                        starty = int(iy*stride)
                        endy = int(iy*stride)+inplane_size
                    else:
                        starty = int(img.shape[2])-inplane_size
                        endy = int(img.shape[2])
                    tmp_img = img[:,startx:endx,starty:endy,slice_start:slice_end,:]
                    tmp_img = torch.from_numpy(np.double(tmp_img)).type(torch.FloatTensor).cuda()
                    tmp = model.predict(tmp_img, verbose=0)[0,:,:,:,0]
                    tmp = tmp.detach().cpu().numpy()
                    cer[startx:endx,starty:endy,slice_start:slice_end] = cer[startx:endx,starty:endy,slice_start:slice_end] + 1
                    pre[startx:endx,starty:endy,slice_start:slice_end] = pre[startx:endx,starty:endy,slice_start:slice_end] + tmp
                    del tmp_img, tmp
    else:
        for ix in range(step+1):                
            if int(ix*stride)+inplane_size<=img.shape[1]:
                startx = int(ix*stride)
                endx = int(ix*stride)+inplane_size
            else:
                startx = int(img.shape[1])-inplane_size
                endx = int(img.shape[1])
            for iy in range(step+1):
                if int(iy*stride)+inplane_size<=img.shape[2]:
                    starty = int(iy*stride)
                    endy = int(iy*stride)+inplane_size
                else:
                    starty = int(img.shape[2])-inplane_size
                    endy = int(img.shape[2])
                tmp_img = img[:,startx:endx,starty:endy,:,:]
                tmp_img = torch.from_numpy(np.double(tmp_img)).type(torch.FloatTensor).cuda()
                tmp = model.predict(tmp_img, verbose=0)[0,:,:,:,0]
                tmp = tmp.detach().cpu().numpy()
                cer[startx:endx,starty:endy,:] = cer[startx:endx,starty:endy,:] + 1
                pre[startx:endx,starty:endy,:] = pre[startx:endx,starty:endy,:] + tmp
                del tmp_img, tmp
    val = pre.copy()
    val = val/cer
    return val