import numpy as np
#import nibabel as nib
from scipy import ndimage
import random
import tensorflow as tf
#from skimage.transform import resize

def patch(x, y, block_size, slice_num):
    l = 0
    #r = x.shape[1]
    if x.shape[2]>slice_num:
        if x.shape[0]<=block_size and x.shape[1]<=block_size:
            start = np.random.randint(x.shape[2]-slice_num)
            x_p = x[:,:,start:start+slice_num,:]
            y_p = y[:,:,start:start+slice_num]
        elif x.shape[0]<=block_size and x.shape[1]>block_size:
            start = np.random.randint([l, 0],[x.shape[1]-block_size, x.shape[2]-slice_num],2)           
            x_p = x[:,start[0]:start[0]+block_size,start[1]:start[1]+slice_num,:]
            y_p = y[:,start[0]:start[0]+block_size,start[1]:start[1]+slice_num]
        elif x.shape[0]>block_size and x.shape[1]<=block_size:
            start = np.random.randint([l, 0],[x.shape[0]-block_size, x.shape[2]-slice_num],2)       
            x_p = x[start[0]:start[0]+block_size,:,start[1]:start[1]+slice_num,:]
            y_p = y[start[0]:start[0]+block_size,:,start[1]:start[1]+slice_num]
        else:            
            start = np.random.randint([l, l, 0],[x.shape[0]-block_size, x.shape[1]-block_size, x.shape[2]-slice_num],3)            
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,start[2]:start[2]+slice_num,:]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,start[2]:start[2]+slice_num]
    else:
        if x.shape[0]<=block_size and x.shape[1]<=block_size:
            x_p = x
            y_p = y
        elif x.shape[0]<=block_size and x.shape[1]>block_size:
            start = np.random.randint(x.shape[1]-block_size)
            x_p = x[:,start:start+block_size,:,:]
            y_p = y[:,start:start+block_size,:]
        elif x.shape[0]>block_size and x.shape[1]<=block_size:
            start = np.random.randint(x.shape[0]-block_size)
            x_p = x[start:start+block_size,:,:,:]
            y_p = y[start:start+block_size,:,:]
        else:            
            start = np.random.randint([l, l],[x.shape[0]-block_size, x.shape[1]-block_size],2)
            x_p = x[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:,:]
            y_p = y[start[0]:start[0]+block_size,start[1]:start[1]+block_size,:]
        x_p, y_p = pad_z(x_p, y_p, slice_num)
    return x_p, y_p

def pad_z(x, y, slice_num):
    if x.shape[2]<slice_num:
        x_new = np.zeros([x.shape[0], x.shape[1], slice_num, x.shape[3]])
        y_new = np.zeros([y.shape[0], y.shape[1], slice_num])
        x_new[:,:,0:x.shape[2],:] = x
        y_new[:,:,0:y.shape[2]] = y
    else:
        x_new = x
        y_new = y
    return x_new, y_new

def process(data_dir, biparametric=False):
    x = np.load(data_dir.decode())['data']
    y = np.load(data_dir.decode().replace('images','labels'))['data']
    x[np.where(np.isnan(x))] = 0
    y[np.where(np.isnan(y))] = 0
    inplane_size = 256
    slice_num = 64
    #print(data_dir)
    
    if len(x.shape)>3:
        t1c_mean = np.mean(x[:,:,:,0])
        t1c_std = np.std(x[:,:,:,0])    
        t1c = x[:,:,:,0]
    else:
        t1c_mean = np.mean(x)
        t1c_std = np.std(x)
        t1c = x
    if biparametric:
        t2 = x[:,:,:,1]
        t2_mean = np.mean(x[:,:,:,1])
        t2_std = np.std(x[:,:,:,1])
    
    t1c = np.expand_dims(t1c, -1)
    if biparametric:
        t2 = np.expand_dims(t2, -1)    
        x = np.concatenate((t1c, t2),-1)
        x_p, y_p = patch(x, y, inplane_size, slice_num)
    else:
        x_p, y_p = patch(t1c, y, inplane_size, slice_num)
    
    flip_ratio = np.random.rand(1)
    if flip_ratio >= 0.5:
        x_p = np.fliplr(x_p)
        y_p = np.fliplr(y_p)
    rot_angle = np.random.randint(-15, 15)
    x_p = ndimage.rotate(x_p, rot_angle, reshape=False)
    y_p = ndimage.rotate(y_p, rot_angle, reshape=False)
    
    t1c_p = x_p[:,:,:,0]-t1c_mean
    t1c_p = t1c_p/t1c_std
    if biparametric:
        t2_p = x_p[:,:,:,1]-t2_mean
        t2_p = t2_p/t2_std   
    
    t1c_p = np.expand_dims(t1c_p, -1)
    if biparametric:
        t2_p = np.expand_dims(t2_p, -1)
        x_p = np.concatenate((t1c_p, t2_p),-1)
    else:
        x_p = t1c_p
        
    y_p[y_p>=0.5] = 1
    y_p[y_p<0.5] = 0
    
    #x_p = np.expand_dims(x_p, 0)
    y_p = np.expand_dims(y_p, -1)
    #y_p = np.expand_dims(y_p, 0)
    return np.array(x_p,dtype=np.float32), np.array(y_p,dtype=np.float32)

def tprocess(data_dir, biparametric=False):
    #print(data_dir)
    x = np.load(data_dir.decode())['data']
    y = np.load(data_dir.decode().replace('images','labels'))['data']
    x[np.where(np.isnan(x))] = 0
    y[np.where(np.isnan(y))] = 0
    inplane_size = 256
    slice_num = 64
    
    if len(x.shape)>3:
        t1c_mean = np.mean(x[:,:,:,0])
        t1c_std = np.std(x[:,:,:,0])    
        t1c = x[:,:,:,0]
    else:
        t1c_mean = np.mean(x)
        t1c_std = np.std(x)
        t1c = x
    if biparametric:
        t2 = x[:,:,:,1]
        t2_mean = np.mean(x[:,:,:,1])
        t2_std = np.std(x[:,:,:,1])
    
    t1c = np.expand_dims(t1c, -1)
    if biparametric:
        t2 = np.expand_dims(t2, -1)    
        x = np.concatenate((t1c, t2),-1)
        x_p, y_p = patch(x, y, inplane_size, slice_num)
    else:
        x_p, y_p = patch(t1c, y, inplane_size, slice_num)
    
    t1c_p = x_p[:,:,:,0]-t1c_mean
    t1c_p = t1c_p/t1c_std
    if biparametric:
        t2_p = x_p[:,:,:,1]-t2_mean
        t2_p = t2_p/t2_std
    
    t1c_p = np.expand_dims(t1c_p, -1)
    if biparametric:
        t2_p = np.expand_dims(t2_p, -1)
        x_p = np.concatenate((t1c_p, t2_p),-1)
    else:
        x_p = t1c_p
        
    y_p[y_p>=0.5] = 1
    y_p[y_p<0.5] = 0
    
    #x_p = np.expand_dims(x_p, 0)
    y_p = np.expand_dims(y_p, -1)
    #y_p = np.expand_dims(y_p, 0)
    return np.array(x_p,dtype=np.float32), np.array(y_p,dtype=np.float32)

def tf_TrDataset(path, batch_size, biparametric):
    train_dataset = tf.data.Dataset.from_tensor_slices(path)
    train_dataset = train_dataset.map(lambda filename: 
                                     tf.numpy_function(func=process,
                                                      inp=[filename, biparametric],
                                                      Tout=[tf.float32, 
                                                            tf.float32]
                                                      ),
                                     num_parallel_calls=16)
    train_dataset = train_dataset.shuffle(20)
    train_dataset = train_dataset.prefetch(10)
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset

def tf_ValDataset(vpath, vbatch_size, biparametric):
    val_dataset = tf.data.Dataset.from_tensor_slices(vpath)
    val_dataset = val_dataset.map(lambda filename: 
                                  tf.numpy_function(func=tprocess,
                                                    inp=[filename, biparametric],
                                                    Tout=[tf.float32, 
                                                          tf.float32]
                                                   ),
                                  num_parallel_calls=16)
    val_dataset = val_dataset.prefetch(10)
    val_dataset = val_dataset.batch(vbatch_size)#.cache()
    return val_dataset