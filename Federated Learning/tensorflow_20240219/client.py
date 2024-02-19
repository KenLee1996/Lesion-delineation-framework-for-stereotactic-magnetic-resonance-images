import flwr as fl
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import mixed_precision
from keras.callbacks import ModelCheckpoint

from losses import dice_coefficient, generalized_dice_loss
from M4modelGnRes import build_model
#from data_generator_20221012 import TrDataGenerator, ValDataGenerator
from tf_Dataset_20240219 import tf_TrDataset, tf_ValDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
mixed_precision.set_global_policy('mixed_float16')
tf.config.run_functions_eagerly(True)

train_image_path = 'path that save the training data'
val_image_path = 'path that save the validation data'

k = os.listdir(train_image_path)
path = []
for j in k:
    path.append(train_image_path + '/' + j)
vpath = []
k = os.listdir(val_image_path)
for j in k:
    vpath.append(val_image_path + '/' + j)

lr = 1e-4
local_epoch_num = 2
batch_size = 1
vbatch_size = 1
biparametric = True
base_path = 'directory to save model'
model_folder = 'model_folder_name'

if biparametric:
    input_channel_num = 2
else:
    input_channel_num = 1

model = build_model(input_shape=(None, None, None, input_channel_num), output_channels=1)
model.compile(optimizer = Adam(learning_rate=lr), loss = generalized_dice_loss, metrics = [dice_coefficient])
modelcheckpoint = ModelCheckpoint(os.path.join(base_path, model_folder, 'local.h5'),
                                  monitor='loss',save_best_only=True,mode='min')

steps_per_epoch = int(len(path)/batch_size)
steps_of_val = int(len(vpath)/vbatch_size)

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)        
        model.fit(
            tf_TrDataset(path, 
                         batch_size, 
                         biparametric),
            epochs=local_epoch_num,                     
            callbacks=[modelcheckpoint])
        return model.get_weights(), len(path), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, dice_coefficient = model.evaluate(tf_ValDataset(vpath, 
                                                              vbatch_size, 
                                                              biparametric))
        #model.save(os.path.join(base_path, model_folder, 'tvgh_global.h5'))
        return loss, len(vpath), {"dice_coefficient": dice_coefficient}

fl.client.start_numpy_client(server_address="120.126.105.200:8080", 
                             client=Client())
