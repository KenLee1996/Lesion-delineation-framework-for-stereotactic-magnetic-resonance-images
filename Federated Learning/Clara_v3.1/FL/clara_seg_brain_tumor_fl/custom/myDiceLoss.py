import tensorflow as tf
from keras import backend as K
from ai4med.components.losses.loss import Loss

class MyClonedDiceLoss(Loss):

    def __init__(self,data_format='channels_first'):
        Loss.__init__(self)
        self.data_format = data_format        

    def get_loss(self, predictions, targets, build_ctx=None):
        return dice_loss(targets,predictions)

def dice_loss(targets, predictions):
    e = 1e-8
    intersection = K.sum(K.abs(targets * predictions), axis=[-3,-2,-1])
    dn = K.sum(K.square(targets) + K.square(predictions), axis=[-3,-2,-1]) + e
    return  - K.mean(2 * intersection / dn, axis=[0,1])
