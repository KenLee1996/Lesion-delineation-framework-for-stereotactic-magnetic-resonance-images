import tensorflow as tf
from keras import backend as K
from ai4med.components.losses.loss import Loss

class MyClonedDiceLoss(Loss):

    def __init__(self,data_format='channels_last'):
        Loss.__init__(self)
        self.data_format = data_format

    def get_loss(self, predictions, targets, build_ctx=None):
        return dice_loss(targets, predictions)

def dice_loss(targets, predictions):
    e = 1e-8
    intersection = K.sum(targets * predictions, axis=[1,2,3]) + e
    dn = K.sum(targets, axis=[1,2,3]) + K.sum(predictions, axis=[1,2,3]) + e
    return 1 - K.mean(2 * intersection / dn, axis=[0,-1])
