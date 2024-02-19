import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.losses import *

def dice_coefficient(y_true, y_pred):
    intersection = K.sum(y_true * y_pred,axis=[0,1,2,3]) + 1e-8
    dn = K.sum(y_true + y_pred, axis=[0,1,2,3]) + 1e-8
    #K.mean(2 * intersection / dn, axis=0)
    #dice = K.mean((2.*intersection)/dn,axis=-1)
    dice = (2.*intersection[-1])/dn[-1]
    return dice

def dice_coef_loss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred,axis=[0,1,2,3]) + 1e-8
    dn = K.sum(y_true + y_pred, axis=[0,1,2,3]) + 1e-8
    dice = K.mean((2.*intersection)/dn,axis=-1)
    return 1-dice

def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2,3))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)