import tensorflow as tf
from ai4med.components.models.model import Model
import tensorflow.contrib.slim as slim

from keras.layers import *
from group_norm import GroupNormalization
from keras.backend.tensorflow_backend import set_session
from switchnorm import SwitchNormalization

class CustomNetwork(Model):

    def __init__(self, num_classes,training=True,data_format='channels_last'):
        Model.__init__(self)
        self.model = None
        self.num_classes = num_classes
        self.training = training
        self.data_format = data_format        

        if data_format == 'channels_first':
            self.channel_axis = 1
        elif data_format == 'channels_last':
            self.channel_axis = -1

    
    def network(self, inputs, training, num_classes, data_format, channel_axis):
        # very shallow Unet Network       
        with tf.variable_scope('CustomNetwork'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            set_session(sess)
            
            conv01 = Conv3D(16,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv01')(inputs)
            conv01 = GroupNormalization(groups=8, axis=-1, name='GroupNorm01')(conv01)
            #conv01 = SwitchNormalization(name='switch01')(conv01)
            conv02 = Conv3D(16,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv02')(inputs)
            #conv02 = SwitchNormalization(name='switch02')(conv02)
            conv02 = GroupNormalization(groups=8, axis=-1, name='GroupNorm02')(conv02)

            conv11 = Conv3D(16,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv11')(conv01)
            #conv11 = SwitchNormalization(name='switch11')(conv11)
            conv11 = GroupNormalization(groups=8, axis=-1, name='GroupNorm11')(conv11)
            conv12 = Conv3D(16,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv12')(conv02)
            #conv12 = SwitchNormalization(name='switch12')(conv12)
            conv12 = GroupNormalization(groups=8, axis=-1, name='GroupNorm12')(conv12)

            conv11 = Conv3D(16,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv11r')(conv11)
            #conv11 = SwitchNormalization(name='switch11r')(conv11)
            conv11 = GroupNormalization(groups=8, axis=-1, name='GroupNorm11r')(conv11)
            conv11 = Add()([conv11, conv01])
            conv12 = Conv3D(16,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv12r')(conv12)
            #conv12 = SwitchNormalization(name='switch12r')(conv12)
            conv12 = GroupNormalization(groups=8, axis=-1, name='GroupNorm12r')(conv12)
            conv12 = Add()([conv12, conv02])

            conv21 = MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv11)
            conv21 = Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv21')(conv21)
            #conv21 = SwitchNormalization(name='switch21')(conv21)
            conv21 = GroupNormalization(groups=8, axis=-1, name='GroupNorm21')(conv21)
            conv22 = MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv12)
            conv22 = Conv3D(32,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv22')(conv22)
            #conv22 = SwitchNormalization(name='switch22')(conv22)
            conv22 = GroupNormalization(groups=8, axis=-1, name='GroupNorm22')(conv22)

            conv31 = Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv31')(conv21)
            #conv31 = SwitchNormalization(name='switch31')(conv31)
            conv31 = GroupNormalization(groups=8, axis=-1, name='GroupNorm31')(conv31)
            conv32 = Conv3D(32,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv32')(conv22)
            #conv32 = SwitchNormalization(name='switch32')(conv32)
            conv32 = GroupNormalization(groups=8, axis=-1, name='GroupNorm32')(conv32)

            conv31 = Conv3D(32,kernel_size=(3,3,1),strides=(1,1,1),padding='same',activation='relu',name='conv31r')(conv31)
            #conv31 = SwitchNormalization(name='switch31r')(conv31)
            conv31 = GroupNormalization(groups=8, axis=-1, name='GroupNorm31r')(conv31)
            conv31 = Add()([conv31, conv21])
            conv32 = Conv3D(32,kernel_size=(1,1,3),strides=(1,1,1),padding='same',activation='relu',name='conv32r')(conv32)
            #conv32 = SwitchNormalization(name='switch32r')(conv32)
            conv32 = GroupNormalization(groups=8, axis=-1, name='GroupNorm32r')(conv32)
            conv32 = Add()([conv32, conv22])

            convc = concatenate([MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv31),MaxPooling3D(pool_size=(2,2,1),strides=(2,2,1),padding='same')(conv32)],axis=-1)
            convc = Conv3D(64,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='convc')(convc)
            #convc = SwitchNormalization(name='switchc')(convc)
            convc = GroupNormalization(groups=8, axis=-1, name='GroupNormc')(convc)

            convc1 = Conv3D(32,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='convc1')(convc)
            #convc1 = SwitchNormalization(name='switchc1')(convc1)
            convc1 = GroupNormalization(groups=8, axis=-1, name='GroupNormc1')(convc1)

            conv4 = Conv3DTranspose(16,kernel_size=(3,3,3),strides=(2,2,1),padding='same',activation='relu',name='conv4')(convc1)
            #conv4 = SwitchNormalization(name='switch4')(conv4)
            conv4 = GroupNormalization(groups=8, axis=-1, name='GroupNorm4')(conv4)
            conv4 = concatenate([Conv3D(32, kernel_size=(1,1,1),strides=(1,1,1),padding='same',name='conv4c')(concatenate([conv31,conv32],axis=-1)),conv4],axis=-1)

            conv5 = Conv3D(24,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='conv5')(conv4)
            #conv5 = SwitchNormalization(name='switch5')(conv5)
            conv5 = GroupNormalization(groups=8, axis=-1, name='GroupNorm5')(conv5)

            conv6 = Conv3DTranspose(16,kernel_size=(3,3,3),strides=(2,2,1),padding='same',activation='relu',name='conv6')(conv5)            
            #conv6 = SwitchNormalization(name='switch6')(conv6)
            conv6 = GroupNormalization(groups=8, axis=-1, name='GroupNorm6')(conv6)
            conv6 = concatenate([Conv3D(16, kernel_size=(1,1,1),strides=(1,1,1),padding='same',name='conv6c')(concatenate([conv11,conv12],axis=-1)),conv6],axis=-1)

            conv7 = Conv3D(16,kernel_size=(3,3,3),strides=(1,1,1),padding='same',activation='relu',name='conv7')(conv6)
            #conv7 = SwitchNormalization(name='switch7')(conv7)
            conv7 = GroupNormalization(groups=8, axis=-1, name='GroupNorm7')(conv7)

            ### Output Block
            output = Conv3D(
                filters=num_classes,  # No. of tumor classes
                kernel_size=(1, 1, 1),
                strides=1,                
                name='output',
                activation='sigmoid')(conv7)

            
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        return output

    # additional custom loss
    def loss(self):
        return 0

    def get_predictions(self, inputs, training, build_ctx=None):
        self.model = self.network(inputs=inputs,training=training,num_classes=self.num_classes
                                  ,data_format=self.data_format,channel_axis=self.channel_axis)
        return self.model

    def get_loss(self):
        return self.loss()
