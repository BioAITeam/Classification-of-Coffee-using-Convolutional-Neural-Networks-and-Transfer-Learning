import seaborn as sn
import cv2
from time import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Lambda, Layer, ReLU
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import datasets,layers,models,Input,Model
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, SGD, Adadelta
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import sys
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, AveragePooling2D, Cropping2D
from tensorflow.keras.layers import Dropout, Activation, Flatten, Concatenate, Dense, Reshape, Add, PReLU, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, SGD, Adadelta
from tensorflow.keras.applications import vgg16, vgg19, InceptionV3, Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence, is_generator_or_sequence
from tensorflow.keras.layers import Lambda, Layer, ReLU

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)




#channel variables
          
img_width = 224
img_height = 224
input_channel = 15



def vgg16_3canal_sin_pesos(pretrained_weights=None, input_size=(img_height, img_width, input_channel)):

    def multify_weights(kernel, out_channels):
      mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
      tiled = np.tile(mean_1d, (out_channels, 1))
      return(tiled)


    def weightify(model_orig, custom_model, layer_modify):
      layer_to_modify = [layer_modify]

      conf = custom_model.get_config()
      layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

      for layer in model_orig.layers:
        if layer.name in layer_names:
          if layer.get_weights() != []:
            target_layer = custom_model.get_layer(layer.name)

            if layer.name in layer_to_modify:    
              kernels = layer.get_weights()[0]
              biases  = layer.get_weights()[1]

              kernels = np.mean(kernels, axis=-2).reshape(kernels[:,:,-1:,:].shape)
              kernels_extra_channel = np.tile(kernels, (15, 1))
                                                  
              target_layer.set_weights([kernels_extra_channel, biases])
              target_layer.trainable = False

            else:
              target_layer.set_weights(layer.get_weights())
              target_layer.trainable = False
            
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, classes=5)
    #vgg16.summary()
    # Get vgg16 config in dictionary format
    config = vgg16.get_config()
    config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, 15)
    #config["layers"][2]["config"]["strides"] = (1, 1)

    vgg16_custom = tf.keras.models.Model.from_config(config)
    modify_name = config["layers"][1]["config"]["name"]
 
    # Create new model with config
    weightify(vgg16, vgg16_custom, modify_name)
    #vgg16_custom.summary()
    tf.keras.backend.clear_session() 
    
    for layer in vgg16_custom.layers[:5]: #150
       layer.trainable = False
    for layer in vgg16_custom.layers[5:]:
       layer.trainable = True

    #Inputs
       
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16_custom(input_tensor)
    layers = Flatten(name="flatten")(out1)

    #Hidden Layers
    layers = Dense(256)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)

    layers = Dense(128)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)
    #classification layer
    predictions = Dense(5, activation="softmax", name="output_1")(layers)
    modelFT = Model(inputs = input_tensor, outputs=predictions)
    
 
    modelFT.compile("adamax",
      loss="categorical_crossentropy",
      metrics=["accuracy"])
    modelFT.summary()
    return modelFT

    
def vgg19_3canal_sin_pesos(pretrained_weights=None, input_size=(img_height, img_width, input_channel)):

    def multify_weights(kernel, out_channels):
      mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
      tiled = np.tile(mean_1d, (out_channels, 1))
      return(tiled)


    def weightify(model_orig, custom_model, layer_modify):
      layer_to_modify = [layer_modify]

      conf = custom_model.get_config()
      layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

      for layer in model_orig.layers:
        if layer.name in layer_names:
          if layer.get_weights() != []:
            target_layer = custom_model.get_layer(layer.name)

            if layer.name in layer_to_modify:    
              kernels = layer.get_weights()[0]
              biases  = layer.get_weights()[1]

              kernels = np.mean(kernels, axis=-2).reshape(kernels[:,:,-1:,:].shape)
              kernels_extra_channel = np.tile(kernels, (15, 1))
                                                  
              target_layer.set_weights([kernels_extra_channel, biases])
              target_layer.trainable = False

            else:
              target_layer.set_weights(layer.get_weights())
              target_layer.trainable = False
            
            
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, classes=5)
    #vgg16.summary()
    # Get vgg19 config in dictionary format
    config = vgg16.get_config()
    config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, 15)
    #config["layers"][2]["config"]["strides"] = (1, 1)

    vgg16_custom = tf.keras.models.Model.from_config(config)
    modify_name = config["layers"][1]["config"]["name"]
 
    # Create new model with config
    weightify(vgg16, vgg16_custom, modify_name)
    #vgg16_custom.summary()
    tf.keras.backend.clear_session() 
    
    for layer in vgg16_custom.layers[:7]: #150
       layer.trainable = False
    for layer in vgg16_custom.layers[7:]:
       layer.trainable = True

    #Inputs  
    
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16_custom(input_tensor)
    layers = Flatten(name="flatten")(out1)

    #Hidden Layers
    layers = Dense(256)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)

    layers = Dense(128)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)
    #classification layer
    predictions = Dense(5, activation="softmax", name="output_1")(layers)
    modelFT = Model(inputs = input_tensor, outputs=predictions)
    
 
    modelFT.compile("adamax",
      loss="categorical_crossentropy",
      metrics=["accuracy"])
    modelFT.summary()
    return modelFT


    
def Inceptionv2_3canal_sin_pesos(pretrained_weights=None, input_size=(img_height, img_width, input_channel)):

    def multify_weights(kernel, out_channels):
      mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
      tiled = np.tile(mean_1d, (out_channels, 1))
      return(tiled)


    def weightify(model_orig, custom_model, layer_modify):
      layer_to_modify = [layer_modify]

      conf = custom_model.get_config()
      layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

      for layer in model_orig.layers:
        if layer.name in layer_names:
          if layer.get_weights() != []:
            target_layer = custom_model.get_layer(layer.name)

            if layer.name in layer_to_modify:    
              kernels = layer.get_weights()[0]
              #biases  = layer.get_weights()[1]

              #kernels_extra_channel = np.concatenate((kernels,
              #                                    multify_weights(kernels, input_channel - 3)),
              #                                    axis=-2) # For channels_last
              
              kernels = np.mean(kernels, axis=-2).reshape(kernels[:,:,-1:,:].shape)
              kernels_extra_channel = np.tile(kernels, (15, 1))
              
              #print(kernels_extra_channel)                                       
              target_layer.set_weights([kernels_extra_channel]) #, biases
              target_layer.trainable = False

            else:
              target_layer.set_weights(layer.get_weights())
              target_layer.trainable = False
            
            
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, classes=5)
    #vgg16.summary()
    # Get Inceptionv2 config in dictionary format
    config = vgg16.get_config()
    config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, 15)
    #config["layers"][2]["config"]["strides"] = (1, 1)

    vgg16_custom = tf.keras.models.Model.from_config(config)
    modify_name = config["layers"][1]["config"]["name"]
 
    # Create new model with config
    weightify(vgg16, vgg16_custom, modify_name)
    #vgg16_custom.summary()
    tf.keras.backend.clear_session() 
    
    for layer in vgg16_custom.layers[:150]: #150
       layer.trainable = False
    for layer in vgg16_custom.layers[150:]:
       layer.trainable = True

    #Inputs
       
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16_custom(input_tensor)
    layers = Flatten(name="flatten")(out1)

    #Hidden Layers
    layers = Dense(256)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)

    layers = Dense(128)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)
    #classification layer
    predictions = Dense(5, activation="softmax", name="output_1")(layers)
    modelFT = Model(inputs = input_tensor, outputs=predictions)
    
 
    modelFT.compile("adamax",
      loss="categorical_crossentropy",
      metrics=["accuracy"])
    modelFT.summary()
    return modelFT

    
def Inceptionv3_3canal_sin_pesos(pretrained_weights=None, input_size=(img_height, img_width, input_channel)):

    def multify_weights(kernel, out_channels):
      mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
      tiled = np.tile(mean_1d, (out_channels, 1))
      return(tiled)


    def weightify(model_orig, custom_model, layer_modify):
      layer_to_modify = [layer_modify]

      conf = custom_model.get_config()
      layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

      for layer in model_orig.layers:
        if layer.name in layer_names:
          if layer.get_weights() != []:
            target_layer = custom_model.get_layer(layer.name)

            if layer.name in layer_to_modify:    
              kernels = layer.get_weights()[0]
              #biases  = layer.get_weights()[1]

              #kernels_extra_channel = np.concatenate((kernels,
              #                                    multify_weights(kernels, input_channel - 3)),
              #                                    axis=-2) # For channels_last
              
              kernels = np.mean(kernels, axis=-2).reshape(kernels[:,:,-1:,:].shape)
              kernels_extra_channel = np.tile(kernels, (15, 1))
          
              #print(kernels_extra_channel)                                       
              target_layer.set_weights([kernels_extra_channel]) #, biases
              target_layer.trainable = False

            else:
              target_layer.set_weights(layer.get_weights())
              target_layer.trainable = False


    inputs = Input(input_size)
    vgg16 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, classes=5)
    #vgg16.summary()
    # Get Inceptionv3 config in dictionary format
    config = vgg16.get_config()
    config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, 15)
    #config["layers"][2]["config"]["strides"] = (1, 1)

    vgg16_custom = tf.keras.models.Model.from_config(config)
    modify_name = config["layers"][1]["config"]["name"]
 
    # Create new model with config
    weightify(vgg16, vgg16_custom, modify_name)
    #vgg16_custom.summary()
    tf.keras.backend.clear_session() 

    for layer in vgg16_custom.layers[:150]: #150
       layer.trainable = False
    for layer in vgg16_custom.layers[150:]:
       layer.trainable = True

    #Inputs
       
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16_custom(input_tensor)
    layers = Flatten(name="flatten")(out1)

    #Hidden Layers
    layers = Dense(256)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)

    layers = Dense(128)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)
    #classification layer
    predictions = Dense(5, activation="softmax", name="output_1")(layers)
    modelFT = Model(inputs = input_tensor, outputs=predictions)
    
 
    modelFT.compile("adamax",
      loss="categorical_crossentropy",
      metrics=["accuracy"])
    modelFT.summary()
    return modelFT

    
def DenseNet201_3canal_sin_pesos(pretrained_weights=None, input_size=(img_height, img_width, input_channel)):

    def multify_weights(kernel, out_channels):
      mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
      tiled = np.tile(mean_1d, (out_channels, 1))
      return(tiled)


    def weightify(model_orig, custom_model, layer_modify):
      layer_to_modify = [layer_modify]

      conf = custom_model.get_config()
      layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

      for layer in model_orig.layers:
        if layer.name in layer_names:
          if layer.get_weights() != []:
            target_layer = custom_model.get_layer(layer.name)

            if layer.name in layer_to_modify:    
              kernels = layer.get_weights()[0]
              #biases  = layer.get_weights()[1]

              #kernels_extra_channel = np.concatenate((kernels,
              #                                    multify_weights(kernels, input_channel - 3)),
              #                                    axis=-2) # For channels_last
              
              kernels = np.mean(kernels, axis=-2).reshape(kernels[:,:,-1:,:].shape)
              kernels_extra_channel = np.tile(kernels, (15, 1))
          
              #print(kernels_extra_channel)                                       
              target_layer.set_weights([kernels_extra_channel]) #, biases
              target_layer.trainable = False

            else:
              target_layer.set_weights(layer.get_weights())
              target_layer.trainable = False
            
            
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, classes=5)
    #vgg16.summary()
    # Get DenseNet201 config in dictionary format
    config = vgg16.get_config()
    config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, 15)
    config["layers"][2]["config"]["strides"] = (1, 1)

    vgg16_custom = tf.keras.models.Model.from_config(config)
    modify_name = config["layers"][2]["config"]["name"]
 
    # Create new model with config
    weightify(vgg16, vgg16_custom, modify_name)
    #vgg16_custom.summary()
    tf.keras.backend.clear_session() 
    
    for layer in vgg16_custom.layers[:150]: #150
       layer.trainable = False
    for layer in vgg16_custom.layers[150:]:
       layer.trainable = True

    #Inputs
       
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16_custom(input_tensor)
    layers = Flatten(name="flatten")(out1)

    #Hidden Layers
    layers = Dense(256)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)

    layers = Dense(128)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    layers = ReLU()(layers)
    #classification layer
    predictions = Dense(5, activation="softmax", name="output_1")(layers)
    modelFT = Model(inputs = input_tensor, outputs=predictions)
    
 
    modelFT.compile("adamax",
      loss="categorical_crossentropy",
      metrics=["accuracy"])
    modelFT.summary()
    return modelFT
