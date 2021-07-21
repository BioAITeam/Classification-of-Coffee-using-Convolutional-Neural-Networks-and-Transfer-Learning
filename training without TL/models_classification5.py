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
    
            
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.VGG16(weights=None,input_tensor=None, include_top=False, classes=5, input_shape=(224,224,15))
    #vgg16.summary()
    
    #Inputs
    
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16(input_tensor)
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

   
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.VGG19(weights=None,input_tensor=None, include_top=False, classes=5, input_shape=(224,224,15))
    #vgg16.summary()
    
    #Inputs
    
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16(input_tensor)
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

  
            
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.InceptionResNetV2(weights=None,input_tensor=None, include_top=False, classes=5, input_shape=(224,224,15))
    #vgg16.summary()

    #Inputs

    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16(input_tensor)
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

 
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.InceptionV3(weights=None,input_tensor=None, include_top=False, classes=5, input_shape=(224,224,15))
    #vgg16.summary()
    
    #Inputs
    
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16(input_tensor)
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

 
    inputs = Input(input_size)
    vgg16 = tf.keras.applications.DenseNet201(weights=None,input_tensor=None, include_top=False, classes=5, input_shape=(224,224,15))
    #vgg16.summary()

    
    input_tensor = Input(shape=(img_height,img_height,15))
    out1 = vgg16(input_tensor)
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
