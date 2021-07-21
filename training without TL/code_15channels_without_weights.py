import glob, os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import  MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.layers import Lambda, Layer, ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Lambda, Layer, ReLU
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import datasets,layers,models,Input,Model
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from scipy.stats import percentileofscore
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score 
from time import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns
from models_classification5 import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
tiempo_inicial = time()

## Functions
def create_model_2(out1):

  model = Sequential()
  model.add(out1)
  model.add(Flatten())
  model.add(Dense(256))
  model.add(keras.layers.BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dense(128))
  model.add(keras.layers.BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dense(5, activation='softmax'))
  model.compile("adamax",loss="categorical_crossentropy",metrics=["accuracy"])
  
  return model

aug = ImageDataGenerator(
		rotation_range=45,
		zoom_range=0.12,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.12,
		horizontal_flip=True,
		fill_mode="nearest")


DESIRED_ACCURACY = 0.999
INIT_LR = 1e-4
def exp_decay(epoch):
  initial_lrate = 1e-4
  k = 0.095
  lrate = initial_lrate * np.exp(-k*epoch)
  return lrate



class LossHistory(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
    self.lr = []

  def on_epoch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
    self.lr.append(exp_decay(len(self.losses)))

class end_train(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,log={}):
            if log.get('val_accuracy')>DESIRED_ACCURACY:
                print("\nReached"+str(DESIRED_ACCURACY)+"% val_accuracy so cancelling training!")
                self.model.stop_training = True
end_t = end_train()
loss_history = LossHistory()
lrate = LearningRateScheduler(exp_decay)       
callbacks_list = [loss_history,lrate,end_t]

callbacks=callbacks_list



#channel variables
          
img_width = 224
img_height = 224
input_channel = 15





print("______________________________________________________________________________________")
print("EXPERIMENT 1: UNBALANCED")

## Load Database
LoadFotosCafe15 = np.load("/clinicfs/userhomes/rtabares/Experimentos_Covid_Jp/Proyecto_cafe/dataset_15Canales_224x224.npy")

## Data distribution

seco = LoadFotosCafe15[:78]
maduro = LoadFotosCafe15[78:238]
semimaduro = LoadFotosCafe15[238:398]
sobremaduro = LoadFotosCafe15[398:510]
verde = LoadFotosCafe15[510:640]

#Choose the data
seco_c = seco[:]
maduro_c = maduro[:]
semimaduro_c = semimaduro[:]
sobremaduro_c = sobremaduro[:]
verde_c = verde[:]


seco = seco[:]
maduro = maduro[:]
semimaduro = semimaduro[:]
sobremaduro = sobremaduro[:]
verde = verde[:]

# Create photo array 15 channels and create label vector

Data_Cafe = np.concatenate((seco,maduro,semimaduro,sobremaduro,verde), axis = 0)
Label_Cafe = np.int32(np.concatenate((0*np.ones((78)),1*np.ones((88)),2*np.ones((95)),3*np.ones((90)),4*np.ones((100))),axis = 0))


Data_Cafe_c = np.concatenate((seco_c,maduro_c,semimaduro_c,sobremaduro_c,verde_c), axis = 0)
Label_Cafe_c = np.int32(np.concatenate((0*np.ones((seco_c.shape[0])),1*np.ones((maduro_c.shape[0])),2*np.ones((semimaduro_c.shape[0])),
                        3*np.ones((sobremaduro_c.shape[0])),4*np.ones((verde_c.shape[0]))),axis = 0)) 

X_train3, X_test3, y_train3, y_test3 = train_test_split(Data_Cafe_c, Label_Cafe_c, test_size=0.3, random_state=0)

X_train = X_train3
X_test = X_test3
y_train = y_train3
y_test= y_test3

print(X_train3.shape)
print(X_test3.shape)

y_train  = to_categorical(y_train, 5)
y_test  = to_categorical(y_test, 5)


print(y_train.shape)
print(y_test.shape)

print("________________________________Vgg16_________________________________________________")




modelFT = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results




print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Vgg19_________________________________________________")

modelFT = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Inception_V2_________________________________________________")


modelFT = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________Inception V3_________________________________________________")


modelFT = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________DenseNet201_________________________________________________")



modelFT = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")



out1 = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))



print("______________________________________________________________________________________")
print("EXPERIMENT 2: DOWNSAMPLIN")
## Load Database
LoadFotosCafe15 = np.load("/clinicfs/userhomes/rtabares/Experimentos_Covid_Jp/Proyecto_cafe/dataset_15Canales_224x224.npy")

## Data distribution

seco = LoadFotosCafe15[:78]
maduro = LoadFotosCafe15[78:238]
semimaduro = LoadFotosCafe15[238:398]
sobremaduro = LoadFotosCafe15[398:510]
verde = LoadFotosCafe15[510:640]

#Choose the data
seco_c = seco[:]
maduro_c = maduro[:]
semimaduro_c = semimaduro[:]
sobremaduro_c = sobremaduro[:]
verde_c = verde[:]

# Create photo array 15 channels and create label vector
seco = seco[:78]
maduro = maduro[:78]
semimaduro = semimaduro[:78]
sobremaduro = sobremaduro[:78]
verde = verde[:78]


# Create photo array 15 channels and create label vector
Data_Cafe = np.concatenate((seco,maduro,semimaduro,sobremaduro,verde), axis = 0)
Label_Cafe = np.int32(np.concatenate((0*np.ones((78)),1*np.ones((88)),2*np.ones((95)),3*np.ones((90)),4*np.ones((100))),axis = 0))


Data_Cafe_c = np.concatenate((seco,maduro,semimaduro,sobremaduro,verde), axis = 0)
Label_Cafe_c = np.int32(np.concatenate((0*np.ones((seco.shape[0])),1*np.ones((maduro.shape[0])),2*np.ones((semimaduro.shape[0])),
                        3*np.ones((sobremaduro.shape[0])),4*np.ones((verde.shape[0]))),axis = 0)) 

X_train3, X_test3, y_train3, y_test3 = train_test_split(Data_Cafe_c, Label_Cafe_c, test_size=0.3, random_state=0)

X_train = X_train3
X_test = X_test3
y_train = y_train3
y_test= y_test3

print(X_train3.shape)
print(X_test3.shape)

y_train  = to_categorical(y_train, 5)
y_test  = to_categorical(y_test, 5)


print(y_train.shape)
print(y_test.shape)


print(X_train.shape)
print(X_test.shape)



print("________________________________Vgg16_________________________________________________")




modelFT = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results




print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Vgg19_________________________________________________")



modelFT = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Inception_V2_________________________________________________")



modelFT = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________Inception V3_________________________________________________")



modelFT = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________DenseNet201_________________________________________________")



modelFT = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")



out1 = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("EXPERIMENT 3: OVERSAMPLING")

## Load Database
LoadFotosCafe15 = np.load("/clinicfs/userhomes/rtabares/Experimentos_Covid_Jp/Proyecto_cafe/dataset_15Canales_224x224.npy")

## Data distribution

seco = LoadFotosCafe15[:78]
maduro = LoadFotosCafe15[78:238]
semimaduro = LoadFotosCafe15[238:398]
sobremaduro = LoadFotosCafe15[398:510]
verde = LoadFotosCafe15[510:640]

#Choose the data
seco_c = seco[:]
maduro_c = maduro[:]
semimaduro_c = semimaduro[:]
sobremaduro_c = sobremaduro[:]
verde_c = verde[:]

seco = seco[:]
maduro = maduro[:]
semimaduro = semimaduro[:]
sobremaduro = sobremaduro[:]
verde = verde[:]

# Create photo array 15 channels and create label vector
Data_Cafe = np.concatenate((seco,maduro,semimaduro,sobremaduro,verde), axis = 0)
Label_Cafe = np.int32(np.concatenate((0*np.ones((78)),1*np.ones((88)),2*np.ones((95)),3*np.ones((90)),4*np.ones((100))),axis = 0)) 

Data_Cafe_c = np.concatenate((seco_c,maduro_c,semimaduro_c,sobremaduro_c,verde_c), axis = 0)
Label_Cafe_c = np.int32(np.concatenate((0*np.ones((seco_c.shape[0])),1*np.ones((maduro_c.shape[0])),2*np.ones((semimaduro_c.shape[0])),
                        3*np.ones((sobremaduro_c.shape[0])),4*np.ones((verde_c.shape[0]))),axis = 0)) 

X_train3, X_test3, y_train3, y_test3 = train_test_split(Data_Cafe_c, Label_Cafe_c, test_size=0.3, random_state=0)

X_train = X_train3
X_test = X_test3
y_train = y_train3
y_test= y_test3

print(X_train3.shape)
print(X_test3.shape)

y_train  = to_categorical(y_train, 5)
y_test  = to_categorical(y_test, 5)


print(y_train.shape)
print(y_test.shape)






print("________________________________Vgg16_________________________________________________")



modelFT = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(aug.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch= 20,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results




print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(aug.flow(inputs[train], targets[train], batch_size=16), steps_per_epoch= 20, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Vgg19_________________________________________________")



modelFT = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(aug.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch= 20,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(aug.flow(inputs[train], targets[train], batch_size=16), steps_per_epoch= 20, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Inception_V2_________________________________________________")


modelFT = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(aug.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch= 20,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(aug.flow(inputs[train], targets[train], batch_size=16), steps_per_epoch= 20, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________Inception V3_________________________________________________")


modelFT = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(aug.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch= 20,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(aug.flow(inputs[train], targets[train], batch_size=16), steps_per_epoch= 20, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________DenseNet201_________________________________________________")



modelFT = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(aug.flow(x=X_train, y=y_train, batch_size=16), steps_per_epoch= 20,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")



out1 = modelFT = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(aug.flow(inputs[train], targets[train], batch_size=16), steps_per_epoch= 20, epochs=75, verbose=0)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))






print("______________________________________________________________________________________")
print("EXPERIMENT 4: WEIGHTING")

## Load Database
LoadFotosCafe15 = np.load("/clinicfs/userhomes/rtabares/Experimentos_Covid_Jp/Proyecto_cafe/dataset_15Canales_224x224.npy")

## Data distribution

seco = LoadFotosCafe15[:78]
maduro = LoadFotosCafe15[78:238]
semimaduro = LoadFotosCafe15[238:398]
sobremaduro = LoadFotosCafe15[398:510]
verde = LoadFotosCafe15[510:640]

#Choose the data
seco_c = seco[:]
maduro_c = maduro[:]
semimaduro_c = semimaduro[:]
sobremaduro_c = sobremaduro[:]
verde_c = verde[:]


seco = seco[:]
maduro = maduro[:]
semimaduro = semimaduro[:]
sobremaduro = sobremaduro[:]
verde = verde[:]

# Create photo array 15 channels and create label vector
Data_Cafe = np.concatenate((seco,maduro,semimaduro,sobremaduro,verde), axis = 0)
Label_Cafe = np.int32(np.concatenate((0*np.ones((78)),1*np.ones((88)),2*np.ones((95)),3*np.ones((90)),4*np.ones((100))),axis = 0)) 

Data_Cafe_c = np.concatenate((seco_c,maduro_c,semimaduro_c,sobremaduro_c,verde_c), axis = 0)
Label_Cafe_c = np.int32(np.concatenate((0*np.ones((seco_c.shape[0])),1*np.ones((maduro_c.shape[0])),2*np.ones((semimaduro_c.shape[0])),
                        3*np.ones((sobremaduro_c.shape[0])),4*np.ones((verde_c.shape[0]))),axis = 0)) 

X_train3, X_test3, y_train3, y_test3 = train_test_split(Data_Cafe_c, Label_Cafe_c, test_size=0.3, random_state=0)



### class_weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train3),
                                                 y_train3)
class_weights = dict(enumerate(class_weights))
print(class_weights)


X_train = X_train3
X_test = X_test3
y_train = y_train3
y_test= y_test3

print(X_train3.shape)
print(X_test3.shape)

y_train  = to_categorical(y_train, 5)
y_test  = to_categorical(y_test, 5)


print(y_train.shape)
print(y_test.shape)




print("________________________________Vgg16_________________________________________________")




modelFT = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks, class_weight = class_weights)

#### Results




print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg16_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0, class_weight = class_weights)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Vgg19_________________________________________________")

modelFT = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks, class_weight = class_weights)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = vgg19_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0, class_weight = class_weights)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

print("________________________________Inception_V2_________________________________________________")


modelFT = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks, class_weight = class_weights)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv2_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0, class_weight = class_weights)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________Inception V3_________________________________________________")


modelFT = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks, class_weight = class_weights)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")


out1 = Inceptionv3_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0, class_weight = class_weights)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))


print("________________________________DenseNet201_________________________________________________")



modelFT = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))


X_train, X_test3_1, y_train, y_test3_1 = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

history = modelFT.fit(x=X_train, y=y_train, batch_size=32,  epochs=100,  verbose=0, validation_data=(X_test3_1,y_test3_1), shuffle=True, callbacks=callbacks, class_weight = class_weights)

#### Results



print(modelFT.evaluate(X_test, y_test, verbose=0))
predictions=modelFT.predict(X_test)
predictions = np.argmax(predictions, axis=-1)
Y_validation= np.argmax(y_test, axis=-1)

print(predictions.shape)
print(Y_validation.shape)


print('Accuracy:', accuracy_score(Y_validation, predictions))
print('F1 score:', f1_score(Y_validation, predictions, average='macro'))
print('Recall:', recall_score(Y_validation, predictions, average='macro'))
print('Precision:', precision_score(Y_validation, predictions, average='macro'))
print('\n clasification report:\n', classification_report(Y_validation, predictions))
print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))


print("________________________________K-folds_________________________________________________")



out1 = DenseNet201_3canal_sin_pesos(input_size=(img_height, img_width, input_channel))

acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train3, X_test3), axis=0)
targets = np.concatenate((y_train3, y_test3), axis=0)

targets  = to_categorical(targets, 5)

print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = create_model_2(out1)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=75, verbose=0, class_weight = class_weights)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

