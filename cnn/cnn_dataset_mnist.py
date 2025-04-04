# -*- coding: utf-8 -*-
"""CNN - dataset MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mpFyNnV2WT1X-4CE2lzj4jP8356Yw3s1

Importo le librerie
"""

import numpy as np
import tensorflow as tf
import cv2 
from random import randint

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.utils import np_utils
from keras import Input
from keras import Model
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

"""Importo dataset → mnist"""

#Importo mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#concateno X e y in quanto il dataset è diviso
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

X_trainC=[]
X_testC=[]
for i in range(0,X_train.shape[0],1):
  X_trainC.append(cv2.cvtColor(X_train[i], cv2.COLOR_GRAY2BGR))
for i in range(0,X_test.shape[0],1):
  X_testC.append(cv2.cvtColor(X_test[i], cv2.COLOR_GRAY2BGR))

#X_trainC[0]
plt.imshow(X_testC[2])

#trasformo il vettore
#X_trainC= X_trainC.reshape(X_train.shape[0], 28, 28, 3)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 3)

X_trainC=np.array(X_trainC)
X_testC=np.array(X_testC)

X_trainC.shape, X_testC.shape, X.shape

#normalizzo i valori
X_trainC = X_trainC / 255
X_testC = X_testC / 255
#converto in float
X_trainC= X_trainC.astype('float32')
X_testC= X_testC.astype('float32')

#codifica one hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# shape[1] lunghezza stringa=numero di classi (10)
n_classi = y_train.shape[1]

"""Definisco la mia rete neurale"""

#creo modello
model = Sequential()
#convolutional layer
model.add(Conv2D(30, kernel_size=3, padding='valid', activation='relu', input_shape=(28,28,3)))
#model.add(Dropout(0.3))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(60, kernel_size=3, activation='relu'))
#model.add(Dropout(0.3))
model.add(MaxPool2D((2, 2)))
# flatten output of conv
model.add(Flatten())

# output layer
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#Compilo modello
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Alleno la mia rete neurale"""

progress=model.fit(X_trainC, y_train, validation_data=(X_testC, y_test), epochs=10, batch_size=128, shuffle=True)

"""Perdita accuratezza (accuracy)"""

plt.plot(progress.history['accuracy'])
plt.plot(progress.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Perdita dell'errore (loss)"""

plt.plot(progress.history['loss'])
plt.plot(progress.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#for i in range(0,test_predictions.shape[0],1):
#  max=0
#  for j in range(0, test_predictions.shape[1], 1):
#    if test_predictions[i][j]>max:
#      max=test_predictions[i][j]
#      print(max)
#  for k in range(0, test_predictions.shape[1], 1):
#    #print(str(test_predictions[0][k]))
#    if test_predictions[i][k]==max:
#      test_predictions[i][k]=1
#      #print(str(test_predictions[0][k]) + " " + str(k))
#    else:
#      test_predictions[i][k]=0

test_predictions = model.predict_classes(X_testC, batch_size=128, verbose=0)

test_labels=np.argmax(y_test, axis=1)

cm = confusion_matrix(test_labels, test_predictions)

cm