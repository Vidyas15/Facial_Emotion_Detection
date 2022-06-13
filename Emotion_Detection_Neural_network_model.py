# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 01:09:15 2019

@author: vidya
"""

import numpy as np
import cv2
import pandas as pd
from sklearn.decomposition import PCA
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pickle
import matplotlib.pyplot as plt
from keras.optimizers import SGD

#import the dataset
raw_data = pd.read_excel('64x64.xlsx')
X = raw_data.iloc[:, :4096].values
Y = raw_data.iloc[:, 4096]
 
#Normalization of input data
X_Norm = X / 255.0
y_categorised = keras.utils.to_categorical(Y, num_classes = 3)

# build and train model
top_layer_model = Sequential()
top_layer_model.add(Dense(1024, input_shape=(4096,), activation='relu'))
top_layer_model.add(Dropout(0.5))
top_layer_model.add(Dense(512, input_shape=(1024,), activation='relu'))
top_layer_model.add(Dropout(0.5))
top_layer_model.add(Dense(128, input_shape=(512,)))
top_layer_model.add(Dropout(0.5))
top_layer_model.add(Dense(3, activation='softmax'))


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
top_layer_model.compile(loss='categorical_crossentropy',
                            optimizer=sgd, metrics=['accuracy'])

history = top_layer_model.fit(X_Norm, y_categorised, epochs=28, batch_size=20)
score = top_layer_model.evaluate(X_Norm, y_categorised, batch_size=20)

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#saving the model
with open('emotion_detector_NN', 'wb') as f:
	pickle.dump(top_layer_model, f)
	
#Plotting confusion Matrix 
y_test_output = top_layer_model.predict(X_Norm, batch_size=20)
y_pred = np.zeros(shape=(262,1))
for i in range(len(y_test_output)):
    for j in range(len(y_test_output[i])):
        k = y_test_output[i].max()
        if (y_test_output[i][j] == k):
            y_pred[i] = j
            
Confusion_Matrix = np.zeros(shape=(3, 3), dtype=int)
for i in range(len(Y)):
    j = y_pred[i][0].astype(int)
    k = Y[i]
    Confusion_Matrix[k][j] = Confusion_Matrix[k][j] + 1
