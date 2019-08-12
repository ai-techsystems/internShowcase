#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 8 17:33:19 2019

@author: gorasiya ravi
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report,confusion_matrix
data = sns.load_dataset("iris")
X = data.iloc[:,0:4].values
y = data.iloc[:,4].values

encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
model = Sequential()

model.add(Dense(4,input_shape=(4,),activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100)

y_pre = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pre,axis=1)


print(classification_report(y_test_class,y_pred_class))