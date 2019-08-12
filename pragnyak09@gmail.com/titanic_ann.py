
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_tr=pd.read_csv("X_train.csv")
y_tr=pd.read_csv("y_train.csv")


X=X_tr.iloc[:,1:].values
y=y_tr.iloc[:,1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',input_dim=7))
#Second  Hidden Layer
classifier.add(Dense(3, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#loss value and metrics value
eval_model=classifier.evaluate(X_train, y_train)
eval_model

#prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



