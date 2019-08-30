import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import to_categorical
from keras.optimizers import Adam 
from keras. initializers import TruncatedNormal 
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import Data
raw_data = load_wine()

# Brief Description of Data
print(raw_data['DESCR'])

# Creating Features and Target Variables
X = raw_data['data']
y = raw_data['target']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Converting to Y train and test to Categorical 
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Fitting Classifier to the Training set
input_dim = len(X_train[0, :])
class_num = len(y_train[0, :]) 

model = Sequential()

#config model 
model.add(Dense(units=50, activation='relu', input_dim=input_dim))
model.add(Dense(units=class_num, activation='softmax'))

#optimizer
adam = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit to the network
train = model.fit(X_train, y_train, epochs=800, validation_data=(X_test, y_test), shuffle=False, verbose=1)

# Finding the accuracy
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
