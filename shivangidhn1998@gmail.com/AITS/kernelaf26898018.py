# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# K-Nearest Neighbors (K-NN)

# Importing the dataset
dataset = pd.read_csv('/kaggle/input/nasa.csv')
dataset['Close Approach Date'] = pd.to_datetime(dataset['Close Approach Date'])
dataset['Orbit Determination Date'] = pd.to_datetime(dataset['Orbit Determination Date'])
dataset['year0'] = dataset['Close Approach Date'].dt.year 
dataset['month0'] = dataset['Close Approach Date'].dt.month 
dataset['day0'] = dataset['Close Approach Date'].dt.day 
dataset['year1'] = dataset['Orbit Determination Date'].dt.year 
dataset['month1'] = dataset['Orbit Determination Date'].dt.month 
dataset['day1'] = dataset['Orbit Determination Date'].dt.day 
dataset['hour1'] = dataset['Orbit Determination Date'].dt.hour 
dataset['minute1'] = dataset['Orbit Determination Date'].dt.minute 
dataset['second1'] = dataset['Orbit Determination Date'].dt.second
X = dataset.iloc[:, [2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,21,23,25,26,27,28,29,30,31,32,33,34,35,36,37,40,41,42,43,44,45,46,47,48]].values
Y = dataset.iloc[:, 39].values
Z = pd.DataFrame(X)

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()
X[:, 17] = labelencoder_X.fit_transform(X[:, 17])
X[:, 19] = labelencoder_X.fit_transform(X[:, 19])
X[:, 32] = labelencoder_X.fit_transform(X[:, 32])
X[:, 33] = labelencoder_X.fit_transform(X[:, 33])
X[:, 34] = labelencoder_X.fit_transform(X[:, 34])
X[:, 35] = labelencoder_X.fit_transform(X[:, 35])
X[:, 36] = labelencoder_X.fit_transform(X[:, 36])
X[:, 37] = labelencoder_X.fit_transform(X[:, 37])
onehotencoder = OneHotEncoder(categorical_features = [[17,19,32,33,34,35,36,37]])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
