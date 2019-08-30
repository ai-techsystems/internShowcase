import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

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

# Fitting Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy = sklearn.metrics.accuracy_score(y_pred, y_test)
print("Accuracy: ", '%.2f'% (accuracy*100),"%")

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


