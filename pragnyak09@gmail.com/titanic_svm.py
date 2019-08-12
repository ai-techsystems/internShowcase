

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_tr=pd.read_csv("X_train.csv")
y_tr=pd.read_csv("y_train.csv")


X=X_tr.iloc[:,1:].values
y=y_tr.iloc[:,1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0,decision_function_shape='ovo',probability=True,degree=1,C=3)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#applied grid search for hyperparameter tuning
'''
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

parameters_svm = {'C':[1,2,3,4,5],'kernel':['rbf','linear'], 'gamma':[0,0.1,'auto'], 'probability':[True,False],
                  'random_state':[0,23,46],'decision_function_shape':['ovo','ovr'],'degree':[3,4,10]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_'''