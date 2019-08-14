import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
## loading data 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y_train = train["label"]


X_train = train.drop(labels = ["label"],axis = 1) 


del train 

Y_train.value_counts()
## checking null and missing values
X_train.isnull().any().describe()
test.isnull().any().describe()
## normalization 
X_train = X_train / 255.0
test = test / 255.0
## reshaping 
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
## label encoding 
Y_train = to_categorical(Y_train, num_classes = 10)
## split training and validation set
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
## CNN model 
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
## setting optimizer and annealer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

## learning process 
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 15
batch_size = 50
history=model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs, 
         validation_data = (X_val, Y_val), verbose = 2)