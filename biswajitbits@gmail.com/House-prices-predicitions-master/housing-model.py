import numpy as np
import pandas as pd
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train[['SalePrice']].mean())
y = np.log1p(train[['SalePrice']])
print(y.mean())
y = y.SalePrice
print(y.mean())

total = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                    test.loc[:,'MSSubClass':'SaleCondition']))

total = pd.get_dummies(total)
total = total.fillna(total.mean())
x_train = np.array(total[:train.shape[0]])
x_test = np.array(total[test.shape[0]+1:])
X_tr, X_val, y_tr, y_val = train_test_split(x_train, y)

X_val.shape
model = Sequential()
BatchNormalization()
model.add(Dense(1024,input_dim=288,activation='relu'))
BatchNormalization()
Dropout(0.5)
model.add(Dense(256,input_dim=1024,activation='relu'))
BatchNormalization()
Dropout(0.8)
model.add(Dense(128))
BatchNormalization()
model.add(Dense(1))
model.compile( optimizer='adam',loss='mse',metrics=['mean_squared_error'])


model.fit(X_tr,y_tr,validation_data=(X_val,y_val),nb_epoch=35,batch_size=100,verbose=2)

model.summary()


preds = model.predict(np.array(X_val))
print(preds.shape)
print(X_val.shape)
print(y_val.shape)


print(y_val.mean())
print(preds.mean())
print(rmse(preds[0],y_val))

preds = model.predict(np.array(x_test))

sub = pd.read_csv("./sample_submission.csv")

sub.iloc[:,1] = np.array(model.predict(np.array(x_test)))
print(sub[['SalePrice']].mean())
sub['SalePrice'] = np.expm1(sub[['SalePrice']])
print(sub[['SalePrice']].mean())
sub.to_csv('kerassubmission1.csv', index=None)
print(sub.to_csv)
