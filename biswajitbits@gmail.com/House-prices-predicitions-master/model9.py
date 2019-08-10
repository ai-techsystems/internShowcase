import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import pandas as pd
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

all_data.info()

all_data.head()

# Categorical Feature
cat_feats = all_data.dtypes[all_data.dtypes == "object"].index
cat_feats

all_data[cat_feats].head()

# Numeric Feature
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
numeric_feats

all_data[numeric_feats].head()


# Ordinal Feature
ordinal_features = ['YrSold']

#log transform skewed numeric features:

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print(skewed_feats)

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

# One Hot Encoder
all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


X_train = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
X_tr.shape


X_tr

X_val


model = Sequential()
BatchNormalization()
model.add(Dense(1024,input_dim=288,activation='relu', kernel_initializer='normal'))
BatchNormalization()
Dropout(0.5)
model.add(Dense(512,input_dim=1024,activation='relu', kernel_initializer='normal'))
BatchNormalization()
Dropout(0.5)
model.add(Dense(256,activation='relu', kernel_initializer='normal'))
BatchNormalization()
Dropout(0.5)
model.add(Dense(512,activation='relu', kernel_initializer='normal'))
BatchNormalization()
Dropout(0.5)
model.add(Dense(512,activation='relu', kernel_initializer='normal'))
BatchNormalization()
Dropout(0.5)
model.add(Dense(1))
model.compile( optimizer='adam',loss='mse',metrics=['mean_squared_error'])



model.fit(X_tr,y_tr,validation_data=(X_val,y_val),nb_epoch=20,batch_size=100,verbose=2)

model.summary()



preds = model.predict(np.array(X_val))
print(preds.shape)
print(X_val.shape)
print(y_val.shape)


print(y_val.mean())
print(preds.mean())
print(rmse(preds[0],y_val))

preds = model.predict(np.array(X_test))

sub = pd.read_csv("./sample_submission.csv")

sub.iloc[:,1] = np.array(model.predict(np.array(X_test)))
print(sub[['SalePrice']].mean())
sub['SalePrice'] = np.expm1(sub[['SalePrice']])
print(sub[['SalePrice']].mean())
sub.to_csv('kerassubmission9.csv', index=None)
print(sub.to_csv)
