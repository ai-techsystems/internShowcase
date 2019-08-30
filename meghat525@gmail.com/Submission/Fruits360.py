#CNN
#Part 1-Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Initialising the CNN
classifier=Sequential()
#Step 1-Convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
#Step 2- Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Adding second convolutional layer
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step 3-Flattening
classifier.add(Flatten())
#Step 4-Full Connection
classifier.add(Dense(activation="relu",units=128))
classifier.add(Dense(activation="softmax",units=114))
#Compiling the ANN
classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
#Fitting the dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Training', #Directory
        target_size=(64, 64), #Same as that in input_shape
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'Test',
        target_size=(64, 64),
        batch_size=50,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=57276, #No. of images in training set
        epochs=7, #No. of epochs
        validation_data=test_set,
        validation_steps=19548) #No. of images in test set