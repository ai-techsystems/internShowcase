import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

classifier =  Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print(classifier.summary())

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1/1.255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'chest_xray/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = train_datagen.flow_from_directory(
        'chest_xray/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

val_set = train_datagen.flow_from_directory(
        'chest_xray/val',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cl=classifier.fit_generator(
        training_set,
        steps_per_epoch=163,
        epochs=10,
        validation_data=test_set,
        validation_steps=624/32)
 

from sklearn.metrics import classification_report, confusion_matrix




labels = test_set.class_indices

labels = {v: k for k, v in labels.items()}

classes = list(labels.values())

y_pred = classifier.predict(test_set)

y_pred =(y_pred>0.5)

print(confusion_matrix(test_set.classes, y_pred))

print(classification_report(test_set.classes, y_pred, target_names=classes))
print(labels)


test_accuracy=classifier.evaluate_generator(test_set,624)
print("Test accuracy:",test_accuracy[1]*100,'%')


from sklearn.metrics import accuracy_score
print(accuracy_score(test_set.classes, y_pred))

from sklearn.metrics import recall_score
recall_score(test_set.classes, y_pred)

from sklearn.metrics import precision_score
precision_score(test_set.classes, y_pred)

from sklearn.metrics import f1_score
f1_score(test_set.classes, y_pred)




import matplotlib.pyplot as plt

plt.plot(cl.history['acc'])
plt.plot(cl.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training_set', 'test_set'], loc='upper left')
plt.show()

#Loss
plt.plot(cl.history['loss'])
plt.plot(cl.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training_set', 'test_set'], loc='upper left')
plt.show()
