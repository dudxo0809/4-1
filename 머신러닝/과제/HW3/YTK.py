
#path = 'C:/Users/ytkim/Desktop/4-1학기/머신러닝/과제/HW3/Multi-class Weather Dataset'
path = 'C:/Users/김영태/Desktop/4-1/머신러닝/과제/HW3/archive/Multi-class Weather Dataset'



import tensorflow as tf
print(tf.__version__)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_iris # 샘플 데이터 로딩
from sklearn.model_selection import train_test_split


import os
import sys
sys.path.append(path)

folders = os.listdir(path)

from keras_preprocessing.image import ImageDataGenerator

#import dataset

data_dir = path
dataset = data_dir

# train dataset

train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                   
                                   rotation_range=90,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.3,
                                   
                                   validation_split=0.2,
                                   vertical_flip=True)


train_generator = train_datagen.flow_from_directory(dataset, 
                                                    batch_size=16,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    target_size=(150,150))

# validation dataset

valid_generator = train_datagen.flow_from_directory(dataset,
                                                    batch_size = 16,
                                                    shuffle=True,
                                                    class_mode='categorical',
                                                    subset='validation',
                                                    target_size=(150,150))
    
# test dataset

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

test_generator = test_datagen.flow_from_directory(dataset,
                                                    batch_size = 16,
                                                    shuffle=True,
                                                    target_size=(150,150))


# AlexNet

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4,activation='softmax')  
])


# Model build

model.compile(optimizer = 'Adam', loss= tf.keras.losses.CategoricalCrossentropy() ,metrics=['accuracy'])
#model.summary()


# Train Model

history = model.fit(train_generator, validation_data = valid_generator,
                    epochs=30
                    ) 


# Import plt
import matplotlib.pyplot as plt


# Print Accuracy & Loss

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1) # jumlah row, jumlah column, column/row berapa
plt.plot(train_acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2) # jumlah row, jumlah column, column/row berapa
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Print plot
plt.show()


# import module confusion matrix & classification_report

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
new_model = model

# Model Prediction
probabilities = new_model.predict(valid_generator)


# Get Prediction value
predictions = []
for prob in probabilities:
   best_index = np.argmax(prob)
   predictions.append(best_index)


labels = valid_generator.classes
# tn,fp,fn,tp = confusion_matrix(labels, predictions).ravel()
cm = confusion_matrix(labels, predictions).ravel()
#cm = [[tp, fp],
#      [fn, tn]]

cr = classification_report(labels, predictions)
print(cm)   # confusion matrix
print(cr)   # classification report


# Evaluate model
loss, accuracy = model.evaluate(test_generator)


# Print Loss & Accuracy
print('Loss = {:.5f}'.format(loss))
print('Accuracy = {:.5f}'.format(accuracy))








