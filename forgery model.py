# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:19:17 2020

@author: ASUS
"""
import tensorflow as tf
import os

directory="CASIA 1.0 groundtruth/"
print(len(os.listdir("CASIA 1.0 groundtruth/CM")))
print(len(os.listdir("CASIA 1.0 groundtruth/Sp")))

os.mkdir("training/")
os.mkdir("training/CM")
os.mkdir("training/Sp")
os.mkdir("testing/")
os.mkdir("testing/CM")
os.mkdir("testing/Sp")

import train_test_set as tst
ydir="CASIA 1.0 groundtruth/CM/"
ndir="CASIA 1.0 groundtruth/Sp/"

ytrain="training/CM/"
ntrain="training/Sp/"

ytest="testing/CM/"
ntest="testing/Sp/"

tst.split(ydir,ytrain,ytest,0.8)
tst.split(ndir,ntrain,ntest,0.8)

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=0.4,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
       
        )
testdatagen=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=0.4,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        
        )

train_dataset=train_datagen.flow_from_directory(directory="training/",
                                                target_size=(224,224),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                batch_size=32
                                                )
test_set=testdatagen.flow_from_directory(directory="testing/",
                                         target_size=(224,224),
                                         class_mode='binary',
                                         color_mode='rgb',
                                         batch_size=16
                                         
                                        )
"""
model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
        #tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same'),
        #tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=64,activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1,activation='sigmoid')
        
        
        ]
        )
"""
vggmodel=tf.keras.applications.vgg16.VGG16(include_top=False,weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                           input_shape=(224,224,3))
for layer in vggmodel.layers:
    layer.trainable=False

model=tf.keras.Sequential([
        vggmodel,
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2,activation="softmax")
        ]
        )
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'],loss='sparse_categorical_crossentropy')


history=model.fit_generator(train_dataset,steps_per_epoch=50,epochs=10,validation_data=test_set,
                            validation_steps=30,
                            verbose=1
                            )
