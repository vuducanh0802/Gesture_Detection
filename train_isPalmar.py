import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import os
import cv2


def load_images_from_folder(folder):
    images = []
    count = 0
    for filename in os.listdir(folder):
        count += 1
        img = cv2.imread(os.path.join(folder, filename))

        #make smaller images (can choose another instead of 30)
        width = int(img.shape[1] / 30)
        height = int(img.shape[0] / 30)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if img is not None:
            images.append(img)
    return images


data = np.array(load_images_from_folder('/content/drive/MyDrive/DATA/Hands'))

#Result csv
df = pd.read_csv('HandInfo.csv')

#for time saving purpose, I just train first ~1200 images
df.aspectOfHand[:data.shape[0]].value_counts()

#Label Encoding
df.aspectOfHand = df.aspectOfHand.replace({'dorsal right':0, 'dorsal left':0,'palmar right':1,'palmar left':1 })

#Training
X = data
y = np.array(df.aspectOfHand[:data.shape[0]])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.1, random_state = 2021)

#Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

#Deep Learning
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(40, 53, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=30)

#Evaluate
print(cnn.evaluate(X_test,y_test))
#>>> 0.6

cnn.save("model"+".h5")



