"""
Title: KNN
Author: Sophia Wewetzer
Date created: 2024/03/21
Last modified: 2023/07/10
Description: term paper, course: bionics
Accelerator: Apple Neural Engine (ANE)
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Ensure TensorFlow uses the Metal backend for acceleration on MacBook Pro
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

# Image files directory
DATADIR = "/Users/username/OneDrive/Soso/Schule/Bionik/Seminararbeit/knn/PetImages"

# Categories images
CATEGORIES = ['Cat', 'Dog']

# Image size
IMG_SIZE = 100

training_data = []

# Function for creating the training data
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        print(f"Processing category: {category}")
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                if img_path.lower().endswith('.jpg'):
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        print(f"Image not read correctly: {img_path}")
                        continue
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([img_array, class_num])
                else:
                    print(f"Skipping non-JPG file: {img_path}")
            except Exception as e:
                print(f"Error reading file {img}: {e}")

# Creating training data
create_training_data()

# Check if training data was loaded correctly
if len(training_data) == 0:
    print("No training data loaded.")
else:
    print(f"{len(training_data)} training data successfully loaded.")

# Shuffle training data
random.shuffle(training_data)

X = []
y = []

# Extract the features and labels
for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X)
y = np.array(y)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Normalize the data
X = X / 255.0

# Save training data
with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)

# Load training data
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# Define the model
model = Sequential()

# Layer 1
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(64))

# Output Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Check if we have enough data
if len(X) == 0 or len(y) == 0:
    print("Not enough data to train the model.")
else:
    model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)