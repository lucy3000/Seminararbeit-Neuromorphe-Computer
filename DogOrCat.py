"""
Title: KNN
Author: Sophia Wewetzer
Date created: 2024/03/21
Last modified: 2023/07/10
Description: term paper, course: bionics
Accelerator: none
"""

# Import libraries
import numpy as np  # used for working with arrays.
import matplotlib.pyplot as plt  # for mathematical representations of all kinds
import os  # portable API of system services
import cv2  # pip install opencv-python (for image processing and computer vision)
import random
import pickle  # binary serialization format
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Backend framework tensorflow
os.environ["KERAS_BACKEND"] = "tensorflow" 

# Image files directory
DATADIR = "/Dateipfad eingeben/"

# Categories images
CATEGORIES = ['Cat', 'Dog']

# Image size
IMG_SIZE = 100

training_data = []

# Function for creating the training data
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        print(f"Processing category: {category}")  # Debugging information
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                # Ensure images are JPG
                if img_path.lower().endswith('.jpg'):
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        print(f"Image not read correctly: {img_path}")  # Debugging information                 
                        continue
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([img_array, class_num])
                else:
                    print(f"Skipping non-JPG file: {img_path}")  # Debugging information
            except Exception as e:
                print(f"Fehler beim Lesen der Datei {img}: {e}")

# Creating training data
create_training_data()

# Check if training data was loaded correctly
if len(training_data) == 0:
    print("Keine Trainingsdaten geladen.")
else:
    print(f"{len(training_data)} Trainingsdaten erfolgreich geladen.")

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

# Save training data
with open("X.pickle", "wb") as pickle_out: #öffnet Datei im Binärmodus (wb=write binary)
    pickle.dump(X, pickle_out) #schreibt Objekt X in geöffnete Datei

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)

# Load training data

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

# Layer 1
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

# Layer 2
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

# Layer 3
model.add(Flatten())
model.add(Dense(64))

# Output Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

# Check if we have enough data
if len(X) == 0 or len(y) == 0:
    print("Nicht genug Daten zum Trainieren des Modells.")
else:
    model.fit(X, y, batch_size = 32, epochs = 3, validation_split = 0.1)
