import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import model_from_json
import numpy as np
import os

# fix random seed for reproducibility
# np.random.seed(7)
filename='train_clean_shuffled.csv'

dataset = np.loadtxt(filename, delimiter=",")
size=dataset.shape[0]
classes = np.unique(dataset[:,-1])
num_classes=classes.size

print(num_classes)

train_x , train_y = dataset[:,0:-1] , dataset[:,-1]

train_y = to_categorical(train_y)  #converts to one hot

feature_size=train_x.shape[1]

# create model
model = Sequential()
model.add(Dense(10, input_dim=feature_size, activation='relu'))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))

model.add(Dense(25, activation='relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))

model.add(Dense(60, activation='relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))

model.add(Dense(70, activation='relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))


model.add(Dense(40, activation='relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))

model.add(Dense(num_classes, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# Fit the model
model.fit(train_x, train_y, epochs=1200, batch_size=1000)
score = model.evaluate(train_x, train_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))








