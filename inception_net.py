import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np

# Load the train data
df_train=pd.read_csv('train.csv',header=None)
Xtrain=df_train.iloc[:,0:-1]
Ytrain=df_train.iloc[:,-1]

# Load the test data
df_test=pd.read_csv('test.csv',header=None)
Xtest=df_test.iloc[:,0:-1]
Ytest=df_test.iloc[:,-1]

# Normalise
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(Xtrain)
pickle.dump(min_max_scaler, open("normaliser.pickle", "wb"))
X_train=min_max_scaler.transform(Xtrain)
X_test=min_max_scaler.transform(Xtest)

print(X_train.shape)
print(X_test.shape)

# Reshape train and test
X_train=X_train.reshape(81,2500,5,1)
X_test=X_test.reshape(54,2500,5,1)

#one-hot encode target column
y_train = to_categorical(Ytrain)
y_test = to_categorical(Ytest)

#==============================================================================================
# Creating the model
input_img=Input(shape=(2500,5,1))

input_1=# Sensor 1 vs sensor 2

tower_1 = Conv2D(15, (5,1), activation='relu')(input_img)
tower_1 = Conv2D(8, (5,1), activation='relu')(tower_1)

tower_2 = Conv2D(10, (5,1), activation='relu')(input_img)
tower_2 = Conv2D(12, (5,1), activation='relu')(tower_2)

# tower_3 = MaxPooling2D((3,0), strides=(1,1), padding='same')(input_img)
tower_3 = Conv2D(15, (1,1), activation='relu')(tower_2)

output = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)
output = Flatten()(output)
out    = Dense(27, activation='softmax')(output)
model = Model(inputs = input_img, outputs = out)

#==============================================================================================

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))