import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np

# Load the train data
df_train=pd.read_csv('train.csv',header=None)
X_train=df_train.iloc[:,0:-1]
Y_train=df_train.iloc[:,-1]

# Load the test data
df_test=pd.read_csv('test.csv',header=None)
X_test=df_test.iloc[:,0:-1]
Y_test=df_test.iloc[:,-1]

# Normalise
min_max_scaler = preprocessing.Normalizer()
min_max_scaler.fit(X_train)
pickle.dump(min_max_scaler, open("normaliser.pickle", "wb"))
X_train=min_max_scaler.transform(X_train)
X_test=min_max_scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

num_train=X_train.shape[0]
num_test=X_test.shape[0]
num_classes=20

# Reshape train and test
X_train=X_train.reshape(num_train,2500,5)
X_test=X_test.reshape(num_test,2500,5)

#one-hot encode target column
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)


# Creating the model
model=Sequential()

#add model layers
model.add(CuDNNLSTM(num_train, input_shape=(2500, 5),return_sequences= True))
model.add(CuDNNLSTM(num_train))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))


