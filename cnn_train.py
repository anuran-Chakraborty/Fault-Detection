import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
import math
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix

def my_init(shape_array, dtype=None):

	print(shape_array)

	a = np.ndarray(shape=shape_array, dtype=float)
	delta_theta = 180 / shape_array[0]

	for i in range(shape_array[0]):
		val = math.sin(math.radians(i*delta_theta))
		delta_val = (1 - 2*math.sin(math.radians(i*delta_theta)))/shape_array[3]

		for j in range(shape_array[3]):
						
			a[i][0][0][j] = val
			val+= delta_val 

	print(a)		
	a = tf.convert_to_tensor(a, np.float)	
	return a


# Load the train data
df_train=pd.read_csv('single_train.csv',header=None)
X_train=df_train.iloc[:,0:-1]
Y_train=df_train.iloc[:,-1]
Y_train.astype(str,inplace=True)

# Load the test data
df_test=pd.read_csv('single_test.csv',header=None)
X_test=df_test.iloc[:,0:-1]
Y_test=df_test.iloc[:,-1]
Y_test.astype(str,inplace=True)

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
num_classes=27

# Reshape train and test
X_train=X_train.reshape(num_train,2500,5,1)
X_test=X_test.reshape(num_test,2500,5,1)

#one-hot encode target column
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)


# Creating the model
model=Sequential()

#add model layers
# kernel_initializer = my_init
model.add(Conv2D(15, kernel_size=(5,1), strides=(1,1), activation='relu', activity_regularizer=l2(0.001), input_shape=(2500,5,1)))
# model.add(MaxPooling2D(pool_size =(2, 1)))

model.add(Conv2D(10, kernel_size=(5,1), activation='relu', activity_regularizer=l2(0.001)))
# model.add(Conv2D(7, kernel_size=(5,1), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(15, kernel_size=(2,1), activation='relu'))
# model.add(Dropout=0.2)
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

model.save('single_pd.h5')

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

y_pred = model.predict(X_test).argmax(axis=1)
print(y_pred)

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred)
df=pd.DataFrame(matrix)
df.to_csv('conf_mat.csv')
print(type(matrix))