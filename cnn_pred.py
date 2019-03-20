import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np


# Load the model
model=load_model('single_pd.h5')
# Load the data
df=pd.read_csv('multi_pd.csv',header=None)

df_X=df.iloc[:,:-1]
df_Y=df.iloc[:,-1]

# Normalise here
with open('normaliser.pickle', 'rb') as handle:
    normalizer = pickle.load(handle)
    arr_X=normalizer.transform(df_X)

arr_Y=df_Y.values

correct=0
cnt=0
for i in range(len(arr_X)):
	x=arr_X[i]
	y=arr_Y[i]
	x=x.reshape(1,2500,5,1)
	print('Actual class: '+str(y))
	# Predict from model
	prediction=model.predict(x)
	
	sorting = (-prediction).argsort()
	class1=sorting[0][0]
	class2=sorting[0][1]

	n_class1=str(class1)+'_'+str(class2)
	n_class2=str(class2)+'_'+str(class1)

	print('Predicted class: '+n_class1)

	if(n_class1==y or n_class2==y):
		correct+=1
	cnt+=1

print(correct)
print(cnt)
print('Accuracy: '+str((correct/cnt)*100))