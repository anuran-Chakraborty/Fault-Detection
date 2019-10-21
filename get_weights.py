import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
import math
from tensorflow.keras import backend as K
import csv

model=load_model('single_pd.h5')

weights =  model.get_weights()[0] 
print(weights)


#print(weights[0][0][0][0])

val = []

for j in range(15):

	temp = []
	for i in range(5):

		print(weights[i][0][0][j])
		temp.append(weights[i][0][0][j])

	print(" ")	
	val.append(temp)

with open('weights.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(val)	

	