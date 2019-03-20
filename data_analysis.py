import pandas as pd
import numpy as np
from sklearn import preprocessing  

def preprocess_train(filename):
	df=pd.read_csv(filename+'.csv')
	print(filename)
	print(df.head())
	# Counting number of nan in each column
	print((df.isna().sum()/df.shape[0])*100)
	# Counting number of classes
	df.fillna(df['s05'].mean(), inplace=True)
	df.to_csv(filename+'_clean.csv', index=False)

preprocess_train('train')
preprocess_train('test')