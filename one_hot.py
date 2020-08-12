# Program to convert target files of time series to onehot vectors

import pandas as pd

def convert_to_one_hot(filename):

	df=pd.read_csv(filename+'.csv',header=None)
	df.columns=['class']
	# df['class'].astype('categorical')
	df=pd.get_dummies(df['class'],prefix='fault')

	print(df)

	df.to_csv(filename+'_one hot.csv',index=False)


convert_to_one_hot('train_target')
convert_to_one_hot('test_target')