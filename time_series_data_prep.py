import pandas as pd
import numpy as np


def prepare(filename):

	df=pd.read_csv(filename,header=None)

	lendf=len(df)

	target=df.iloc[:,-1]
	df=df.iloc[:,0:-1]

	df=df.values

	df=df.reshape(lendf,2500,5)

	df_final=pd.DataFrame(columns=['series_id','me_no','s01','s02','s03','s04','s05'])

	i=0

	mno=list(range(1,2501))

	for se in df:

		df_temp_X=pd.DataFrame(se,columns=['s01','s02','s03','s04','s05'])
		# add series id
		df_temp_X['series_id']=i
		# add measurement no
		df_temp_X['me_no']=mno

		# Append to existing dataframe
		df_final=pd.concat([df_final, df_temp_X],ignore_index=True)

		i=i+1


	print(df_final)

	# Save to file
	df_final.to_csv(filename+'_timeseries.csv',index=False)
	# Save target to file
	target.to_csv(filename+'_target.csv',index=False)

prepare('train.csv')
prepare('test.csv')