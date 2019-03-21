import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

import pickle



if __name__ == '__main__':
	# read train and test files
	df_train=pd.read_csv('train_timeseries.csv')
	df_test=pd.read_csv('test_timeseries.csv')

	df_train_X=extract_features(df_train,column_id = "series_id", chunksize=7, column_sort = "me_no", impute_function=impute, n_jobs=7)
	df_train_X.to_csv('train_feature.csv',index=False)

	del df_train
	del df_train_X

	df_test_X=extract_features(df_test,column_id = "series_id", chunksize=7, column_sort = "me_no", impute_function=impute, n_jobs=7)
	df_test_X.to_csv('test_feature.csv',index=False)