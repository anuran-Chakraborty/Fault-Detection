import numpy as np
import os
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pickle

from imblearn.over_sampling import SMOTE
from pandas.tools.plotting import table
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import preprocessing

def predict(x,model_file):

	file = open(model_file,'rb')
	model = pickle.load(file)
	file.close()

	# file=open('normaliser.pickle','rb')
	# min_max_scaler=pickle.load(file)
	# x=min_max_scaler.transform(x)

	y_pred = model.predict(x)
	# print(classification_report(Y, y_pred))
	# print(accuracy_score(Y,y_pred)*100)
	return y_pred

df=pd.read_csv('cleaned_insurance_predict2.csv')
# Select those features which have been slelected by feature selection
# Load
# print('Loading from file feature selection...')
# selection=pickle.load(open("feature_selection.pickle","rb"))
# df=selection.transform(df)

predictions=(predict(df,'logistic.pickle'))

preds=pd.DataFrame({'Have.life.insurance':predictions})
preds['Have.life.insurance'] = preds['Have.life.insurance'].replace({1: 'Yes', 0: 'No'})
print(preds)
preds.to_csv('insurance_predictions.csv', index=None)