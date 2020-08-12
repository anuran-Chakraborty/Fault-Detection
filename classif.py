import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from pandas.tools.plotting import table
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

filter_method=mutual_info_classif
num_of_features=300

def predict(x,Y,model_file):

	file = open(model_file,'rb')
	model = pickle.load(file)
	file.close()

	y_pred = model.predict(x)
	print(classification_report(Y, y_pred))
	print(accuracy_score(Y,y_pred)*100)
	return y_pred

def random_forest(x,y):
	classifier = RandomForestClassifier(n_estimators = 1500, n_jobs=-1, verbose=2)
	classifier.fit(x,y)

	f = open('rf.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

def mlp(x,y):
	classifier = MLPClassifier(hidden_layer_sizes = (225, 150, 100, 60), batch_size=20, learning_rate='adaptive', early_stopping=True, n_iter_no_change=50 ,max_iter=2000, verbose=True)
	classifier.fit(x,y)

	f = open('mlp.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

def xgb(x,y):
	classifier = XGBClassifier(n_estimators = 1500, max_depth=10, n_jobs=-1, verbose=2, learning_rate=0.01)
	classifier.fit(x,y)

	f = open('xgb.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

def svm(x,y):
	classifier = SVC(kernel = 'rbf', degree=3)
	classifier.fit(x,y)

	f = open('svm.pickle','wb')
	pickle.dump(classifier,f)
	f.close()


features=pd.read_csv('FeatureIndex.csv',header=None)

Xtrain=pd.read_csv('train_feature.csv')

Ytrain=pd.read_csv('train_target.csv',header=None)

print(Xtrain.shape)

Xtest=pd.read_csv('test_feature.csv')
Ytest=pd.read_csv('test_target.csv',header=None)

# Normalisation
minmax=MinMaxScaler()
Xtrain=minmax.fit_transform(Xtrain)
Xtest=minmax.transform(Xtest)

print('Performing feature_selection...')
selection=SelectKBest(score_func=filter_method,k=num_of_features)
selection.fit(Xtrain,Ytrain)
# Save the selector
pickle.dump(selection, open("feature_selection.pickle", "wb"))
selection=pickle.load(open("feature_selection.pickle", "rb"))

Xtrain=selection.transform(Xtrain)
Xtest=selection.transform(Xtest)

dfXtr=pd.DataFrame(Xtrain)
dfXte=pd.DataFrame(Xtest)

dfXtr.to_csv('train_selected.csv',index=False)
dfXte.to_csv('test_selected.csv',index=False)

# Train
xgb(Xtrain,Ytrain)
predict(Xtrain,Ytrain,'xgb.pickle')
predict(Xtest,Ytest,'xgb.pickle')


