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

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def predict(x,Y,model_file):

	file = open(model_file,'rb')
	model = pickle.load(file)
	file.close()

	y_pred = model.predict(x)
	print(classification_report(Y, y_pred))
	print(accuracy_score(Y,y_pred)*100)
	return y_pred

def naive_bayes(x,y):
	classifier = MultinomialNB()
	classifier.fit(x,y)
	
	# Save model
	f = open('naivebayes.pickle','wb')
	pickle.dump(classifier,f)
	f.close()


def svm(x,y):
	classifier = SVC(kernel='rbf')
	classifier.fit(x,y)

	f = open('svm.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

def knn(x,y):
	classifier = KNeighborsClassifier(n_neighbors=100)
	classifier.fit(x,y)

	f = open('knn.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

def logistic(x,y):
	classifier = LogisticRegression()
	classifier.fit(x,y)

	f = open('logistic.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

def random_forest(x,y):
	classifier = RandomForestClassifier(n_estimators = 42)
	classifier.fit(x,y)

	f = open('randomforest.pickle','wb')
	pickle.dump(classifier,f)
	f.close()

# Parameters
filter_method=mutual_info_classif
num_of_features=12

print('Loading dataset...')
# loading the dataset
df_tr=pd.read_csv('train_clean.csv')
print('Splitting....')
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(df.drop('Have.life.insurance', axis=1), df['Have.life.insurance'], train_size=0.6, test_size=0.4)
Ytrain=df_tr['class']
Xtrain=df_tr.drop('class', axis=1)

df_te=pd.read_csv('test_clean.csv')
Ytest=df_te['class']
Xtest=df_te.drop('class', axis=1)

# Normalise
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(Xtrain)
pickle.dump(min_max_scaler, open("normaliser.pickle", "wb"))
Xtrain=min_max_scaler.transform(Xtrain)
Xtest=min_max_scaler.transform(Xtest)

# # Perform feature selection
# print('Performing feature selelction')
# selection=SelectKBest(score_func=filter_method,k=num_of_features)
# selection.fit(Xtrain,Ytrain)
# # Save the selector
# pickle.dump(selection, open("feature_selection.pickle", "wb"))

# # Transform train and test
# Xtrain=selection.transform(Xtrain)
# # Xtest=selection.transform(Xtest)

# print(Xtrain)

# Now fit classfier
# print('Training naive bayes')
# naive_bayes(Xtrain,Ytrain)
# predict(Xtrain,Ytrain,'naivebayes.pickle')
# predict(Xtest,Ytest,'naivebayes.pickle')

print('Training random_forest')
random_forest(Xtrain, Ytrain)
predict(Xtrain,Ytrain,'randomforest.pickle')
predict(Xtest,Ytest,'randomforest.pickle')

print('Training logistic')
logistic(Xtrain,Ytrain)
predict(Xtrain,Ytrain,'logistic.pickle')
predict(Xtest,Ytest,'logistic.pickle')

print('Training knn')
knn(Xtrain,Ytrain)
predict(Xtrain,Ytrain,'knn.pickle')
predict(Xtest,Ytest,'knn.pickle')

print('Training svm')
svm(Xtrain,Ytrain)
predict(Xtrain,Ytrain,'svm.pickle')
predict(Xtest,Ytest,'svm.pickle')