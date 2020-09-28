# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
#from sklearn.model_selection import 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from pandas import read_csv

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names=['id','dia','rad_mean','tex_mean','per_mean','are_mean','smo_mean','com_mean','con_mean','conp_mean','sym_mean','fra_mean','rad_se','tex_se','per_se','are_se','smo_se','com_se','con_se','conp_se','sym_se','fra_se','rad_worst','tex_worst','per_worst','are_worst','smo_worst','com_worst','con_worst','conp_worst','sym_worst','fra_worst']
dataset = read_csv(url, names=names)

#Plot data
#corr = dataset.corr()

#fig = plt.figure(figsize = (16,16))

#ax = fig.add_subplot(111)
#cax = ax.matshow(corr, vmin=-1, vmax=1)

#fig.colorbar(cax)

#ticks = np.arange(0,32,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)

#ax.set_xticklabels(names)
#ax.set_yticklabels(names)

#plt.show()


array = dataset.values

#Split array into input features and output values
X = array[:,2:]
y = array[:,1:2]

#Split out data set into train and validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))

#print('mean and std for each model')

# evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	kfold = StratifiedKFold(n_splits=10, random_state=7)
#	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#	results.append(cv_results)
#	names.append(name)
#	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#KNN
modelKNN = KNeighborsClassifier()
modelKNN.fit(X_train, Y_train)

predictionsKNN = modelKNN.predict(X_validation)

#Evaluate Prediction
print("------KNN------")
print("Accuracy Score ", accuracy_score(Y_validation, predictionsKNN))
print("Confusion Matrix\n", confusion_matrix(Y_validation, predictionsKNN))
print("Classification Report\n", classification_report(Y_validation, predictionsKNN))

#LR
modelLR = LogisticRegression()
modelLR.fit(X_train, Y_train)

predictionsLR = modelLR.predict(X_validation)

#Evaluate
print("------LR------")
print("Accuracy Score ", accuracy_score(Y_validation, predictionsLR))
print("Confusion Matrix\n",confusion_matrix(Y_validation, predictionsLR))
print("Classification Report\n",classification_report(Y_validation, predictionsLR))

#LDA
modelLDA = LinearDiscriminantAnalysis()
modelLDA.fit(X_train, Y_train)

predictionsLDA = modelLDA.predict(X_validation)

#Evaluate
print("------LDA------")
print("Accuracy Score ", accuracy_score(Y_validation, predictionsLDA))
print("Confusion Matrix\n",confusion_matrix(Y_validation, predictionsLDA))
print("Classification Report\n",classification_report(Y_validation, predictionsLDA))

#DTC
modelDTC = DecisionTreeClassifier()
modelDTC.fit(X_train, Y_train)

predictionsDTC = modelDTC.predict(X_validation)

#Evaluate
print("------DTC------")
print("Accuracy Score ", accuracy_score(Y_validation, predictionsDTC))
print("Confusion Matrix\n",confusion_matrix(Y_validation, predictionsDTC))
print("Classification Report\n",classification_report(Y_validation, predictionsDTC))

#NB
modelNB = GaussianNB()
modelNB.fit(X_train, Y_train)

predictionsNB = modelNB.predict(X_validation)

#Evaluate
print("------NB------")
print("Accuracy Score ", accuracy_score(Y_validation, predictionsNB))
print("Confusion Matrix\n",confusion_matrix(Y_validation, predictionsNB))
print("Classification Report\n",classification_report(Y_validation, predictionsNB))

#SVC
modelSVC = SVC()
modelSVC.fit(X_train, Y_train)

predictionsSVC = modelSVC.predict(X_validation)

#Evaluate
print("------SVC------")
print("Accuracy Score ", accuracy_score(Y_validation, predictionsSVC))
print("Confusion Matrix\n",confusion_matrix(Y_validation, predictionsSVC))
print("Classification Report\n",classification_report(Y_validation, predictionsSVC))

