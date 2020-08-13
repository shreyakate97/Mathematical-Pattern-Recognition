# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:38:00 2020

@author: shrey
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

d_train = pd.read_csv('D_Train1.csv')
d_test = pd.read_csv('D_Test1.csv')

# training data
X_data = d_train.drop('Location',axis=1)
X_label = d_train['Location']

X_data = X_data.to_numpy()
X_label = X_label.to_numpy()

# test data
test_data = d_test.drop('Location',axis=1)
test_label = d_test['Location']

test_data = test_data.to_numpy()
test_label = test_label.to_numpy()

X_datap = np.empty([1600,7])
X_datan = np.empty([1600,7])
test_datap = np.empty([400,7])
test_datan = np.empty([400,7])

# SVM classifier
print('----------SVM CLASSIFIER----------')
print('NO PREPROCESSING\n')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
parameters = {'C':[0.1, 1, 5, 10,100], 'gamma':[0.001, 0.05, 0.1, 0.5,0.75, 1]}
svc = SVC()
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
clf = GridSearchCV(svc, parameters,cv=cv)
clf.fit(X_data, X_label)
print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))

# training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, X_label, test_size=0.2)
c = SVC(kernel='rbf', C=clf.best_params_['C'], gamma = clf.best_params_['gamma'])
c.fit(X_train, y_train)
y_pred = c.predict(X_val)
y_pred_train = c.predict(X_train)
print('train accuracy',accuracy_score(y_train,y_pred_train))
print('validation accuracy',accuracy_score(y_val,y_pred))

# test set
y_pred_test = c.predict(test_data)
print('test accuracy',accuracy_score(test_label,y_pred_test),'\n')

# Preprocessing

# Standardization
for i in range(0,7):
    X_datap[:,i] = ( X_data[:,i] - np.mean(X_data[:,i]) )/np.std(X_data[:,i])
    test_datap[:,i] = ( test_data[:,i] - np.mean(test_data[:,i]) )/np.std(test_data[:,i])
    
print('STANDARDIZATION\n')

parameters = {'C':[0.1, 1, 5, 10,100], 'gamma':[0.001, 0.05, 0.1, 0.5,0.75, 1]}
svc = SVC()
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
clf = GridSearchCV(svc, parameters,cv=cv)
clf.fit(X_datap, X_label)

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
# training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_datap, X_label, test_size=0.2)
c = SVC(kernel='rbf', C=clf.best_params_['C'], gamma = clf.best_params_['gamma'])
c.fit(X_train, y_train)
y_pred = c.predict(X_val)
y_pred_train = c.predict(X_train)
print('train accuracy',accuracy_score(y_train,y_pred_train))
print('validation accuracy',accuracy_score(y_val,y_pred))

# test set
y_pred_test = c.predict(test_datap)
print('test accuracy',accuracy_score(test_label,y_pred_test),'\n')
# print(confusion_matrix(test_label,y_pred_test))

#Normalization
for i in range(0,7):
    X_datan[:,i] = (X_data[:,i] - np.amin(X_data[:,i]) )/(np.amax(X_data[:,i])- np.amin(X_data[:,i]))
    test_datan[:,i] = ( test_data[:,i] - np.amin(test_data[:,i]) )/(np.amax(test_data[:,i])- np.amin(test_data[:,i]))

print('NORMALIZATION\n')

parameters = {'C':[0.1, 1, 5, 10,100], 'gamma':[0.001, 0.05, 0.1, 0.5,0.75, 1]}
svc = SVC()
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
clf = GridSearchCV(svc, parameters,cv=cv)
clf.fit(X_datan, X_label)

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
# training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_datan, X_label, test_size=0.2)
c = SVC(kernel='rbf', C=clf.best_params_['C'], gamma = clf.best_params_['gamma'])
c.fit(X_train, y_train)
y_pred = c.predict(X_val)
y_pred_train = c.predict(X_train)
print('train accuracy',accuracy_score(y_train,y_pred_train))
print('validation accuracy',accuracy_score(y_val,y_pred))

# test set
y_pred_test = c.predict(test_datan)
print('test accuracy',accuracy_score(test_label,y_pred_test),'\n')
# print(confusion_matrix(test_label,y_pred_test))

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_datap, X_label)
print(clf.feature_importances_) 

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_datap)
test_new = model.transform(test_datap)

parameters = {'C':[0.1, 1, 5, 10,100], 'gamma':[0.001, 0.05, 0.1, 0.5,0.75, 1]}
svc = SVC()
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
clf = GridSearchCV(svc, parameters,cv=cv)
clf.fit(X_new, X_label)

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
# training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_new, X_label, test_size=0.2)
c = SVC(kernel='rbf', C=clf.best_params_['C'], gamma = clf.best_params_['gamma'])
c.fit(X_train, y_train)
y_pred = c.predict(X_val)
y_pred_train = c.predict(X_train)
print('train accuracy',accuracy_score(y_train,y_pred_train))
print('validation accuracy',accuracy_score(y_val,y_pred))

# test set
y_pred_test = c.predict(test_new)
print('test accuracy',accuracy_score(test_label,y_pred_test),'\n')
# print(confusion_matrix(test_label,y_pred_test))


import seaborn as sns
cm = confusion_matrix(test_label,y_pred_test)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(cm, annot=True, vmin=0.0, vmax=100.0, fmt='.2f', cmap=cmap)
plt.xticks([0.5,1.5,2.5,3.5], [ '1', '2', '3','4'],va='center')
plt.yticks([0.5,1.5,2.5,3.5], [ '1', '2', '3','4'],va='center')
