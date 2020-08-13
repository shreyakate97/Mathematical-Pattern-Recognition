# -*- coding: utf-8 -*-
"""
Created on Fri May  8 03:20:03 2020

@author: shrey
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
from sklearn.metrics import confusion_matrix, accuracy_score


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
y_pred = np.empty([400,1])
# random classifier

for i in range(400):
    y_pred[i] = randint(1,4)
    
print('test accuracy for random classifier:',accuracy_score(test_label,y_pred),'\n')