#Problem 1 (b)

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold
from scipy.spatial.distance import euclidean
features = pd.read_csv('features_q1.csv',header = None, sep =' ')
labels = pd.read_csv('class_q1.csv',header = None)
feature_NO = features.shape[1]
correlations = np.empty(feature_NO)
top_n = 100

for i in range(0, feature_NO):
    correlations[i] = np.corrcoef(labels.ix[:,0] , features.ix[:,i])[0,1]


top_n_index = sorted(range(len(correlations)), key=lambda i: correlations[i])[-top_n:]

top_n_feature = features.ix[:, top_n_index]

# k-fold cross validation using top_n features
k = 10

y = np.array(labels.ix[:,0])
x = top_n_feature.values
# y = labels.values
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(y, n_folds=k)
error_count = 0
for i, (train, test) in enumerate(cv):

    x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]


    for j in range(0, len(x_test)):
        distances = np.empty(len(x_test))
        for k in range(0, len(x_train)):
            distances[j] = euclidean(x_train[k,:], x_test[j,:])
        temp = sorted(range(len(distances)), key=lambda i: distances[i])[0]
        y_predict = y_train[temp]
        if y_predict != y_test[j]:
            error_count += 1

cv_error_rate = error_count/float(len(labels))
print cv_error_rate


#Problem 1 (c)
x = features.values
error_count = 0
correlations = np.empty(feature_NO)
for i, (train, test) in enumerate(cv):

    x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
    for l in range(0, feature_NO):
        correlations[l] = np.corrcoef(y_train , x_train[:,l])[0,1]


    top_n_index = sorted(range(len(correlations)), key=lambda i: correlations[i])[-top_n:]

    x_train = x_train[:, top_n_index]
    x_test = x_test[:, top_n_index]
    for j in range(0, len(x_test)):
        distances = np.empty(len(x_test))
        for k in range(0, len(x_train)):
            distances[j] = euclidean(x_train[k,:], x_test[j,:])
        temp = sorted(range(len(distances)), key=lambda i: distances[i])[0]
        y_predict = y_train[temp]
        if y_predict != y_test[j]:
            error_count += 1

cv_error_rate = error_count/float(len(labels))
print cv_error_rate



