import pandas as pd
import numpy as np
from scipy.stats import ranksums
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.learning_curve import validation_curve

df_train = pd.read_table('brcaTrainExpr.txt')
df_train.set_index('GENE', inplace=True)
df_train = df_train.transpose()

y_train = pd.read_table('brcaTrainPheno.txt')
y_train = y_train.iloc[:,1]
le = LabelEncoder()
y_train = le.fit_transform(y_train)

df_test = pd.read_table('brcaTestExpr.txt')
df_test.set_index('GENE', inplace=True)
df_test = df_test.transpose().values

y_test = pd.read_table('brcaTrainPheno.txt')
y_test = y_test.iloc[:,1]
le = LabelEncoder()
y_test = le.fit_transform(y_test)

feature_number = df_train.shape[1]
pvalues = np.empty(feature_number)
feature_name = df_train.columns
top_n = 100
def select_feature(x,y):
    for i in range(0, feature_number):
        temp0 = x[y==0,i]
        temp1 = x[y==1,i]
        pvalues[i] = ranksums(temp0,temp1 )[1]
    top_n_index = sorted(range(len(feature_name)), key=lambda i: pvalues[i])[0:top_n]
    return top_n_index
top_n_index = select_feature(df_train.values,y_train)

df_train = df_train.iloc[:, top_n_index[0:1000]]

pipe_lr = Pipeline([('scl',StandardScaler()),('clf', LogisticRegression(penalty='l2')) ])
# print pipe_lr.get_params().keys()
param_range =np.arange(0.01, 0.8, 0.01)

train_scores, test_scores = validation_curve(
    estimator= pipe_lr,
    X = df_train,
    y = y_train,
    param_name= 'clf__C',
    param_range= param_range,
    cv = 10
)
# df_train = StandardScaler().fit_transform(df_train)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color = 'blue',marker ='o', markersize = 5, label = 'training accuray')

plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = 0.15, color ='blue')

plt.plot(param_range, test_mean, color = 'red',marker ='s', markersize = 5, label = 'validation accuray')

plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha = 0.15, color ='red')

plt.grid()

plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accurary')
plt.ylim([0,1])
plt.show()
