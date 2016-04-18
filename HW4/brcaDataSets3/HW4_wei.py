import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.learning_curve import validation_curve
from plot_decision_region import plot_decision_regions
from sklearn.svm import SVC
from scipy.stats import ranksums

df_train = pd.read_table('brcaTrainExpr.txt')
df_train.set_index('GENE', inplace=True)
df_train = df_train.transpose()

y_train = pd.read_table('brcaTrainPheno.txt')
y_train = y_train.iloc[:,1]
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

df_test = pd.read_table('brcaTestExpr.txt')
df_test.set_index('GENE', inplace=True)
df_test = df_test.transpose()

y_test = pd.read_table('brcaTestPheno.txt')
y_test = y_test.iloc[:,1]
le = LabelEncoder()

y_test = le.fit_transform(y_test)

feature_number = df_train.shape[1]
pvalues = np.empty(feature_number)
feature_name = df_train.columns
top_n = 1000
def select_feature(x,y):
    for i in range(0, feature_number):
        temp0 = x[y==0,i]
        temp1 = x[y==1,i]
        pvalues[i] = ranksums(temp0,temp1 )[1]
    top_n_index = sorted(range(len(feature_name)), key=lambda i: pvalues[i])[0:top_n]
    return top_n_index
top_n_index = select_feature(df_train.values,y_train)

df_train = df_train.iloc[:, top_n_index[0:20]]
df_test = df_test.iloc[:, top_n_index[0:20]]

def question_a():
    F = open('top_feature_HW3.txt', 'w')

    F.write('top 10 features: \n' + str(feature_name[top_n_index[0:10]]))
def tune_parameter():
    pipe_lr = Pipeline([('scl',StandardScaler()), ('clf', LogisticRegression(penalty='l2')) ])
    # print pipe_lr.get_params().keys()
    param_range =np.arange(0.0001,0.1,0.001)

    train_scores, test_scores = validation_curve(
        estimator= pipe_lr,
        X = df_train,
        y = y_train,
        param_name= 'clf__C',
        param_range= param_range,
        cv = 10
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, color = 'blue',marker ='o', markersize = 5, label = 'training accuray')

    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = 0.15, color ='blue')

    plt.plot(param_range, test_mean, color = 'red',marker ='s', markersize = 5, label = 'validation accuray')

    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha = 0.15, color ='red')

    plt.grid()

    # plt.xscale('log')
    plt.legend(loc = 'lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accurary')
    plt.ylim([0,1])
    plt.show()
tune_parameter()
st = StandardScaler().fit(df_train)
df_train = st.transform(df_train)
df_test = st.transform(df_test)

pca = PCA(n_components= 2)
lr = LogisticRegression(C=0.06)
# lr =SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0, degree=3,
# gamma=0.001, kernel='rbf', max_iter=-1, probability=True,
# random_state=None, shrinking=True, tol=0.001, verbose=False)
X_train_pca = pca.fit_transform(df_train)
X_test_pca = pca.transform(df_test)

lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc ='lower left')
plt.savefig('PCA_prediction.jpg')
plt.show()

fig = plt.figure(figsize=(7,5))
# mean_tpr = 0
# mean_tpr = np.linspace(0,1,100)
all_tpr = []
### plot ROC curve for test data
probas = lr.fit(df_train, y_train).predict_proba(df_test)
fpr, tpr, thresholds = roc_curve(y_test, probas[:,1], pos_label=1)

roc_auc = auc(fpr, tpr)
plt.plot(fpr,tpr,lw =1, label = 'ROC curve for test data (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1], linestyle = '--', label = 'random guessing')

plt.xlim([-0.05,1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')

plt.title('ROC curve')
plt.legend(loc ='lower right')
plt.savefig('ROC_curve_test_data.jpg')
plt.show()

