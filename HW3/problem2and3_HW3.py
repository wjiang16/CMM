from scipy.stats import ranksums
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
features = pd.read_csv('features_q2.csv',index_col = 0)
labels = pd.read_csv('class_q2.csv',index_col = 0)
feature_number = features.shape[0]
feature_name = features.index.values
pvalues = np.empty(feature_number)
x = features.values
y = labels.values.flatten()
top_n = 100
def select_feature(x,y):
    for i in range(0, feature_number):
        temp0 = x[i, y==0]
        temp1 = x[i, y==1]
        pvalues[i] = ranksums(temp0,temp1 )[1]
    top_n_index = sorted(range(len(feature_name)), key=lambda i: pvalues[i])[0:top_n]
    return top_n_index
top_n_index = select_feature(x,y)
def question_a():
    F = open('problem2_HW3.txt', 'w')

    F.write('top 10 features: \n' + str(feature_name[top_n_index[0:10]]))


# Question (b)

#Split the data into development set and validation set
# indices = StratifiedShuffleSplit(y,1, test_size = 0.5)
index_yes =np.where(y==1)[0]
index_no = np.where(y==0)[0]

train_indices = np.concatenate((index_yes[0: len(index_yes)/2],index_no[0: len(index_no)/2]))
test_indices = np.concatenate((index_yes[len(index_yes)/2:],index_no[len(index_no)/2:]))

X_train, X_test, y_train, y_test = x[:,train_indices], x[:,test_indices] , y[train_indices], y[test_indices]

classifier = {'feature':None, 'criterion':None, 'threshold':None, 'error_rate':1,'feature_index':None}
def find_classifier(top_n_index, X_train = X_train, y_train = y_train):
    for i in top_n_index:
        feature = X_train[i]

        for threshold in feature:
            # Classfier1: predicted to be 1 if x > threshold
            y_predict_larger = [0 if j <= threshold else 1 for j in feature]
            del j
            # Classfier2: predicted to be 1 if x < threshold
            y_predict_smaller = [0 if k >= threshold else 1 for k in feature]
            del k
            error_rate_larger = sum(abs(y_train - y_predict_larger)) / float(len(y_train))
            error_rate_smaller = sum(abs(y_train - y_predict_smaller)) / float(len(y_train))
            if error_rate_larger <= error_rate_smaller:
                if error_rate_larger < classifier['error_rate']:
                    classifier['feature'], classifier['criterion'] = feature_name[i], 'larger_than_threshold:1'
                    classifier['threshold'] , classifier['error_rate'] = threshold, error_rate_larger

                    classifier['feature_index'] = i
            else:
                if error_rate_smaller < classifier['error_rate']:
                    classifier['feature'], classifier['criterion'] = feature_name[i], 'less_than_threshold:1'
                    classifier['threshold'] , classifier['error_rate'] = threshold, error_rate_smaller
                    classifier['feature_index'] = i
    return classifier



classifier_b = find_classifier(top_n_index)
def eval_test_error(classifier= classifier_b, X_test = X_test, y_test = y_test):
    # Evaluate error rate on test set
    if classifier['criterion'] == 'larger_than_threshold:1':
        y_test_predict = [0 if x <= classifier['threshold'] else 1 for x in X_test[classifier['feature_index']]]
    else:
        y_test_predict = [0 if x >= classifier['threshold'] else 1 for x in X_test[classifier['feature_index']]]

    error_rate_test = sum(abs(y_test - y_test_predict)) / float(len(y_test))
    return error_rate_test

def question_b():

    print classifier_b
    error_rate_test = eval_test_error(classifier=classifier_b)
    print error_rate_test

del top_n_index
# question (C)
top_n_index_c = select_feature(X_train,y_train)
def question_c():

    classifier_c = find_classifier(top_n_index_c)
    error_test = eval_test_error(classifier=classifier_c)
    print classifier_c
    print error_test

#####################################################
# Problem 3
#####################################################

def plot_roc(ind, filename,title,data = X_train,y = y_train):
    """

    :param ind: int
    :type filename: string
    """
    thresholds = set(data[ind,:])
    TPR = np.empty(len(thresholds))
    FPR = np.empty(len(thresholds))

    for i, threshold in enumerate(thresholds) :
        y_predict = [0 if x <= threshold else 1 for x in data[ind,:]]
        temp1 = y - y_predict
        temp2 = y + y_predict

        TP = sum(temp2 == 2)
        TN = sum(temp2 == 0)
        FP = sum(temp1 == -1)
        FN = sum(temp1 == 1)

        TPR[i] = TP / float(TP + FN)
        FPR[i] = FP / float(FP + TN)

    plt.figure()
    plt.plot(FPR, TPR, '*')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve' + title)
    plt.savefig(filename)

top_n_index = select_feature(x,y)
# get index of most differentially expressed feature
pvalues = np.empty(feature_number)
for i in top_n_index:
    feature = X_train[i]
    temp0 = X_train[i, y_train == 1]
    temp1 = X_train[i, y_train == 0]
    pvalues[i] = ranksums(temp0,temp1)[1]
# get index of most differentially expressed feature
ind = top_n_index[np.where(pvalues == min(pvalues))[0]]
plot_roc(ind = ind, filename='roc_training2.png',title=' training(Question2)')
plot_roc(data= X_test, y = y_test,ind = top_n_index[0], filename='roc_testing2.png', title = ' testing(Question2)')

top_n_index = select_feature(X_train,y_train)
plot_roc(ind = top_n_index[0], filename='roc_training3.png',title=' training(Question3)')
plot_roc(data= X_test, y = y_test,ind = top_n_index[0], filename='roc_testing3.png',title=' testing(Question3)')

question_b()
question_c()