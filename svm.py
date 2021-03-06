import sys
import numpy as np
import tools.formatter
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectFdr, f_classif, f_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

def calculateAvg(list):
    return reduce(lambda x, y: x + y, list) / len(list)

features_train, features_test, labels_train, labels_test, features, labels, data_dict = tools.formatter.preprocess('./datasets/fertility.txt')

skf = StratifiedKFold(n_splits=12)
C = 2
gamma = 0.2

tuned_parameters = {
    'accuracy': [
        { 'kernel': ['rbf'],
          'gamma': [0.775],
          'C': [1]
        }
    ],
    'roc_auc': [
        { 'kernel': ['rbf'],
          'gamma': [0.15635625],
          'C': [123]
        }
    ]
}


scores = ['accuracy', 'roc_auc']

best_params = {
    'accuracy': [],
    'roc_auc': []
}

for score in scores:
    aucs = []
    accs = []
    for train_idx, test_idx in skf.split(features, labels):
        features_train = map(lambda idx: features[idx], train_idx)
        labels_train = map(lambda idx: labels[idx], train_idx)
        features_test = map(lambda idx: features[idx], test_idx)
        labels_test = map(lambda idx: labels[idx], test_idx)

        clf = GridSearchCV(svm.SVC(probability=True), tuned_parameters[score], cv=5,
                           scoring=score)
        clf.fit(features_train, labels_train)

        best_params[score].append(clf.best_params_)

        pred = clf.predict(features_test)
        prob = clf.predict_proba(features_test)[:,1]
        acc = accuracy_score(labels_test, pred)
        auc = roc_auc_score(np.array(labels_test), np.array(prob))
        aucs.append(auc)
        accs.append(acc)

    print '---------------------------------------------------------------------------------', score
    print 'Average: Accuracy', calculateAvg(accs), 'AUC', calculateAvg(aucs)

print '---------------------------------------------------------------------------------'
for score in scores:
    Cs= map(lambda v: v['C'], best_params[score])
    gammas = map(lambda v: v['gamma'], best_params[score])
    print 'Average best params: scoring:', score, ' C:', calculateAvg(Cs), ' gammas ', calculateAvg(gammas)
