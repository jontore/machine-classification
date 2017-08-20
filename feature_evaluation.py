import sys
import numpy as np
import tools.formatter
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectFdr, f_classif, f_regression

features_train, features_test, labels_train, labels_test, features, labels, data_dict = tools.formatter.preprocess('./datasets/fertility.txt')

scaler = preprocessing.StandardScaler().fit(features)
selector = SelectFdr(f_classif)
selector.fit(scaler.transform(features), np.array(labels))

features_names = data_dict.keys()
scores = -np.log10(selector.pvalues_)

plt.bar(range(len(features_names)), np.array(scores))
plt.xticks(range(len(features_names)), features_names, rotation='vertical')
plt.tight_layout()
plt.show()
