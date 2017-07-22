import sys
import numpy as np
import tools.formatter
import matplotlib.pyplot

from sklearn import preprocessing
from sklearn.feature_selection import SelectFdr, f_classif, f_regression

features_train, features_test, labels_train, labels_test, features, labels, data_dict = tools.formatter.preprocess('./datasets/fertility.txt')

scaler = preprocessing.StandardScaler().fit(features)
selector = SelectFdr(f_classif)
selector.fit(scaler.transform(features), np.array(labels))

features_names = data_dict.keys()
scores = -np.log10(selector.pvalues_)

matplotlib.pyplot.bar(range(len(features_names)), np.array(scores))
matplotlib.pyplot.xticks(range(len(features_names)), features_names, rotation='vertical')
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()
