import sys
import numpy as np
import tools.formatter
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectFdr, f_classif, f_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

SCALER = [None, preprocessing.StandardScaler()]
REDUCER__N_COMPONENTS = [2, 4, 6, 8, 10]

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

h = .02  # step size in the mesh

X = []
alcohol_consumption = data_dict['number_of_hours_sitting']
age = data_dict['age']
for idx, val in enumerate(alcohol_consumption):
    X.append(np.array([float(alcohol_consumption[idx]), float(age[idx])]))

X = np.array(X)
y = labels
X_train, X_test, y_train, y_test  = train_test_split(X, np.array(y), test_size=0.2, random_state=42)


C = 4  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
navie_bays = GaussianNB().fit(X_train, y_train)
decision_tree = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1).fit(X_train, y_train)

# create a mesh to plot in
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


for i, clf in enumerate((rbf_svc, poly_svc, lin_svc, navie_bays, decision_tree)):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title(clf)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    print 'Score', clf.score(X_test, y_test)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.show()
