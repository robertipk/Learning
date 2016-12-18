#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm, model_selection as ms, neighbors, tree
import csv

chessdata = pd.read_csv("chessboard.csv")
a = chessdata[['A','B']]
b = chessdata['label']

data = csv.reader(open('chessboard.csv','rb'), delimiter=",", quotechar='|')
A1, B1, A0, B0, all_samples, label1, label0, all_labels = [], [], [], [], [], [], [], []
# skip headers
next(data, None)
for row in data:
    all_samples.append([row[0],row[1]])
    all_labels.append(row[2])
    if row[2]=='0':
      A0.append(row[0])
      B0.append(row[1])
      label0.append(row[2])
    else:
      A1.append(row[0])
      B1.append(row[1])
      label1.append(row[2])

# scatterplot
# plt.scatter(A1, B1, label="Label: 1", color='blue', s=15)
# plt.scatter(A0, B0, label="Label: 0", color='red', s=15)
# plt.xlabel('A')
# plt.ylabel('B')
# plt.title("chessboard data")
# plt.legend()
# plt.show()

# 10-fold cross-validation with mean and standard deviation
def kCrossValidation(model, classifier, samples, labels):
  print("Classifier: " + model)
  scores = cross_val_score(classifier, all_samples, all_labels, cv=10)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# K Neighbors Classifier
knn1 = neighbors.KNeighborsClassifier(n_neighbors=1)
kCrossValidation("K Nearest Neighbors: 1", knn1, all_samples, all_labels)

knn5 = KNeighborsClassifier(n_neighbors=5)
kCrossValidation("K Nearest Neighbors: 5", knn5, all_samples, all_labels)

knn10 = KNeighborsClassifier(n_neighbors=10)
kCrossValidation("K Nearest Neighbors: 10", knn10, all_samples, all_labels)

knn15 = KNeighborsClassifier(n_neighbors=15)
kCrossValidation("K Nearest Neighbors: 15", knn15, all_samples, all_labels)

# SVM
linear_svc = svm.SVC(kernel='linear')
kCrossValidation("Linear SVC", linear_svc, all_samples, all_labels)

rbf_svc = svm.SVC(kernel='rbf')
kCrossValidation("RBF SVC", rbf_svc, all_samples, all_labels)

sigmoid_svc = svm.SVC(kernel='sigmoid')
kCrossValidation("Sigmoid SVC", sigmoid_svc, all_samples, all_labels)

polynomial_svc = svm.SVC(kernel='poly')
kCrossValidation("Polynomial SVC", polynomial_svc, all_samples, all_labels)

# Decision tree
dt1 = tree.DecisionTreeClassifier(max_depth=1)
kCrossValidation("Decision Tree - max depth 1", dt1, all_samples, all_labels)

dt2 = tree.DecisionTreeClassifier(max_depth=2)
kCrossValidation("Decision Tree - max depth 2", dt2, all_samples, all_labels)

dt4 = tree.DecisionTreeClassifier(max_depth=4)
kCrossValidation("Decision Tree - max depth 4", dt4, all_samples, all_labels)

dt8 = tree.DecisionTreeClassifier(max_depth=8)
kCrossValidation("Decision Tree - max depth 8", dt8, all_samples, all_labels)

dt16 = tree.DecisionTreeClassifier(max_depth=16)
kCrossValidation("Decision Tree - max depth 16", dt16, all_samples, all_labels)
