#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, model_selection as ms, neighbors, tree
import csv


chessdata = pd.read_csv("chessboard.csv")
# print(chessdata)
a = chessdata[['A','B']]
b = chessdata['label']

# print(a)

data = csv.reader(open('chessboard.csv','rb'), delimiter=",", quotechar='|')
# for scatter plot
A1, B1, A0, B0, all_samples, label1, label0 = [], [], [], [], [], [], []
# for SVM
all_labels = []

# print(data)
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
all_samples.pop(0)
all_labels.pop(0)

A1.pop(0)
B1.pop(0)
label1.pop(0)
A0.pop(0)
B0.pop(0)
label0.pop(0)
# print(label0)
# print(A0)
# print(B0)

# scatterplot
# plt.scatter(A1, B1, label="Label: 1", color='blue', s=15)
# plt.scatter(A0, B0, label="Label: 0", color='red', s=15)
# plt.xlabel('A')
# plt.ylabel('B')
# plt.title("chessboard data")
# plt.legend()
# plt.show()

# K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
neigh1 = KNeighborsClassifier(n_neighbors=1)
neigh5 = KNeighborsClassifier(n_neighbors=5)
neigh10 = KNeighborsClassifier(n_neighbors=10)
neigh15 = KNeighborsClassifier(n_neighbors=15)

# neigh.fit(all_samples, label)

from sklearn import svm
linear_svc = svm.SVC(kernel='linear')
linear_svc.kernel
linear_svc.fit(all_samples, all_labels)

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.kernel
rbf_svc.fit(all_samples, all_labels)

sigmoid_svc = svm.SVC(kernel='sigmoid')
sigmoid_svc.kernel
sigmoid_svc.fit(all_samples, all_labels)

polynomial_svc = svm.SVC(kernel='poly')
polynomial_svc.kernel
polynomial_svc.fit(all_samples, all_labels)

#
# linear_svc = svm.SVC(kernel='linear')
# linear_svc = svm.SVC(kernel='linear')
#
# clf = svm.SVC()
# clf.fit(all_samples, all_labels)
from sklearn.tree import DecisionTreeClassifier
clf1 = tree.DecisionTreeClassifier(max_depth=1)
clf1.fit(all_samples, all_labels)

# print("now predictions" + clf.predict([[2,5]])[0])

from sklearn.model_selection import train_test_split


# print(neigh.predict([[1.1,3]]))
# print(neigh.predict([[1.1,666663]]))

# print(neigh.predict_proba([[0.9]]))
