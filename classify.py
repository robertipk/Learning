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
A1, B1, A0, B0, kcc, label1, label0 = [], [], [], [], [], [], [], []

# print(data)
for row in data:
    kcc.append([row[0],row[1]])
    if row[2]=='0':
      A0.append(row[0])
      B0.append(row[0])
      label0.append(row[2])
    else:
      A1.append(row[0])
      B1.append(row[1])
      label1.append(row[2])
A1.pop(0)
B1.pop(0)
kcc1.pop(0)
label1.pop(0)
A0.pop(0)
B0.pop(0)
kcc0.pop(0)
label0.pop(0)
# print(label0)
# print(A0)
# print(B0)

# scatterplot
plt.scatter(A1, B1, label="Label: 1", color='blue', s=15)
plt.scatter(A0, B0, label="Label: 0", color='red', s=15)
plt.xlabel('A')
plt.ylabel('B')
plt.title("chessboard data")
plt.legend()
plt.show()

# K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(kcc, label)

from sklearn import svm

from sklearn.model_selection import train_test_split

# print(neigh.predict([[1.1,3]]))
# print(neigh.predict([[1.1,666663]]))

# print(neigh.predict_proba([[0.9]]))
