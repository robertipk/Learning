#!/usr/bin/python

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, model_selection as ms, neighbors, tree
import csv


chessdata = pd.read_csv("chessboard.csv")
# print(chessdata)
a = chessdata[['A','B']]
b = chessdata['label']

print(a)

data = csv.reader(open('chessboard.csv','rb'), delimiter=",", quotechar='|')
column1, column2 = [], []

for row in data:
    column1.append(row[0])
    column2.append(row[1])
column1.pop(0)
column2.pop(0)

print(column1)
print(column2)


#
# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
# print(neigh.predict([[1.1]]))
# print(neigh.predict_proba([[0.9]]))



plt.scatter(column1, column2, label="hello", color='k', s=15, marker=".")
plt.xlabel('column1')
plt.ylabel('column2')
plt.title("testing graph")
plt.legend()
plt.show()
