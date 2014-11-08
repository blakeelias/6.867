from numpy import *
from pprint import pprint

train = loadtxt('data/kaggle_train.csv', delimiter = ',')
X = train[:, 1:55].copy()
Y = train[:, 55:56].copy()


#classNames = sorted(list(set(Y.T[0])))

classes = {i: [] for i in range(1, 8)}
numPerClass = {i: 0 for i in range(1, 8)}

desiredClassSize = 3

for i in range(len(X)):
    y = Y[i][0]
    if numPerClass[y] < desiredClassSize:
        classes[y].append(X[i])
        numPerClass[y] += 1

print(numPerClass)

#classes = {c: [X[i] for i in range(len(X)) if Y[i] == c] for c in classNames}
pprint(classes)