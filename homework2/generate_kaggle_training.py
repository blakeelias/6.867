import numpy as np
from pprint import pprint
import random

#train = loadtxt('data/kaggle_train.csv', delimiter = ',')
#X = train[:, 1:55].copy()
#Y = train[:, 55:56].copy()

def numPerClass(X, Y, classes, numPerClass):
    #classNames = sorted(list(set(Y.T[0])))

    # classes = {i: [] for i in range(1, numClasses + 1)}
    numPerClassSoFar = {i: 0 for i in classes}

    subsetX = []
    subsetY = []

    n = 0
    for i in range(len(X)):
        y = Y[i][0]
        if numPerClassSoFar[y] < numPerClass:
            #print('y', y)
            #classes[y].append(X[i])
            numPerClassSoFar[y] += 1
            subsetX.append(X[i])
            subsetY.append(Y[i])
            n += 1

    #print(n)
    return np.array(subsetX), np.array(subsetY)

def randomData(X, Y, n):
    indices = range(len(X))
    random.shuffle(indices)
    return np.array([X[i] for i in indices[0:n]]), np.array([Y[i] for i in indices[0:n]])