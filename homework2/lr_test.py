from numpy import *
import numpy as np
from plotBoundary import *
import klr
from generate_kaggle_training import numPerClass, randomData

# parameters
name = 'ls2'
# load data from csv files
train = loadtxt('data/kaggle_train.csv', delimiter = ',')
X = train[:, 1:55].copy()
Y = train[:, 55:56].copy()

classes = range(1, 8)
trainX, trainY = randomData(X, Y, 1000)

probK = {}

for k in classes:
    #print('training for class %d' % k)
    Yk = np.array([[1.] if y == k else [-1.] for y in trainY])
    #print(Yk)

    predictLR = klr.lr(array(trainX), array(trainY))
    probK[k] = [predictLR(matrix(X[i]), soft = True) for i in range(len(X))]

yPredicted = [np.argmax(np.array([probK[k][i] for k in probK])) + 1 for i in range(len(X))]
print('yPredicted', yPredicted)

nError = 0
for i in range(len(yPredicted)):
    if yPredicted[i] != Y[i]:
        nError += 1
validationErrorRate = nError*1.0/len(yPredicted)

#print('weights')
#print(weights)
#print('geometric margin 1/||w||')

print(validationErrorRate) # trainingErrorRate
print('')