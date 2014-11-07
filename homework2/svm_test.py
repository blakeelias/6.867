from numpy import *
import numpy as np
from plotBoundary import *
#import svm
from cvxopt import matrix
import svmcmpl
from svm import svmWeights

# parameters
#name = 'ls'
name = 'smallOverlap'
#print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
#X = [hstack((x, array(1,))) for x in X]
Y = train[:, 2:3].copy()

'''print('X')
print(X)
print('Y')
print(Y)'''
# Carry out training, primal and/or dual
### TODO ###
#predictSVM = svm.svm(X, Y, 1)
# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

print('C', 'geometric margin 1/||w||', 'error rate', 'supportVectors')
for C in [0.01, 0.1, 1, 10, 100]:
    #sol = svmcmpl.softmargin(matrix(X), matrix(Y), C, kernel='rbf')
    sol = svmcmpl.softmargin(matrix(X), matrix(Y), C, kernel='linear')
    predictSVM = sol['classifier']
    weights = sol['weights']
    supportVectors = sol['supportVectors']

    # plot training results
    #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
    #print(predictSVM(matrix(X)))

    #print '======Validation======'
    # load data from csv files
    #validate = loadtxt('data/data_'+name+'_validate.csv')
    validate = loadtxt('data/data_'+name+'_test.csv')
    X = validate[:, 0:2]
    Y = validate[:, 2:3]
    # plot validation results
    #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

    yPredicted = predictSVM(matrix(X))
    nError = 0
    for i in range(len(yPredicted)):
        if yPredicted[i] != Y[i]:
            nError += 1

    #print('weights')
    #print(weights)
    #print('geometric margin 1/||w||')

    print(C, 1.0/sum([x*x for x in weights]), nError*1.0/len(yPredicted), supportVectors)

    #print('nError', nError)
    #print('nTotal', len(yPredicted))



