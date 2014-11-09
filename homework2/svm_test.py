from numpy import *
import numpy as np
from plotBoundary import *
#import svm
from cvxopt import matrix
import svmcmpl
from svm import svmWeights
from generate_kaggle_training import numPerClass, randomData

# parameters
#name = 'ls'
print('name', 'C', 'b', 'kernel', 'error on validation set') # 'error on training set',
#for name in ['smallOverlap', 'bigOverlap', 'ls', 'nonsep']:
for name in ['kaggle_train']:
    #print '======Training======'
    # load data from csv files
    #train = loadtxt('data/data_'+name+'_train.csv')
    train = loadtxt('data/'+name+'.csv', delimiter = ',')

    # use deep copy here to make cvxopt happy
    X = train[:, 1:55].copy()
    #X = [hstack((x, array(1,))) for x in X]
    Y = train[:, 55:56].copy()

    classes = range(1, 8)
    #trainX, trainY = numPerClass(X, Y, classes, 200)
    trainX, trainY = randomData(X, Y, 7000)

    #print('trainX', trainX, len(trainX))
    #print('trainY', trainY, len(trainY))

    '''print('X')
    print(X)
    print('Y')
    print(Y)'''
    # Carry out training, primal and/or dual
    ### TODO ###
    #predictSVM = svm.svm(X, Y, 1)
    # Define the predictSVM(x) function, which uses trained parameters
    ### TODO ###

    #print('C', 'b', 'error rate')
    C = 10
    b = 10
    kernel = 'rbf'
    #for C in [0.1, 1, 10, 100]:
    #    for b in np.arange(1, 1000, 100):
            # b = 1/(2*sigma)
            # sigma = 1/(2*b)

    probK = {}

    #C = 1
    for k in classes:
        #print('training for class %d' % k)
        Yk = np.array([[1.] if y == k else [-1.] for y in trainY])
        #print(Yk)

        sol = svmcmpl.softmargin(matrix(trainX), matrix(Yk), C, kernel=kernel) #, sigma=1.0/(2*b))
        predictSVM = sol['classifier']
        errors = sol['misclassified']
        totalError = len(errors[0] + errors[1])
        trainingErrorRate = totalError*1.0/len(trainX)

        #print(errors)
        #print('trainingErrorRate', trainingErrorRate)

        # plot training results
        #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
        #print(predictSVM(matrix(X)))

        #print '======Validation======'
        # load data from csv files
        #validate = loadtxt('data/data_'+name+'_validate.csv')
        #validate = loadtxt('data/data_'+name+'_test.csv')
        #X = validate[:, 0:2].copy()
        #Y = validate[:, 2:3].copy()
        # plot validation results
        #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

        probK[k] = predictSVM(matrix(X), soft = True)

    yPredicted = [np.argmax(np.array([probK[k][i] for k in probK])) + 1 for i in range(len(X))]

    nError = 0
    for i in range(len(yPredicted)):
        if yPredicted[i] != Y[i]:
            nError += 1
    validationErrorRate = nError*1.0/len(yPredicted)

    #print('weights')
    #print(weights)
    #print('geometric margin 1/||w||')

    print(name, C, b, kernel, validationErrorRate) # trainingErrorRate
    print('')

        #print('nError', nError)
        #print('nTotal', len(yPredicted))



