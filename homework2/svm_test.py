from numpy import *
import numpy as np
from plotBoundary import *
#import svm
from cvxopt import matrix
import svmcmpl
from svm import svmWeights

# parameters
#name = 'ls'
print('name', 'C', 'b', 'error on validation set') # 'error on training set',
for name in ['smallOverlap', 'bigOverlap', 'ls', 'nonsep']:

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

    #print('C', 'b', 'error rate')
    for C in [0.01, 0.1, 1, 10, 100]:
        for b in np.arange(1, 10, 1):
            # b = 1/(2*sigma)
            # sigma = 1/(2*b)

            #C = 1
            sol = svmcmpl.softmargin(matrix(X), matrix(Y), C, kernel='rbf', sigma=1.0/(2*b))
            predictSVM = sol['classifier']
            errors = sol['misclassified']
            totalError = len(errors[0] + errors[1])
            trainingErrorRate = totalError*1.0/len(X)

            # plot training results
            #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')
            #print(predictSVM(matrix(X)))

            #print '======Validation======'
            # load data from csv files
            #validate = loadtxt('data/data_'+name+'_validate.csv')
            validate = loadtxt('data/data_'+name+'_test.csv')
            X = validate[:, 0:2].copy()
            Y = validate[:, 2:3].copy()
            # plot validation results
            #plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

            yPredicted = predictSVM(matrix(X))
            nError = 0
            for i in range(len(yPredicted)):
                if yPredicted[i] != Y[i]:
                    nError += 1
            validationErrorRate = nError*1.0/len(yPredicted)

            #print('weights')
            #print(weights)
            #print('geometric margin 1/||w||')

            print(name, C, b, validationErrorRate) # trainingErrorRate
            print('')

            #print('nError', nError)
            #print('nTotal', len(yPredicted))



