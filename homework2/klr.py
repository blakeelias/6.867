import numpy as np
from scipy import optimize
import pylab as pl
import random
from math import sqrt, exp, log

#from svm import numericalGradient

def numericalGradient(func, intervalWidth = 1e-3):
    def gradient(point):
        def numericalDerivative(func, x, intervalWidth):
            high = func(x + 0.5 * intervalWidth)
            low = func(x - 0.5 * intervalWidth)
            return (high - low)/intervalWidth

        answer = []
        for i in range(len(point)):
            def componentFunction(x):
                newPoint = np.array(point)
                newPoint[i] = x
                #print('point, newPoint: %s, %s' % (point, newPoint))
                val = func(newPoint)
                #print('value at newPoint: %f' % val)
                return val
            answer.append(numericalDerivative(componentFunction, \
                point[i], intervalWidth))
        return np.array(answer)
    return gradient

currentAlphas = []
onePreviousAlphas = []

def lr(X, Y, verbose=False, epsilon=0.0001, regularizeLambda = 1.0, kernel='rbf'):
    X = np.array(X)
    Y = np.array(Y)

    memo = {}

    def objective(alpha):
        global currentAlphas
        global onePreviousAlphas
        global i

        onePreviousAlphas = currentAlphas
        currentAlphas = alpha

        #if random.random() < 0.01:
        #    raise Exception('failing on purpose for test')
        #print(currentAlphas)
        # \sum_i \log(1 + \exp(-y^{(i)} (f(x^{(i)}) + w_0)  )
        # \sum_j K(x, x^{(j)}) \alpha_j
        w_0 = alpha[0]
        alpha = np.array(alpha[1:])
        def dot(x, X, alpha):
            return x.dot(X.T).dot(alpha)

        if kernel == 'linear':
            ans = sum([log(1 + exp(-Y[i]*(dot(X[i], X, alpha) + w_0))) for i in range(len(X))]) + \
                regularizeLambda * sqrt(alpha.dot(alpha) + epsilon)

        if kernel == 'rbf':
            def kernelFunc(x1, x2):
                return exp(-np.linalg.norm(x1 - x2)**2 / (2*1))
            
            def kernelVec(i, X):
                if i in memo:
                    return memo[i]
                memo[i] = [kernelFunc(X[i], x) for x in X]
                return memo[i]

            ans = sum([log(1 + exp(-Y[i]*(np.dot(kernelVec(i, X), alpha) + w_0))) for i in range(len(X))]) + \
                regularizeLambda * sqrt(alpha.dot(alpha) + epsilon)

        print(ans)
        return ans

    try:
        alpha = optimize.minimize(objective,
            np.array([random.random() for i in range(len(X) + 1)]),
            jac = numericalGradient(objective),
            )['x']
    except Exception as e:
        print(e)
        alpha = onePreviousAlphas

    print('retrieving from alpha')
    print(alpha)
    w_0 = alpha[0]
    alpha = alpha[1:]
    W = X.T.dot(alpha)

    def logit(x):
        return 1.0 / (1.0 + exp(-x))

    def classify(x):
        return 1 if logit(x.dot(W) + w_0) > 0.5 else -1

    return classify