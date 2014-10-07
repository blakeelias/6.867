import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
from matplotlib import pyplot

# Problem 1
def gradientDescent(func, gradient=None, guess=0, stopChange=0.001, stepRate=0.01, momentumWeight=0.1, verbose=False):
    print('performing gradient descent')
    #print('examining guess parameter, func takes inputs with %s component(s)' % len(guess))
    lastChange = float('inf')
    prevGuess = guess
    funcVal = func(guess)

    nFuncCalls = 1
    nGradCalls = 0

    if gradient == None:
        gradient = numericalGradient(func)
    
    if verbose:
        print('guess, gradient, funcVal, lastChange')

    while lastChange > stopChange:
        g = gradient(guess)
        nGradCalls += 1
        if verbose: print('%s, %s, %s, %s' % (guess, g, funcVal, lastChange))

        try:
            newGuess = guess - stepRate * g \
                + momentumWeight*(guess - prevGuess)
        except Exception as e:
            print(e)
            print('guess, stepRate, g, momentumWeight, prevGuess:', guess, stepRate, g, momentumWeight, prevGuess)
            break
        prevGuess = guess
        guess = newGuess

        lastFuncVal = funcVal
        funcVal = func(guess)
        nFuncCalls += 1
        lastChange = abs(funcVal - lastFuncVal)
    
    print('Function evaluations: %d' % nFuncCalls)
    print('Gradient evaluations: %d' % nGradCalls)
    return guess

# Problem 1
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
        return answer
    return gradient

# Problem 2.1
def designMatrix(X, order, includeConstantTerm=True):
    return np.array([[x[0]**i for i in range(0 if includeConstantTerm else 1, order+1)] for x in X])
    # okay because 0**0 == 1

def linearDesignMatrix(X, includeConstantTerm=True):
    if includeConstantTerm:
        return np.hstack((np.array([[1]] * len(X)), X))
    return X

# Problem 2.1
def regressionFit(X, Y, phi, params):
    #print(phi)
    #print(phi.T)
    a = pl.dot(phi.T, phi)
    b = pl.dot(np.linalg.inv(a), phi.T)
    return pl.dot(b, Y)

# Problem 2.1
# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order, fitMethod=regressionFit, params={}, verbose=False, plot=True, validationData=None):
    if verbose:
        print('X', X)
        print('Y', Y)
    if plot:
        pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
        if validationData != None:
            X_validate, Y_validate = validationData
            pl.plot(X_validate.T.tolist()[0],Y_validate.T.tolist()[0], 'bo')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    params['order'] = order
    w = fitMethod(X, Y, phi, params)
    
    if verbose:
        print 'w', w
    try:
        Y_estimated = pl.dot(w.T, phi.T)
    except:
        print('matrices not aligned: ', w.T.size, phi.T.size)
    if verbose:
        print('Y_estimated', Y_estimated)
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = applyWeights(pts, order, w)
    if plot:
        pl.plot(pts, Yp.tolist()[0])
    error = sumOfSquaresErrorGenerator(phi, Y)(w)
    return (error, w)
    #print('error: %f' % error)
    #print('analytical error gradient: %s' % sumOfSquaresErrorGradientGenerator(phi, Y)(w))
    #print('numerical error gradient: %s' % \
    #    numericalGradient(sumOfSquaresErrorGenerator(phi, Y), w, 1e-5))

def applyWeights(X, order, weights):
    phi = designMatrix(X, order)
    return pl.dot(weights.T, phi.T)

# Problem 2.2
def sumOfSquaresErrorGenerator(phi, Y):
    def sumOfSquaresError(w):
        '''Given data points X, a vector Y of values, a feature (design) matrix phi,
        and a weight vector w, compute the sum of squares error (SSE)'''
        Yp = pl.dot(w.T, phi.T)
        #print(Y.T)
        #print(Yp)
        #print(Yp - Y)
        return np.linalg.norm(Yp - Y.T)**2
    return sumOfSquaresError

# Problem 2.2
def sumOfSquaresErrorGradientGenerator(phi, Y):
    def sumOfSquaresErrorGradient(w):
        return 2 * pl.dot(phi.T, pl.dot(phi, w) - Y)
    return sumOfSquaresErrorGradient

# Problem 2.3
def gradientDescentFit(X, Y, phi, params):
    func = sumOfSquaresErrorGenerator(phi, Y)
    #grad = sumOfSquaresErrorGradientGenerator(phi, Y)
    guess = np.array([0]*len(X[0])).T
    print(guess)
    print(func(guess))
    #print(grad(guess))
    return fmin_bfgs(func, guess)

    return gradientDescent(sumOfSquaresErrorGenerator(phi, Y), \
        sumOfSquaresErrorGradientGenerator(phi, Y), \
        np.array([0]*len(X[0])).T, \
        verbose=True)

# Problem 3.1
def ridgeFit(X, Y, phi, params, verbose=False):
    #phi = designMatrix(X, params['order'], includeConstantTerm=False)
    l = params['lambda']
    phi_avg = sum(phi)*1.0 / len(phi)
    Z = phi - phi_avg
    Y_avg = sum(Y)*1.0 / len(Y)
    Yc = Y - Y_avg
    if verbose:
        print('phi', phi)
        print('phi_avg', phi_avg)
        print('Z', Z)
        print('Yc', Yc)

    a = pl.dot(Z.T, Z) + l * pl.eye(len(Z.T))
    b = np.linalg.inv(a).dot(Z.T)
    W = b.dot(Yc)
    W_0 = np.array([Y_avg - W.T.dot(phi_avg)])
    if verbose:
        print('W_0', W_0)
        print('W', W)
    return np.hstack((W_0, W.T)).T

# Problem 3.2
def modelSelection(trainingData, validationData, verbose=False):
    X, Y = trainingData
    X_validate, Y_validate = validationData
    orderErrors = []
    for M in range(9):
        regularizationWeights = np.hstack((np.arange(0, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000, 100), np.arange(1000, 10000, 1000)))
        lambdaErrors = []
        for l in regularizationWeights:
            error, weights = regressionPlot(X, Y, M, ridgeFit, {'lambda': l}, plot=False)
            validateError = sumOfSquaresErrorGenerator(designMatrix(X_validate, M), Y_validate)(weights)
            lambdaErrors.append((validateError, l, weights))
        (error, l, weights) = min(lambdaErrors)
        if verbose:
            print(M, l, error, weights)
        orderErrors.append((error, M, l, weights))
    model = min(orderErrors)
    if verbose:
        print('optimal: (error, M, l, weights) = ', model)
    regressionPlot(X, Y, M, fitMethod=regressionFit, params={'lambda': l}, verbose=verbose, plot=True, validationData=(X_validate, Y_validate))

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

# Problem 3.3
def getDataCSV(xName, yName):
    return (pl.loadtxt(xName, delimiter=','), np.array([pl.loadtxt(yName, delimiter=',')]).T)


def blogTrainData():
    return getDataCSV('dataset/x_train.csv', 'dataset/y_train.csv')

def blogValidateData():
    return getDataCSV('dataset/x_val.csv', 'dataset/y_val.csv')

def blogTestData():
    return getDataCSV('dataset/x_test.csv', 'dataset/y_test.csv')

def blogRegression(X, Y, fitMethod=ridgeFit, params={}, verbose=False, validationData=None):
    if verbose:
        print('X', X)
        print('Y', Y)

    phi = linearDesignMatrix(X, includeConstantTerm=False)
    # compute the weight vector
    w = fitMethod(X, Y, phi, params)
    
    if verbose:
        print 'w', w

    phi = linearDesignMatrix(X, includeConstantTerm=True)
    try:
        Y_estimated = pl.dot(w.T, phi.T)
    except:
        print('matrices not aligned: ', w.T.shape, phi.T.shape)
    #Y_estimated = pl.dot(w.T, phi.T)
    if verbose:
        print('Y_estimated', Y_estimated)
    
    error = sumOfSquaresErrorGenerator(phi, Y)(w)
    return (error, w)

# Problem 3.3
def blogModelSelection(trainingData, validationData, verbose=False):
    X, Y = trainingData
    X_validate, Y_validate = validationData
    #regularizationWeights = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7] #np.hstack((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000, 100), np.arange(1000, 10000, 1000)))
    regularizationWeights = np.arange(31600, 31800, 10)
    lambdaErrors = []
    for l in regularizationWeights:
        error, weights = blogRegression(X, Y, ridgeFit, {'lambda': l})
        validateError = sumOfSquaresErrorGenerator(linearDesignMatrix(X_validate), Y_validate)(weights)
        lambdaErrors.append((validateError, l)) #, weights))
        if verbose:
            print('lambda = %f, error = %f' % (l, validateError))
    (error, l) = min(lambdaErrors) # (error, l, weights) = 
    return (error, l)

if __name__ == '__main__':
    '''def bowl(x):
        a, b = x
        return a*a + b*b

    def bowlGradient(x):
        a, b = x
        return np.array((2*a, 2*b))

    def makeGaussian(mean, variance):
        def gaussian(x):
            return 1 / np.sqrt(2 * np.pi * variance) * \
                np.exp(- (x - mean)**2 / (2 * variance))
        def gradient(x):
            return gaussian(x) * (mean - x) / variance
        return gaussian, gradient

    def negate(func):
        return lambda x: -func(x)

    print('quadratic bowl:')
    z = gradientDescent(bowl, bowlGradient, np.array((3, 5)), 
        stopChange=0.00000001,
        stepRate=0.01,
        momentumWeight=0.1)
    print(z)
    print('scipy.optimize.fmin_bfgs: ')
    print(fmin_bfgs(bowl, np.array((3, 5)), bowlGradient))
    print('-'*60)
    print('inverted gaussian:')

    gaussian, gaussianGradient = makeGaussian(10, 4)
    invertedGaussian = negate(gaussian)
    invertedGaussianGradient = negate(gaussianGradient)
    print(gradientDescent(invertedGaussian, invertedGaussianGradient, np.array((5)),
        stopChange = 1e-7,
        stepRate = 0.05,
        momentumWeight = 0.1))
    print(fmin_bfgs(invertedGaussian, np.array((5)), invertedGaussianGradient))


    print numericalGradient(bowl, np.array((3.0, 5.0)), intervalWidth = 1e-10)
    print bowlGradient(np.array((3.0, 5.0)))'''

    #modelSelection(regressAData(), validateData(), verbose=True)
    #modelSelection(regressBData(), validateData(), verbose=True)

    #X, Y = blogTrainData()
    #print(blogRegressionPlot(X, Y, params={'lambda': 0.000001}, plot=False))
    print(blogModelSelection(blogTrainData(), blogValidateData(), verbose=True))


