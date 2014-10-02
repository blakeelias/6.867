import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
from matplotlib import pyplot

# Problem 1
def gradientDescent(func, gradient, guess, stopChange=0.001, stepRate=0.01, momentumWeight=0.1, verbose=False):
    print('performing gradient descent')
    #print('examining guess parameter, func takes inputs with %s component(s)' % len(guess))
    lastChange = float('inf')
    prevGuess = guess
    funcVal = func(guess)

    nFuncCalls = 1
    nGradCalls = 0
    
    if verbose:
        print('guess, gradient, funcVal, lastChange')

    while lastChange > stopChange:
        g = gradient(guess)
        nGradCalls += 1
        if verbose: print('%s, %s, %s, %s' % (guess, g, funcVal, lastChange))

        newGuess = guess - stepRate * g \
         + momentumWeight*(guess - prevGuess)
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
def numericalGradient(func, point, intervalWidth = 1e-3):
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

# Problem 2.1
def designMatrix(X, order, includeConstantTerm=True):
    return np.array([[x[0]**i for i in range(0 if includeConstantTerm else 1, order+1)] for x in X])

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
def regressionPlot(X, Y, order, fitMethod=regressionFit, params={}):
    print('X', X)
    print('Y', Y)
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    params['order'] = order
    w = fitMethod(X, Y, phi, params)
    print 'w', w

    Y_estimated = pl.dot(w.T, phi.T)
    print('Y_estimated', Y_estimated)
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])
    print('error: %f' % sumOfSquaresErrorGenerator(phi, Y)(w))
    print('analytical error gradient: %s' % sumOfSquaresErrorGradientGenerator(phi, Y)(w))
    print('numerical error gradient: %s' % \
        numericalGradient(sumOfSquaresErrorGenerator(phi, Y), w, 1e-5))

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
def ridgeFit(X, Y, phi, params):
    phi = designMatrix(X, params['order'], includeConstantTerm=False)
    print('phi', phi)
    l = params['lambda']
    phi_avg = sum(phi)*1.0 / len(phi)
    print('phi_avg', phi_avg)
    Z = phi - phi_avg
    print('Z', Z)
    Y_avg = sum(Y)*1.0 / len(Y)
    Yc = Y - Y_avg
    print('Yc', Yc)

    a = pl.dot(Z.T, Z) + l * pl.eye(len(Z.T))
    b = np.linalg.inv(a).dot(Z.T)
    W = b.dot(Yc)
    W_0 = np.array([Y_avg - W.T.dot(phi_avg)])
    print('W_0', W_0)
    print('W', W)
    return np.hstack((W_0, W.T)).T

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

    X, Y = bishopCurveData()
    #X = np.array([[1], [2], [3], [4]])
    #Y = np.array([[2], [3], [6], [9]])
    for M in [0, 1, 3, 9]:
        #regressionPlot(X, Y, M, regressionFit)
        #regressionPlot(X, Y, M, gradientDescentFit)
        regressionPlot(X, Y, M, ridgeFit, {'lambda': 0.00001})
        pl.show()
        print('-'*30)
    '''X, Y = regressBData()
    for M in [1, 3, 9]:
        #regressionPlot(X, Y, M, regressionFit)
        #regressionPlot(X, Y, M, gradientDescentFit)
        regressionPlot(X, Y, M, ridgeFit, {'lambda': 0.00001})
        pl.show()
        print('-'*30)'''