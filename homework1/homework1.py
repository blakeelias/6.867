import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np

def gradientDescent(func, gradient, guess, stopChange=0.001, stepRate=0.01, momentumWeight=0.1):
    lastChange = float('inf')
    prevGuess = guess
    print('guess, gradient, lastChange')
    while lastChange > stopChange:
        g = gradient(guess)
        print('%s, %s, %s' % (guess, g, lastChange))
        newGuess = guess - stepRate * g \
                   + momentumWeight*(guess - prevGuess)
        prevGuess = guess
        guess = newGuess
        lastChange = np.linalg.norm(guess - prevGuess)
    return guess

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    print 'w', w
    # produce a plot of the values of the function 
    pts = [[p] for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])

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
    def bowl(x):
        a, b = x
        return a*a + b*b

    def bowlGradient(x):
        a, b = x
        return np.array((2*a, 2*b))

    z = gradientDescent(bowl, bowlGradient, np.array((3, 5)), 
        stopChange=0.00000001,
        stepRate=0.01,
        momentumWeight=0.1)
    print(z)
