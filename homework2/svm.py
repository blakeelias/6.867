import numpy as np
from scipy import optimize
import pylab as pl
#from mpl_toolkits.mplot3d.axes3d import Axes3D
from cvxopt import matrix, solvers

import unittest

def svm(x, y, slackTightness, verbose=False):
    '''Computes a support vector machine from training data 'data' (which is in the form:
    data = [(x, classification), ...]
        x = (x0, x1, ...)
        classification = +1 or -1),
    and slackTightness, a positive real number indicating how hard we should try to
    keep all of the points on the appropriate side of the margin.'''

    multipliers = svmMultipliers(x, y, slackTightness, verbose)
    if verbose:
        print('slackTightness', slackTightness)
        print('multipliers', multipliers)

    w = svmWeights(x, y, multipliers)
    M = [i
        for i in range(len(multipliers))
        if 0 < multipliers[i] < slackTightness]
    
    if verbose:
        print('M', M)

    S = [i
        for i in range(len(multipliers))
        if multipliers[i] != 0] # support vectors

    w_0 = 1./len(M) * sum([
        y[j] - sum([
            multipliers[i]*y[i]*x[j].T.dot(x[i])
            for i in S])
        for j in M])

    def classify(example):
        return w.T.dot(example) + w_0 > 0

    if verbose:
        print(w, w_0)

    return classify

def svmWeights(x, y, multipliers):
    return sum([
        multipliers[i] * x[i] * y[i]
        for i in range(len(multipliers))])

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

def svmMultipliers(x, y, slackTightness, verbose=False):
    Q = matrix([
        [float(y[i]*y[j]*x[i].dot(x[j]))
            for j in range(len(x))]
        for i in range(len(x))])
    print('Q')
    print(Q)

    p = matrix([1.0 for i in range(len(x))])
    
    I = np.identity(len(x))
    G = matrix(np.vstack((I, -I)))

    h = matrix([0.0] * len(x)
        + [slackTightness] * len(x))

    A = matrix([1.0] * len(x), (1, len(x)))

    b = matrix(0.0)

    sol=solvers.qp(Q, p, G, h, A, b)

    return sol['x']

def testOptimize():
    def objectiveFunction(args):
        x, y = args
        return x*x + y*y

    jacobian = numericalGradient(objectiveFunction)

    constraints = ({'type': 'eq',
                    'fun': lambda args: args[0] - 5},
                    {'type': 'eq',
                    'fun': lambda args: args[1] - 5})

    x0 = np.random.randn(2)
    res_cons = optimize.minimize(objectiveFunction,x0,jac=jacobian,constraints=constraints,
        method='SLSQP',options={'disp':False})

    return res_cons['x']

class TestSVM(unittest.TestCase):

    def test_svmWeights(self):
        x = map(lambda xx: np.array(xx), [(1, 2), (2, 2), (0, 0), (-2, 3)])
        y = [1, 1, -1, -1]
        weights = [2, 3, 4, 5]
        print(svmWeights(x, y, weights))
        print(np.array((18, -5)))

    def test_svmMultipliers(self):
        pass

def main():
    x = [(1, 2), (2, 2), (0, 0), (-2, 3)]
    y = [1, 1, -1, -1]
    classifier = svm(map(lambda xx: np.array(xx), x), y, 100)

    print(classifier(np.array((1, 2))))
    print(classifier(np.array((2, 2))))
    print(classifier(np.array((0, 0))))
    print(classifier(np.array((-2, 3))))

    print(classifier(np.array((10, 10))))
    print(classifier(np.array((-10, -10))))

if __name__ == '__main__':
    #print(testOptimize())
    main()
    #unittest.main()
