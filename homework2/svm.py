import numpy as np
from scipy import optimize
import pylab as pl
#from mpl_toolkits.mplot3d.axes3d import Axes3D

import unittest

def svm(x, y, slackTightness):
    '''Computes a support vector machine from training data 'data' (which is in the form:
    data = [(x, classification), ...]
        x = (x0, x1, ...)
        classification = +1 or -1),
    and slackTightness, a positive real number indicating how hard we should try to
    keep all of the points on the appropriate side of the margin.'''

    multipliers = svmMultipliers(x, y, slackTightness)
    w = svmWeights(x, y, multipliers)
    M = [i
        for i in range(len(multipliers))
        if 0 < multipliers[i] < slackTightness]

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

def svmMultipliers(x, y, slackTightness):
    print('x', x)
    print('len(x)', len(x))
    def objectiveFunction(multipliers):
        return sum(multipliers) - 0.5 * sum([
            sum([
                multipliers[i]*multipliers[j]*y[i]*y[j]*(x[i].dot(x[j]))
                for j in range(len(x))])
            for i in range(len(x))])

    jacobian = numericalGradient(objectiveFunction)

    constraints = ({'type': 'ineq',
                    'fun': lambda mult: mult},

                    {'type': 'ineq',
                    'fun': lambda mult: slackTightness - mult},

                    {'type': 'eq',
                    'fun': lambda mult: 
                        sum([mult[i] * y[i]
                            for i in range(len(mult))])})

    x0 = np.random.randn(len(x[0]))
    res_cons = optimize.minimize(objectiveFunction,x0,jac=jacobian,constraints=constraints,
        method='SLSQP',options={'disp':False})

    return res_cons['x']

def testOptimize():
    def objectiveFunction(args):
        x, y = args
        return x*x + y*y

    jacobian = numericalGradient(objectiveFunction)

    constraints = ({'type': 'ineq',
                    'fun': lambda mult: mult},

                    {'type': 'ineq',
                    'fun': lambda mult: slackTightness - mult},

                    {'type': 'eq',
                    'fun': lambda mult: 
                        sum([mult[i] * y[i]
                            for i in range(len(mult))])})

    x0 = np.random.randn(len(x))
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
    classifier = svm(map(lambda xx: np.array(xx), x), y, 10)
    print(classifier(np.array((10, 10))))
    print(classifier(np.array((-10, -10))))

if __name__ == '__main__':
    #testOptimize()
    main()
    #unittest.main()
