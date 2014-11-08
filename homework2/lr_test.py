from numpy import *
from plotBoundary import *
import klr

# parameters
name = 'ls2'
print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

print('X')
print(X)
print('Y')
print(Y)

# Carry out training.
### TODO ###

predictLR = klr.lr(array(X), array(Y))

# Define the predictLR(x) function, which uses trained parameters
### TODO ###

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_test.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
