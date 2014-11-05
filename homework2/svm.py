def svm(x, y, slackTightness):
    '''Computes a support vector machine from training data 'data' (which is in the form:
    data = [(x, classification), ...]
        x = (x0, x1, ...)
        classification = +1 or -1),
    and slackTightness, a positive real number indicating how hard we should try to
    keep all of the points on the appropriate side of the margin.'''

    multipliers = svmMultipliers(data)
    w = svmWeights(x, y, multipliers)
    M = [i
        for i in range(len(multipliers))
        if 0 < multipliers[i] < C]

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

def svmWeights(x, y, multipliers):
    return sum([
        multipliers[i] * x[i] * y[i]
        for i in range(len(multipliers))])

def svmMultipliers(data):
    pass