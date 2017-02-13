import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def cost_function(par_c, par_x, par_y):
    return sigmoid_value(par_c[0] + par_c[1] * par_x) - par_y


def logistic_regression(par_x, par_y):
    alpha = 0.01
    old_c = np.array([0, 0])
    new_c = old_c+0.1
    precision = 0.001
    change = abs(new_c-old_c)
    i = 0
    while (change > precision).any():
        old_c = np.array(new_c)
        new_c[0] += -alpha * np.sum(cost_function(old_c, par_x, par_y))
        new_c[1] += -alpha * np.sum(np.multiply(cost_function(old_c, par_x, par_y), par_x))
        change = abs(new_c - old_c)
        i += 1
        if i > 20000:
            break
    return new_c


def function(data):
    x = data[:, 0]
    y = data[:, 1]
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
    sss.get_n_splits(x, y)
    for train_index, test_index in sss.split(x, y):
        x_training, x_test = x[train_index], x[test_index]
        y_training, y_test = y[train_index], y[test_index]
    coefficients = logistic_regression(x_training, y_training)
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(x_test.size):
        if y_test[i] == 1 and sigmoid_value(coefficients[0] + coefficients[1] * x_test[i]) > 0.5:
            tp += 1
        if y_test[i] == 1 and sigmoid_value(coefficients[0] + coefficients[1] * x_test[i]) < 0.5:
            fn += 1
        if y_test[i] == 0 and sigmoid_value(coefficients[0] + coefficients[1] * x_test[i]) > 0.5:
            fp += 1
        if y_test[i] == 0 and sigmoid_value(coefficients[0] + coefficients[1] * x_test[i]) < 0.5:
            tn += 1
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return LogisticObject(x_training, y_training, x_test, y_test, coefficients, LogisticError(P, R, F, acc))


class LogisticObject:
    def __init__(self, x_training, y_training, x_test, y_test, coefficients, error):
        self.x_training = x_training
        self.y_training = y_training
        self.x_test = x_test
        self.y_test = y_test
        self.coefficients = coefficients
        self.error = error


class LogisticError:
    def __init__(self, P, R, F, acc):
        self.P = P
        self.R = R
        self.F = F
        self.acc = acc


def sigmoid_value(x):
    return 1/(1+np.exp(-x))
