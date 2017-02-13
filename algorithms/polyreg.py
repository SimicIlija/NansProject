import numpy as np
from sklearn.model_selection import train_test_split


def create_matrix(par_x, m):
    arr = [np.ones_like(par_x)]
    for i in range(1, m + 1):
        temp = np.power(par_x, i)
        arr = np.append(arr, [temp], axis=0)
    return np.transpose(arr)


def polynomial_regression(x, y, m):
    A = create_matrix(x, m)
    return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A), A)), np.transpose(A)), y)


def polynomial_value(par_x, par_c):
    i = 0
    res = 0
    for c in par_c:
        res += c * (par_x ** i)
        i += 1
    return res


def sum_of_square_errors(par_x, par_y, par_c):
    sub = polynomial_value(par_x, par_c) - par_y
    res = np.sum(np.multiply(sub, sub))
    return res


def sum_of_square_total(par_y):
    sub = par_y - np.mean(par_y)
    res = np.sum(np.multiply(sub, sub))
    return res


def sum_of_square_regression(par_x, par_y, par_c):
    sub = polynomial_value(par_x, par_c) - np.mean(par_y)
    res = np.sum(np.multiply(sub, sub))
    return res


def coefficient_of_determination(par_x, par_y, par_c):
    return sum_of_square_regression(par_x, par_y, par_c)/sum_of_square_total(par_y)


def typical_prediction_error(par_x, par_y, par_c, m):
    res = np.sqrt(sum_of_square_errors(par_x, par_y, par_c)/(np.size(par_x) - m - 1))
    return res


def coefficient_of_correlation(par_x, par_y):
    sub_x = par_x - np.mean(par_x)
    sub_y = par_y - np.mean(par_y)
    numerator = np.sum(np.multiply(sub_x, sub_y))
    denominator = (np.size(par_x)) * np.std(par_x) * np.std(par_y)
    res = numerator/denominator
    return res


class ErrorObject:
    def __init__(self, sse, sst, ssr, r2, s, r):
        self.sse = sse
        self.sst = sst
        self.ssr = ssr
        self.r2 = r2
        self.s = s
        self.r = r


class PolyObject:
    def __init__(self, x_training, y_training, x_test, y_test, coefficients, error):
        self.x_training = x_training
        self.y_training = y_training
        self.x_test = x_test
        self.y_test = y_test
        self.coefficients = coefficients
        self.error = error


def function(data, m):

    x = data[:, 0]
    y = data[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    coefficients = polynomial_regression(np.array(x_train), np.array(y_train), m)
    sse = sum_of_square_errors(x_test, y_test, coefficients)
    sst = sum_of_square_total(y_test)
    ssr = sum_of_square_regression(x_test, y_test, coefficients)
    r2 = coefficient_of_determination(x_test, y_test, coefficients)
    s = typical_prediction_error(x_test, y_test, coefficients, m)
    r = coefficient_of_correlation(x_test, y_test)

    return PolyObject(x_train, y_train, x_test, y_test, coefficients, ErrorObject(sse, sst, ssr, r2, s, r))