import pandas as pd
import algorithms.linreg as lin
import algorithms.logreg as log
import algorithms.polyreg as poly


def read_data_from_csv(path):
    data_frame = pd.read_csv(path, index_col=0)
    return data_frame


def manipulate_data(filename, type_reg, exp):
    my_data = read_data_from_csv('data/' + filename).as_matrix()
    if type_reg == 'LinReg':
        linear_object = lin.function(my_data)
        return linear_object
    if type_reg == 'PolyReg':
        polynomial_object = poly.function(my_data, exp)
        return polynomial_object
    if type_reg == 'LogReg':
        logistic_object = log.function(my_data)
        return logistic_object
    return 0
