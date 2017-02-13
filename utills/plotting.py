from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.layouts import gridplot
import numpy as np
from algorithms.linreg import LinearObject
from algorithms.logreg import LogisticObject, sigmoid_value
from algorithms.polyreg import PolyObject, polynomial_value


class Components:
    def __init__(self, script, div, error, reg_type, coefficients):
        self.script = script
        self.div = div
        self.error = error
        self.reg_type = reg_type
        self.coefficients = coefficients


def create(processed_data):
    if type(processed_data) is LinearObject:
        # training dataset
        p = figure(title="Training dataset", x_axis_label='x', y_axis_label='y')
        p.scatter(processed_data.x_training, processed_data.y_training, line_width=2)
        x_line = np.linspace(min(processed_data.x_training), max(processed_data.x_training),
                             (max(processed_data.x_training) - min(processed_data.x_training)) * 100)
        y_line = processed_data.coefficients[1] + processed_data.coefficients[0] * x_line
        p.line(x_line, y_line, line_width=2, color="#FF0000")

        # test dataset
        test = figure(title="Test dataset", x_axis_label='x', y_axis_label='y')
        test.scatter(processed_data.x_test, processed_data.y_test, line_width=2)
        x_line = np.linspace(min(processed_data.x_training), max(processed_data.x_training),
                             (max(processed_data.x_training) - min(processed_data.x_training)) * 100)
        y_line = processed_data.coefficients[1] + processed_data.coefficients[0] * x_line
        test.line(x_line, y_line, line_width=2, color="#FF0000")

        # merge
        plot = gridplot([[p, test]])
        script, div = components(plot)

        result = Components(script, div,
                            processed_data.error, 'Linear regression', np.flipud(processed_data.coefficients))
        return result
    if type(processed_data) is PolyObject:

        # training dataset
        p = figure(title="Training dataset", x_axis_label='x', y_axis_label='y')
        p.scatter(processed_data.x_training, processed_data.y_training, line_width=2)
        x_line = np.linspace(min(processed_data.x_training), max(processed_data.x_training),
                             (max(processed_data.x_training) - min(processed_data.x_training)) * 100)
        y_line = polynomial_value(x_line, processed_data.coefficients)
        p.line(x_line, y_line, color="#FF0000")

        # test dataset
        test = figure(title="Test dataset", x_axis_label='x', y_axis_label='y')
        x_line = np.linspace(min(processed_data.x_test), max(processed_data.x_test),
                             (max(processed_data.x_test) - min(processed_data.x_test)) * 100)
        y_line = polynomial_value(x_line, processed_data.coefficients)
        test.line(x_line, y_line,  color="#FF0000")
        test.scatter(processed_data.x_test, processed_data.y_test, line_width=2)

        # merge
        plot = gridplot([[p, test]])
        script, div = components(plot)

        result = Components(script, div, processed_data.error, 'Polynomial regression', processed_data.coefficients)
        return result

    if type(processed_data) is LogisticObject:

        # training dataset
        p = figure(title="Training dataset", x_axis_label='x', y_axis_label='y')
        p.scatter(processed_data.x_training, processed_data.y_training, line_width=2)
        x_line = np.linspace(min(processed_data.x_training), max(processed_data.x_training),
                             (max(processed_data.x_training) - min(processed_data.x_training)) * 100)
        y_line = sigmoid_value(processed_data.coefficients[0] + processed_data.coefficients[1] * x_line)
        p.line(x_line, y_line, color="#FF0000")

        # test dataset
        test = figure(title="Test dataset", x_axis_label='x', y_axis_label='y')
        x_line = np.linspace(min(processed_data.x_test), max(processed_data.x_test),
                             (max(processed_data.x_test) - min(processed_data.x_test)) * 100)
        y_line = sigmoid_value(processed_data.coefficients[0] + processed_data.coefficients[1] * x_line)
        test.line(x_line, y_line,  color="#FF0000")
        test.scatter(processed_data.x_test, processed_data.y_test)

        # merge
        plot = gridplot([[p, test]])
        script, div = components(plot)

        result = Components(script, div, processed_data.error, 'Logistic regression', processed_data.coefficients)
        return result
    return 0
