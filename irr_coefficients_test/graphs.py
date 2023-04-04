import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def plot_agreement_coefficients(agreement_data, title='', x_axis_title='', y_axis_title=''):
    plt.figure(figsize=(10, 6))

    coefficients = list(agreement_data.values())[0].keys()
    
    for coefficient in coefficients:
        x = []
        y = []
        for percentage, coef_data in agreement_data.items():
            x.append(percentage)
            y.append(coef_data[coefficient])
        plt.plot(x, y, label=coefficient)

    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

