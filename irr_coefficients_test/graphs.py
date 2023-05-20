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

    # Enlarging axes titles
    plt.xlabel(x_axis_title, fontsize=16)
    plt.ylabel(y_axis_title, fontsize=16)
    plt.title(title, fontsize=16)

    # Enlarging axes values
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Setting y-axis to start at -1
    plt.ylim(bottom=-1)

    # Enlarging legend values
    plt.legend(fontsize=16)

    plt.grid()
    plt.show()



