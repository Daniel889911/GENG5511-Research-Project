import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_edit_distance_agreements(agreement_data, title='', x_axis_title='', y_axis_title=''):
    plt.figure(figsize=(10, 6))
    
    x = [key for key in agreement_data.keys()]
    y = [value['Overall Reliability'] for value in agreement_data.values()]
    plt.plot(x, y)

    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(title)
    plt.grid()
    plt.show()



