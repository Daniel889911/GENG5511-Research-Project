import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def create_bar_chart(data, x_label, y_label, title):
    # Create a bar chart using Seaborn
    sns.set(style="whitegrid")
    ax = sns.barplot(x=[title], y=[data], palette="Blues_d")

    # Add labels to the chart
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.set_ylim([0, 1])  # Set the y-axis limits to start from 0 and go up to 1

    # Add the value on top of the bar
    ax.text(0, data + 0.01, str(round(data, 3)), color='black', ha='center')

    # Show the chart
    plt.show()



