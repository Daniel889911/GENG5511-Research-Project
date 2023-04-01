import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def create_pie_chart(label_proportions: dict, threshold: float = 5.0):
    # Group smaller proportions into an "Others" category
    grouped_proportions = {}
    others = 0
    for key, value in label_proportions.items():
        if value >= threshold:
            grouped_proportions[key] = value
        else:
            others += value
    if others > 0:
        grouped_proportions["Others"] = others

    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(grouped_proportions.values(), labels=grouped_proportions.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
    plt.show()



