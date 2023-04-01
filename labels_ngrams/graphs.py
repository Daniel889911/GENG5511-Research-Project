import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def create_heatmap2(ngram_metrics):
    data = []
    for metric_dict in ngram_metrics:
        for label, (ngram, percentage_value) in metric_dict.items():
            data.append({'Label': label, 'Ngram': ngram, 'Metric': percentage_value})
    df = pd.DataFrame(ngram_metrics, columns=['Label', 'Ngram', 'Metric'])
    df2 = df.sort_values(by="Metric", ascending=True)
    df3 = df2.pivot("Label", "Ngram", values='Metric')

    # Set up the figure and gridspec with an increased figure height
    fig = plt.figure(figsize=(10, 30))  # Adjust the 20 to a suitable height
    gs = gridspec.GridSpec(nrows=len(df3), ncols=1, height_ratios=[1] * len(df3))

    # Create the heatmap with increased vertical spacing
    ax = plt.subplot(gs[:])
    sns.heatmap(df3, cmap='RdYlGn', annot=True, ax=ax)

    # Adjust the y-axis label size
    ax.yaxis.set_tick_params(labelsize=8)

    # Show the heatmap
    plt.show()

def create_heatmap(ngram_metrics):
    data = []
    for metric_dict in ngram_metrics:
        for label, (ngram, percentage_value) in metric_dict.items():
            data.append({'Label': label, 'Ngram': ngram, 'Metric': percentage_value})
    df = pd.DataFrame(data, columns=['Label', 'Ngram', 'Metric'])
    df2 = df.sort_values(by="Metric", ascending=True)
    df3 = df2.pivot("Label", "Ngram", values='Metric')
    sns.heatmap(df3, cmap='RdYlGn', annot=True)
