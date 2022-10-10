import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def create_pie_chart(label_list):
    for label_data in label_list:
        # Creating dataset
        label = ['FULL AGREEMENT','PARTIAL AGREEMENT']
        
        data = [label_data[1], label_data[2]]    
        
        # Creating explode data
        explode = (0.1, 0.0)
        
        # Creating color parameters
        colors = ( "blue", "magenta")
        
        # Wedge properties
        wp = { 'linewidth' : 1, 'edgecolor' : "green" }
        
        # Creating autocpt arguments
        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} g)".format(pct, absolute)
        
        # Creating plot
        fig, ax = plt.subplots(figsize =(10, 7))
        wedges, texts, autotexts = ax.pie(data,
                                        autopct = lambda pct: func(pct, data),
                                        explode = explode,
                                        labels = label,
                                        shadow = True,
                                        colors = colors,
                                        startangle = 90,
                                        wedgeprops = wp,
                                        textprops = dict(color ="yellow"))
        
        # Adding legend
        ax.legend(wedges, label,
                title = label_data[0],
                loc ="center left",
                bbox_to_anchor =(1, 0, 0.5, 1))
        
        plt.setp(autotexts, size = 8, weight ="bold")
        ax.set_title("Individual Label Annotation Metrics")
        
        # show plot
        plt.show()

def create_heatmap(ngram_metrics):
    df = pd.DataFrame(ngram_metrics, columns =['Label', 'Ngram', 'Metric'])
    df2 = df.sort_values(by="Metric", ascending=True)
    df3 = df2.pivot("Label", "Ngram",values='Metric')
    sns.heatmap(df3,cmap='RdYlGn', annot=True)