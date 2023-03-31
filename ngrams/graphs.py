import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def create_pie_chart(label_list):
    for label_data in label_list:
        # Creating dataset
        label = ['FULL LABEL AGREEMENT','PARTIAL LABEL AGREEMENT']
        
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
                                        textprops = dict(color ="black"))
        
        # Adding legend
        ax.legend(wedges, label,
                title = label_data[0],
                loc ="center left",
                bbox_to_anchor =(1, 0, 0.5, 1))
        
        plt.setp(autotexts, size = 8, weight ="bold")
        ax.set_title("Individual Label Annotation Metrics")
        
        # show plot
        plt.show()

def create_pie_chart_ngram(ngram_list):
    for ngram_data in ngram_list:
        # Creating dataset
        label = ['FULL NGRAM AGREEMENT','PARTIAL NGRAM AGREEMENT']
        
        data = [ngram_data[1], ngram_data[2]]    
        
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
                                        textprops = dict(color ="black"))
        
        # Adding legend
        ax.legend(wedges, label,
                title = ngram_data[0],
                loc ="center left",
                bbox_to_anchor =(1, 0, 0.5, 1))
        
        plt.setp(autotexts, size = 8, weight ="bold")
        ax.set_title("Ngram Type Annotation Metrics")
        
        # show plot
        plt.show()

def create_pie_chart_individual_ngram(individual_ngram_metrics):
        # Creating dataset
        label = ['FULL NGRAM AGREEMENT','PARTIAL NGRAM AGREEMENT']
        
        data = [individual_ngram_metrics[1], individual_ngram_metrics[2]]    
        
        # Creating explode data
        explode = (0.1, 0.0)
        
        # Creating color parameters
        colors = ( "silver", "orchid")
        
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
                                        textprops = dict(color ="black"))
        
        # Adding legend
        ax.legend(wedges, label,
                title = individual_ngram_metrics[0],
                loc ="center left",
                bbox_to_anchor =(1, 0, 0.5, 1))
        
        plt.setp(autotexts, size = 8, weight ="bold")
        ax.set_title("Individual Annotator Ngram Annotation Metrics")
        
        # show plot
        plt.show()


