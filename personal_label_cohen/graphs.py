import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def create_pie_chart(agreements_list):
    for annotator_data in agreements_list:
        annotator = list(annotator_data.keys())[0]
        labels_data = annotator_data[annotator]

        for label_data in labels_data:
            label = ["Agreement", "Disagreement"]
            key = label_data[0]  # The label
            data = label_data[1:]  # Agreement and Disagreement

            # Creating explode data
            explode = (0.1, 0.0)

            # Creating color parameters
            colors = ("blue", "magenta")

            # Wedge properties
            wp = {'linewidth': 1, 'edgecolor': "green"}

            # Creating autocpt arguments
            def func(pct, allvalues):
                absolute = int(pct / 100. * np.sum(allvalues))
                return "{:.1f}%\n({:d})".format(pct, absolute)

            # Creating plot
            fig, ax = plt.subplots(figsize=(10, 7))
            wedges, texts, autotexts = ax.pie(data,
                                              autopct=lambda pct: func(pct, data),
                                              explode=explode,
                                              labels=label,
                                              shadow=True,
                                              colors=colors,
                                              startangle=90,
                                              wedgeprops=wp,
                                              textprops=dict(color="black"))

            # Adding legend
            ax.legend(wedges, label,
                      title=f"{annotator}: {key}",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))

            plt.setp(autotexts, size=8, weight="bold")
            ax.set_title(f"{annotator}: Individual Label Annotation Metrics")

            # Show plot
            plt.show()


