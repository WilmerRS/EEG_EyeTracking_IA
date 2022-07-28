import abc

import numpy as np
import matplotlib.pyplot as plt


class BarPlotResult(metaclass=abc.ABCMeta):
    def plot(self, results, names):
        X = np.arange(len(results[0]))
        fig = plt.subplots(figsize =(12, 8))
        # ax = fig.add_axes([0, 0, 1, 1])
        plt.bar(X + 0.00, results[0], color='b', width=0.10, label="Accuracy")
        plt.bar(X + 0.10, results[1], color='g', width=0.10, label="Precision")
        plt.bar(X + 0.20, results[2], color='c', width=0.10, label="Recall")
        plt.bar(X + 0.30, results[3], color='r', width=0.10, label="F1")

        plt.xlabel('Modelos', fontweight='bold', fontsize=15)
        plt.ylabel('Porcentaje', fontweight='bold', fontsize=15)
        plt.xticks([r + 0.10 for r in range(len(results[0]))],
                   names)
        plt.legend()
        plt.grid()
        plt.show()
