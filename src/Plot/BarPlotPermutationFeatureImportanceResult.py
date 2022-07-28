import abc

import numpy as np
import matplotlib.pyplot as plt


class BarPlotPermutationFeatureImportanceResult(metaclass=abc.ABCMeta):
    def plot(self, results, attributes):
        for name, result in results.items():
            self.sub_plot(result, name, attributes)

    def sub_plot(self, result, name, attributes):
        tree_importance_sorted_idx = np.argsort(result)
        tree_importance_sorted = result[tree_importance_sorted_idx]

        np_attributes = np.array(attributes)
        attributes_sorted = np_attributes[tree_importance_sorted_idx]

        fig = plt.figure(figsize=(10, 5))
        plt.barh(attributes_sorted, tree_importance_sorted, color='maroon')

        plt.xlabel("Importancia")
        plt.ylabel("Atributo")
        plt.title(f"Permutation feature importance [{name}]")
        plt.show()
