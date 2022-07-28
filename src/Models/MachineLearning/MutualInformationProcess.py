import abc

# machine learning
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MutualInformationProcess(metaclass=abc.ABCMeta):

    def _mutual_information_graph(self, scores):
        plt.figure(dpi=100, figsize=(8, 5))
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")
        plt.show()

    def _mutual_information(self, X, y):
        discrete_features = self._encode_names(X)

        print("discrete_features", discrete_features, len(discrete_features))
        
        scores = mutual_info_classif(X, y, discrete_features=discrete_features)
        scores = pd.Series(scores, name="MI Scores", index=X.columns)
        scores = scores.sort_values(ascending=False)
        return scores

    def _encode_names(self, X):
        for col_name in X.select_dtypes("object"):
            X[col_name], _ = X[col_name].factorize()

        # All discrete features should now have integer dtypes (double-check this before using MI!)
        return X.dtypes == int
