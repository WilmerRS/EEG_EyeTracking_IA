import abc

# machine learning
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance


class PermutationImportanceProcess(metaclass=abc.ABCMeta):
    def _permutation_importance_cross(self, X, y, classifiers, names, splits=15, n_repeats=30):
        skf = StratifiedKFold(n_splits=splits)

        result_cross = {}

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            for name, classifier in zip(names, classifiers):
                print(f'\n ** {name} ** \n')
                classifier.fit(X_train, y_train)
                result = permutation_importance(
                    classifier,
                    X_train,
                    y_train,
                    n_repeats=n_repeats,
                    random_state=0
                )
                importance_mean = result.importances_mean
                print(importance_mean)
                if name not in result_cross:
                    result_cross[name] = importance_mean / splits
                else:
                    result_cross[name] += importance_mean / splits

                if "general" not in result_cross:
                    result_cross["general"] = importance_mean / splits
                else:
                    result_cross["general"] += importance_mean / splits

        return result_cross
