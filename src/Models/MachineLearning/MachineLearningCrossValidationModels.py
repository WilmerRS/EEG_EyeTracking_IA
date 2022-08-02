import abc
import os

# machine learning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import cross_val_score

# models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

# domain
from src.utils.get_root_path import get_root_path


class MachineLearningCrossValidationModels(metaclass=abc.ABCMeta):
    def _ml_models(self):
        names = [
            # "Nea-Nearest Neighbors: (KNeighborsClassifier)",
            "SVM-Linear SVM: (SVC)",
            "1RBF-RBF SVM: (SVC)",
            # "2RBF-RBF SVM: (SVC)",
            # "3RBF-RBF SVM: (SVC)",
            # "4RBF-RBF SVM: (SVC)",
            # "5RBF-RBF SVM: (SVC)",
            "GPC-Gaussian Process: (GaussianProcessClassifier)",
            "DT-Decision Tree: (DecisionTreeClassifier)",
            "RF-Random Forest: (RandomForestClassifier)",
            "MPL1-Neural Net: (MLPClassifier)",
            # "MPL2-Deep Neural Net: (MLPClassifier)",
            # "AB1-AdaBoost: (AdaBoostClassifier)",
            # "AB2-AdaBoost: (AdaBoostClassifier)",
            "AB3-AdaBoost: (AdaBoostClassifier)",
            # "GNB-Naive Bayes: (GaussianNB)",
            # "QDA-QDA: (QuadraticDiscriminantAnalysis)",
            "XGBC-XGBClassifier: (XGBClassifier)",
            # "XGBC2-XGBClassifier: (XGBClassifier)",
            # "XGBC3-XGBClassifier: (XGBClassifier)",
            # "XGBC4-XGBClassifier: (XGBClassifier)",
        ]

        classifiers = [
            # KNeighborsClassifier(2),
            SVC(kernel="linear", C=0.025),
            SVC(cache_size=1,gamma='scale', C=1),
            # SVC(cache_size=1,gamma='scale', C=0.15),
            # SVC(cache_size=1,shrinking=True,gamma='scale', C=0.025, kernel='poly'),
            # SVC(cache_size=1,probability=True,shrinking=True,gamma='scale', C=0.025, kernel='rbf'),
            # SVC(cache_size=1,gamma='scale', C=0.025, kernel='sigmoid'),
            # SVC(cache_size=1,gamma='scale', C=0.025, kernel='precomputed'),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1
            ),
            MLPClassifier(alpha=1, max_iter=1000),
            # MLPClassifier(alpha=0.01, hidden_layer_sizes=(100, 100)),
            # AdaBoostClassifier(),
            # AdaBoostClassifier(learning_rate=0.01),
            AdaBoostClassifier(learning_rate=0.0001),
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis(),
            XGBClassifier(
                objective='binary:logistic',
                n_estimators=1000,
                seed=0,
                learning_rate=0.01,
                random_state=1
            ),
            # XGBClassifier(
            #     objective='binary:logistic',
            #     n_estimators=100,
            #     seed=0,
            #     learning_rate=0.0001,
            #     random_state=1
            # ),
            # XGBClassifier(
            #     objective='binary:logistic',
            #     n_estimators=200,
            #     seed=0,
            #     learning_rate=0.00001,
            #     random_state=1
            # ),
            # XGBClassifier(
            #     objective='binary:logistic',
            #     n_estimators=300,
            #     seed=0,
            #     learning_rate=0.000001,
            #     random_state=1
            # ),
        ]
        return classifiers, names

    def _validate_ml_classifier_models(self, X, y):
        classifiers, names = self._ml_models()

        cv = RepeatedKFold(n_splits=3, n_repeats=20, random_state=1)

        accuracies = []
        precision_s = []
        f1_s = []
        recalls = []

        for name, classifier in zip(names, classifiers):
            # accuracy = cross_val_score(
            #     classifier, X, y,
            #     cv=cv, scoring='accuracy'
            # )
            # precision =cross_val_score(
            #     classifier, X, y,
            #     cv=cv, scoring='precision'
            # )
            # recall = cross_val_score(
            #     classifier, X, y,
            #     cv=cv, scoring='recall'
            # )
            f1 = cross_val_score(
                classifier, X, y,
                cv=cv, scoring='f1'
            )
            print(f'\n ** {name} ** \n')
            # print(
            #     f"accuracy:   {accuracy.mean():.3f} (std) {accuracy.std():.3f}")
            # print(f"precision:        {precision.mean():.3f} (std) {precision.std():.3f}")
            # print(f"recall:     {recall.mean():.3f} (std) {recall.std():.3f}")
            print(f"f1:         {f1.mean():.3f} (std) {f1.std():.3f}")

            accuracies.append(0)#accuracy.mean())
            precision_s.append(0)#precision.mean())
            recalls.append(0)#recall.mean())
            f1_s.append(f1.mean())

            # classifier.fit(X_train, y_train)
            # predictions_val = classifier.predict(X_valid)
            # c_matrix = confusion_matrix(y_valid, predictions_val.astype(int))

            # print("Example one confusion matrix:")
            # print(c_matrix)
        return [
            accuracies,
            precision_s,
            recalls,
            f1_s,
        ], [name.split('-')[0] for name in names ]
