import abc
import os

# machine learning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

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
            # "SVM-Linear SVM: (SVC)",
            # "1RBF-RBF SVM: (SVC)",
            # "2RBF-RBF SVM: (SVC)",
            # "3RBF-RBF SVM: (SVC)",
            # "4RBF-RBF SVM: (SVC)",
            # "5RBF-RBF SVM: (SVC)",
            # "GPC-Gaussian Process: (GaussianProcessClassifier)",
            # "DT-Decision Tree: (DecisionTreeClassifier)",
            # "RF-Random Forest: (RandomForestClassifier)",
            # "2RF-Random Forest: (RandomForestClassifier)",
            # "3RF-Ra-*ndom Forest: (RandomForestClassifier)",
            # "MPL1-Neural Net: (MLPClassifier)",
            # "MPL2-Deep Neural Net: (MLPClassifier)",
            # "AB1-AdaBoost: (AdaBoostClassifier)",
            # "AB2-AdaBoost: (AdaBoostClassifier)",
            # "AB3-AdaBoost: (AdaBoostClassifier)",
            # "GNB-Naive Bayes: (GaussianNB)",
            # "QDA-QDA: (QuadraticDiscriminantAnalysis)",
            "XGBC-XGBClassifier: (XGBClassifier)",
            # "XGBC2-XGBClassifier: (XGBClassifier)",
            # "XGBC3-XGBClassifier: (XGBClassifier)",
            # "XGBC4-XGBClassifier: (XGBClassifier)",
        ]

        classifiers = [
            # KNeighborsClassifier(2),
            # SVC(kernel="linear", C=0.025),
            # SVC(cache_size=1,gamma='scale', C=1),
            # SVC(cache_size=1,gamma='scale', C=0.15),
            # SVC(cache_size=1,shrinking=True,gamma='scale', C=0.025, kernel='poly'),
            # SVC(cache_size=1,probability=True,shrinking=True,gamma='scale', C=0.025, kernel='rbf'),
            # SVC(cache_size=1,gamma='scale', C=0.025, kernel='sigmoid'),
            # SVC(cache_size=1,gamma='scale', C=0.025, kernel='precomputed'),
            # GaussianProcessClassifier(1.0 * RBF(1.0)),
            # DecisionTreeClassifier(max_depth=20),
            # RandomForestClassifier(
            #     max_depth=20, n_estimators=1000
            # ),
            # RandomForestClassifier(
            #     max_depth=20, n_estimators=1000, max_features="log2"
            # ),
            # RandomForestClassifier(
            #     max_depth=20, n_estimators=400, max_features="sqrt", criterion="entropy"
            # ),
            # MLPClassifier(alpha=1, max_iter=1000),
            # MLPClassifier(alpha=0.01, hidden_layer_sizes=(100, 100)),
            # AdaBoostClassifier(),
            # AdaBoostClassifier(learning_rate=0.01),
            AdaBoostClassifier(learning_rate=0.0001), # 81
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis(),
            # XGBClassifier(
            #     # objective='binary:logistic',
            #     # seed=0,
            #     # random_state=1,
            #     # learning_rate=0.01,
            #     # n_estimators=100,
            #     # max_features="sqrt",
            #     # colsample_bytree=1,
            #     # subsample=1,
            #     # objective='binary:logitraw',
            #     # objective='binary:hinge',
            #     # eval_metric="error",
            #     objective='binary:logistic',
            #     scale_pos_weight=6,
            #     learning_rate=4,
            #     n_estimators=500,
            #     reg_alpha=0.9,
            #     max_depth=40,
            #     gamma=0.1
            # ), # 80 -> 81 sin vars

            # """"
            # {
            #   'colsample_bytree': 0.8609113313025927,
            #   'gamma': 1.720202686427517,
            #   'learning_rate': 4.3,
            #   'max_depth': 31.0,
            #   'min_child_weight': 20.0,
            #   'n_estimators': 763.0,
            #   'reg_alpha': 0.46,
            #   'reg_lambda': 0.2898655418771495,
            #   'scale_pos_weight': 6.9}
            # """
            # XGBClassifier(
            #     objective='binary:logistic',
            #     scale_pos_weight=6.9,
            #     learning_rate=4.3,
            #     n_estimators=763,
            #     reg_alpha=0.46,
            #     max_depth=31,
            #     gamma=1.720
            # ),

            # XGBClassifier(
            #     objective='binary:logistic',
            #     scale_pos_weight=6.9,
            #     learning_rate=0.001,
            #     n_estimators=15,
            #     reg_alpha=0.46,
            #     max_depth=3,
            #     gamma=1.720
            # ), #-> mejor
            #{
            # 'colsample_bytree': 0.503622957973121, 
            # 'gamma': 3.2928411977775074, 
            # 'learning_rate': 1.8800000000000001, 
            # 'max_depth': 46.0, 
            # 'min_child_weight': 2.0, 
            # 'n_estimators': 139.0, 
            # 'reg_alpha': 0.47000000000000003, 
            # 'reg_lambda': 0.6605267779087194, 
            # 'scale_pos_weight': 19.71
            # }

            # XGBClassifier(
            #     objective='binary:logistic',
            #     scale_pos_weight=3.42,
            #     learning_rate=1.88,
            #     n_estimators=139,
            #     reg_alpha=0.47,
            #     max_depth=46,
            #     gamma=3.29,
            #     min_child_weight=2,
            #     colsample_bytree=0.503,
            # ),

            # XGBClassifier(
            #     objective='binary:logistic',
            #     scale_pos_weight=18.24,
            #     learning_rate=7.8100000,
            #     n_estimators=214,
            #     reg_alpha=0.52,
            #     max_depth=53,
            #     gamma=8.79374,
            #     min_child_weight=13,
            #     colsample_bytree=0.77836,
            # ), # 75

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

        cv = RepeatedKFold(n_splits=6, n_repeats=50)
        cv = StratifiedKFold(n_splits=6)
        cv = StratifiedKFold(n_splits=10)

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

            accuracies.append(0)  # accuracy.mean())
            precision_s.append(0)  # precision.mean())
            recalls.append(0)  # recall.mean())
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
        ], [name.split('-')[0] for name in names]
