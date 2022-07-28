import abc
import os

# machine learning
import pandas as pd
from sklearn.model_selection import train_test_split

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


class BaseModel(metaclass=abc.ABCMeta):

    def get_dataframe_from_cvs_assets(self, path: str) -> None:
        full_path = self._get_full_path(path)
        return pd.read_csv(full_path)

    @abc.abstractmethod
    def _mutual_information_without_biometrics(self) -> None:
        raise "[BaseModel] Method without implementation"

    def _get_full_path(self, path: str) -> str:
        root_path = get_root_path()
        return os.path.join(root_path, 'assets', path)

    def describe(self, dataframe):
        print("Head:")
        print(dataframe.head())

        print("Describe:")
        print(dataframe.describe())

        print("Columnas: " )
        print(dataframe.columns)

        print('Por STEM')
        by_STEM_class = dataframe.groupby([self._define_target_attribute()])
        print(by_STEM_class.count())  

        print('Por Sexo')
        by_sex_class = dataframe.groupby(['Sexo'])
        print(by_sex_class.count())  

    def split_dataset_by_attributes(self, define_attributes_to_learn):
        attributes_to_learn = define_attributes_to_learn()
        target = self._define_target_attribute()

        y = self._dataframe[target]
        X = self._dataframe[attributes_to_learn]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, random_state=1)
        return X, y, X_train, X_valid, y_train, y_valid

    def _all_attributes(self):
        print("[BaseModel] Method without implementation")

    def _define_target_attribute(self):
        print("[BaseModel] Method without implementation")
