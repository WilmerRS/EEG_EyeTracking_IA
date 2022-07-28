# python
import os

import numpy as np

# domain
from src.Models.BaseModel import BaseModel
from src.Models.MachineLearning import MachineLearningCrossValidationModels
from src.Models.MachineLearning import MutualInformationProcess
from src.Models.MachineLearning import PermutationImportanceProcess
from src.Models.DeepLearning import DeepLearningCrossValidationModels
from src.Plot import BarPlotResult
from src.Plot import BarPlotPermutationFeatureImportanceResult


class EEGEyeTracking(
    BaseModel,
    MachineLearningCrossValidationModels,
    MutualInformationProcess,
    PermutationImportanceProcess,
    DeepLearningCrossValidationModels
):
    _EEG_EYE_TRACKING_CSV_PATH = os.path.join(
        'EEGEyeTracking',
        'eeg-eye-tracking-test.csv'
    )

    _dataframe = None
    _verbose = False

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._dataframe = self.get_dataframe_from_cvs_assets(
            self._EEG_EYE_TRACKING_CSV_PATH
        )
        # self.describe(self._dataframe)
        
        # Features utilities 
        # self._permutation_feature_importance_cross()
        # self._mutual_information_without_biometrics()

        # Machine Learnig
        # self._test_ml_models_without_biometrics()
        # self._test_ml_models_without_biometrics_and_sex()

        # Deep learning
        self._test_dl_models_without_biometrics_and_sex()

    def _mutual_information_without_biometrics(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes
        )

        scores = self._mutual_information(X, y)
        self._mutual_information_graph(scores)

    def _test_ml_models_without_biometrics(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_without_biometrics_and_sex(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes_and_sex
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_without_biometrics_one_hot_encoding(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _permutation_feature_importance_cross(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes_and_sex
        )
        classifiers, names = self._ml_models()
        result_cross = self._permutation_importance_cross(
            X, y, classifiers, names)
        plotter = BarPlotPermutationFeatureImportanceResult()
        plotter.plot(result_cross, self._attributes_without_likes())

    def _test_dl_models_without_biometrics_and_sex(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes_and_sex
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)

    def _all_attributes(self):
        return ['Teorica1 ', 'Teorica2 ', 'Gustos1 ', 'Gustos2', 'Gustos3',
                'Edad ', 'Sexo', 'Afirmacion1 ', 'Afirmacion2 ',
                'PreguntaTrabajoCiencia  ', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo']

    def _attributes_without_likes(self):
        return ['Teorica1 ', 'Teorica2 ',
                'Edad ', 'Sexo', 'Afirmacion1 ', 'Afirmacion2 ',
                'PreguntaTrabajoCiencia  ', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo']

    def _attributes_without_likes_and_sex(self):
        return ['Teorica1 ', 'Teorica2 ',
                'Edad ', 'Afirmacion1 ', 'Afirmacion2 ',
                'PreguntaTrabajoCiencia  ', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo']

    def _define_target_attribute(self):
        return 'TipoStem '
