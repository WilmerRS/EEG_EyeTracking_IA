# python
import os

import numpy as np

# domain
from src.Models.BaseModel import BaseModel
from src.Models.MachineLearning import MachineLearningCrossValidationModels
from src.Models.MachineLearning import MutualInformationProcess
from src.Models.MachineLearning import PermutationImportanceProcess
from src.Models.DeepLearning import DeepLearningCrossValidationModels
from src.Models.MachineLearning.XGBOptimization import XGBOptimization
from src.Plot import BarPlotResult
from src.Plot import BarPlotPermutationFeatureImportanceResult


class EEGEyeTracking(
    BaseModel,
    MachineLearningCrossValidationModels,
    MutualInformationProcess,
    PermutationImportanceProcess,
    DeepLearningCrossValidationModels
):
    _EEG_EYE_TRACKING_CSV_PATH_MINI = os.path.join(
        'EEGEyeTracking',
        'eeg-eye-tracking-test.csv'
    )

    _EEG_EYE_TRACKING_CSV_PATH_FULL = os.path.join(
        'EEGEyeTracking',
        'eeg-eye-tracking-test-full.csv'
    )

    _EEG_EYE_TRACKING_CSV_PATH_FULL_WITH_VARS = os.path.join(
        'EEGEyeTracking',
        'eeg-eye-tracking-test-full-with-vars.csv'
    )

    # _EEG_EYE_TRACKING_CSV_PATH = _EEG_EYE_TRACKING_CSV_PATH_FULL
    _EEG_EYE_TRACKING_CSV_PATH = _EEG_EYE_TRACKING_CSV_PATH_FULL_WITH_VARS

    _dataframe = None
    _verbose = False

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._dataframe = self.get_dataframe_from_cvs_assets(
            self._EEG_EYE_TRACKING_CSV_PATH
        )
        # mini
        # self.describe(self._dataframe, self._define_target_attribute_full())

        # Features utilities
        # self._mutual_information_without_biometrics()
        # self._permutation_feature_importance_cross()

        # Machine Learnig
        # self._test_ml_models_without_biometrics()
        # self._test_ml_models_without_biometrics_and_sex()

        # Deep learning
        # self._test_dl_models_without_biometrics_and_sex()

        # full
        # self.describe(self._dataframe, self._define_target_attribute_full())

        # Features utilities
        # self._mutual_information_full()
        # self._permutation_feature_importance_cross_full()

        # Machine Learnig
        # self._test_ml_models_full()
        # self._test_ml_models_minus_svm_full()
        # self._test_ml_models_without_eeg_full()
        # self._test_ml_models_without_eeg_and_tft_full()

        # Deep learning
        # self._test_dl_models_full()
        # self._test_dl_models_without_svm_full()
        # self._test_dl_models_without_eeg_full()
        # self._test_dl_models_without_eeg_and_tft_full()

        # Features utilities all
        # self._mutual_information_full_with_vars()
        # self._permutation_feature_importance_cross_full_with_vars()

        # self._test_ml_models_full_with_vars()
        self._test_dl_models_without_eeg_and_tft_full_with_vars()
        # self._xgb_optimization()

    def _mutual_information_without_biometrics(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes(),
            self._define_target_attribute()
        )

        scores = self._mutual_information(X, y)
        self._mutual_information_graph(scores)

    def _test_ml_models_without_biometrics(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes(),
            self._define_target_attribute()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_without_biometrics_and_sex(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes_and_sex(),
            self._define_target_attribute()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_without_biometrics_one_hot_encoding(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes(),
            self._define_target_attribute()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _permutation_feature_importance_cross(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes_and_sex(),
            self._define_target_attribute()
        )
        classifiers, names = self._ml_models()
        result_cross = self._permutation_importance_cross(
            X, y, classifiers, names)
        plotter = BarPlotPermutationFeatureImportanceResult()
        plotter.plot(result_cross, self._attributes_without_likes())

    def _test_dl_models_without_biometrics_and_sex(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._attributes_without_likes_and_sex(),
            self._define_target_attribute()
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)
    # mini

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

    # --------------------------------------------------------------------------------------------
    # full ---------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------
    def _define_target_attribute_full(self):
        return 'TipoStem'

    def _all_attributes_full(self):
        return ['Teorica1', 'Teorica2 ', 'Gustos1', 'Gustos2',
                'Gustos3', 'Edad', 'Sexo', 'Afirmacion1', 'Afirmacion2',
                'PreguntaTrabajoCiencia', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo',
                'PTeorica1', 'PTeorica2', 'PGustos1', 'PGustos2',
                'PGustos3', 'NfOpt1_P1', 'NfOpt2_P1', 'NfOpt1_P2', 'NfOpt2_P2',
                'NfOpt1_P3', 'NfOpt2_P3', 'NfOpt1_P4', 'NfOpt2_P4', 'NfOpt1_P5',
                'NfOpt2_P5', 'TtfOpt1_P1', 'TtfOpt2_P1', 'TtfOpt1_P2', 'TtfOpt2_P2',
                'TtfOpt1_P3', 'TtfOpt2_P3', 'TtfOpt1_P4', 'TtfOpt2_P4', 'TtfOpt1_P5',
                'TtfOpt2_P5'
                ]

    def _attributes_without_likes_full(self):
        return ['Teorica1', 'Teorica2 ', 'Edad', 'Sexo', 'Afirmacion1', 'Afirmacion2',
                'PreguntaTrabajoCiencia', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo',
                'PTeorica1', 'PTeorica2', 'PGustos1', 'PGustos2',
                'PGustos3', 'NfOpt1_P1', 'NfOpt2_P1', 'NfOpt1_P2', 'NfOpt2_P2',
                'NfOpt1_P3', 'NfOpt2_P3', 'NfOpt1_P4', 'NfOpt2_P4', 'NfOpt1_P5',
                'NfOpt2_P5', 'TtfOpt1_P1', 'TtfOpt2_P1', 'TtfOpt1_P2', 'TtfOpt2_P2',
                'TtfOpt1_P3', 'TtfOpt2_P3', 'TtfOpt1_P4', 'TtfOpt2_P4', 'TtfOpt1_P5',
                'TtfOpt2_P5'
                ]

    def _attributes_without_minus_svg_full(self):
        return ['Teorica1', 'Edad',
                'PreguntaTrabajoCiencia',
                'PTeorica1', 'PTeorica2', 'PGustos1', 'PGustos2',
                'PGustos3', 'NfOpt1_P1', 'NfOpt2_P1', 'NfOpt1_P2', 'NfOpt2_P2',
                'NfOpt1_P3', 'NfOpt2_P3', 'NfOpt1_P4', 'NfOpt2_P4', 'NfOpt1_P5',
                'NfOpt2_P5', 'TtfOpt1_P5',
                ]

    def _attributes_without_eeg_full(self):
        return ['Teorica1', 'Teorica2 ', 'Edad', 'Afirmacion1', 'Afirmacion2',
                'PreguntaTrabajoCiencia', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo',
                'NfOpt1_P1', 'NfOpt2_P1', 'NfOpt1_P2', 'NfOpt2_P2',
                'NfOpt1_P3', 'NfOpt2_P3', 'NfOpt1_P4', 'NfOpt2_P4', 'NfOpt1_P5',
                'NfOpt2_P5', 'TtfOpt1_P1', 'TtfOpt2_P1', 'TtfOpt1_P2', 'TtfOpt2_P2',
                'TtfOpt1_P3', 'TtfOpt2_P3', 'TtfOpt1_P4', 'TtfOpt2_P4', 'TtfOpt1_P5',
                'TtfOpt2_P5'
                ]

    def _attributes_without_eeg_and_tft_full(self):
        return ['Teorica1', 'Teorica2 ', 'Edad', 'Afirmacion1', 'Afirmacion2',
                'PreguntaTrabajoCiencia', 'PeguntaConstruccion', 'PreguntaMultidipli',
                'Asignaturas', 'ConocimientosProblemas', 'AnaliticoCreativo',
                'NfOpt1_P1', 'NfOpt2_P1', 'NfOpt1_P2', 'NfOpt2_P2',
                'NfOpt1_P3', 'NfOpt2_P3', 'NfOpt1_P4', 'NfOpt2_P4', 'NfOpt1_P5',
                'NfOpt2_P5'
                ]

    def _mutual_information_full(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._all_attributes_full(),
            self._define_target_attribute_full()
        )

        scores = self._mutual_information(X, y)
        self._mutual_information_graph(scores)

    def _permutation_feature_importance_cross_full(self):
        print('Attributes count:')
        attrs = self._attributes_without_likes_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )
        classifiers, names = self._ml_models()
        result_cross = self._permutation_importance_cross(
            X, y, classifiers, names)
        plotter = BarPlotPermutationFeatureImportanceResult()
        plotter.plot(result_cross, attrs)

    def _test_ml_models_full(self):
        print('\n\n_test_ml_models_full - Attributes count:\n\n')
        attrs = self._attributes_without_likes_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_without_eeg_full(self):
        print('\n\n_test_ml_models_full - Attributes count:\n\n')
        attrs = self._attributes_without_eeg_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_without_eeg_and_tft_full(self):
        print('\n\n_test_ml_models_full - Attributes count:\n\n')
        attrs = self._attributes_without_eeg_and_tft_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_ml_models_minus_svm_full(self):
        print('\n\n_test_ml_models_minus_svm_full - Attributes count:\n\n')
        attrs = self._attributes_without_minus_svg_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_dl_models_without_svm_full(self):
        print('\n\n_test_dl_models_without_svm_full - Attributes count:\n\n')
        attrs = self._attributes_without_minus_svg_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_dl_models_full(self):
        print('\n\_test_dl_models_full - Attributes count:\n\n')
        attrs = self._attributes_without_likes_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_dl_models_without_eeg_full(self):
        print('\n\_test_dl_models_full - Attributes count:\n\n')
        attrs = self._attributes_without_eeg_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_dl_models_without_eeg_and_tft_full(self):
        print('\n\_test_dl_models_full - Attributes count:\n\n')
        attrs = self._attributes_without_eeg_and_tft_full()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)

    def _all_attributes_full_with_vars(self):
        return [
            'Teorica1', 'Teorica2 ',
            # 'Gustos1', 'Gustos2', 'Gustos3',
            # 'Edad',
            # 'Sexo',
            'Afirmacion1', 'Afirmacion2',
            'PreguntaTrabajoCiencia',
            'PeguntaConstruccion', 'PreguntaMultidipli',
            'Asignaturas', # -> mejora si la quito
            'ConocimientosProblemas', 'AnaliticoCreativo',
            'PTeorica1', 
            'PTeorica2', 
            'PGustos1', 'PGustos2',
            'PGustos3',
            'NfOpt1_P1', 'NfOpt2_P1', 'NfOpt1_P2', 'NfOpt2_P2',
            'NfOpt1_P3', 'NfOpt2_P3', 'NfOpt1_P4', 'NfOpt2_P4', 'NfOpt1_P5',
            'NfOpt2_P5',
            'TtfOpt1_P1', 'TtfOpt2_P1', 'TtfOpt1_P2', 'TtfOpt2_P2',
            'TtfOpt1_P3', 'TtfOpt2_P3', 'TtfOpt1_P4', 'TtfOpt2_P4', 'TtfOpt1_P5',
            'TtfOpt2_P5',
            'P1_NAM_STEM', 'P2_NAM_STEM', 'P3_NAM_STEM',
            'P4_NAM_STEM', 'P5_NAM_STEM'
        ]

    def _mutual_information_full_with_vars(self):
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            self._all_attributes_full_with_vars(),
            self._define_target_attribute_full()
        )

        scores = self._mutual_information(X, y)
        self._mutual_information_graph(scores)

    def _permutation_feature_importance_cross_full_with_vars(self):
        print('Attributes count:')
        attrs = self._all_attributes_full_with_vars()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )
        classifiers, names = self._ml_models()
        result_cross = self._permutation_importance_cross(
            X, y, classifiers, names)
        plotter = BarPlotPermutationFeatureImportanceResult()
        plotter.plot(result_cross, attrs)

    def _test_ml_models_full_with_vars(self):
        print('\n\_test_ml_models_full_with_vars - Attributes count:\n\n')
        attrs = self._all_attributes_full_with_vars()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_ml_classifier_models(
            X, y
        )

        bar = BarPlotResult()
        bar.plot(results, names)

    def _test_dl_models_without_eeg_and_tft_full_with_vars(self):
        print(
            '\n\_test_dl_models_without_eeg_and_tft_full_with_vars - Attributes count:\n\n')
        attrs = self._all_attributes_full_with_vars()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )

        results, names = self._validate_dl_classifier_models(
            X, y
        )
        bar = BarPlotResult()
        bar.plot(results, names)

    def _xgb_optimization(self):
        print('\n\_xgb_optimization - Attributes count:\n\n')
        attrs = self._all_attributes_full_with_vars()
        print(len(attrs))
        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes(
            attrs,
            self._define_target_attribute_full()
        )
        hp = XGBOptimization.best_hyper_params(X, y)
        print(hp)
