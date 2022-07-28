# python
import os

import numpy as np

# domain
from src.Models.MachineLearning import BaseModel
from src.Models.MachineLearning import MachineLearningCrossValidationModels
from src.Models.MachineLearning import MutualInformationProcess
from src.Models.MachineLearning import PermutationImportanceProcess


class StarbucksCustomerSurvey(
    BaseModel,
    MachineLearningCrossValidationModels,
    MutualInformationProcess,
    PermutationImportanceProcess
):
    _STARBUCKS_CUSTOMER_SURVEY_CSV_PATH = os.path.join(
        'StarbucksCustomerSurvey',
        'starbucks-satisfactory-survey.csv'
    )

    _dataframe = None
    _verbose = False

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._mutual_information_without_biometrics()

    def _mutual_information_without_biometrics(self):
        self._dataframe = self.get_dataframe_from_cvs_assets(
            self._STARBUCKS_CUSTOMER_SURVEY_CSV_PATH
        )

        # if self._verbose:
        #     self.describe(self._starbucks_dataframe)

        X, y, X_train, X_valid, y_train, y_valid = self.split_dataset_by_attributes()

        # scores = self._mutual_information(X, y)
        # self._mutual_information_graph(scores)

        self._permutation_importance_cross(X, y)

        # self._validate_classifier_models(
        #     X, y,
        #     X_train, X_valid, y_train, y_valid
        # )

    def _all_attributes(self):
        return ['gender', 'age', 'status', 'income', 'visitNo', 'method',
                'timeSpend', 'location', 'membershipCard', 'itemPurchaseCoffee',
                'itempurchaseCold', 'itemPurchasePastries', 'itemPurchaseJuices',
                'itemPurchaseSandwiches', 'itemPurchaseOthers', 'spendPurchase',
                'productRate', 'priceRate', 'promoRate', 'ambianceRate', 'wifiRate',
                'serviceRate', 'chooseRate', 'promoMethodApp', 'promoMethodSoc',
                'promoMethodEmail', 'promoMethodDeal', 'promoMethodFriend',
                'promoMethodDisplay', 'promoMethodBillboard', 'promoMethodOthers']
        # return ['spendPurchase', 'priceRate', 'status', 'age', 'chooseRate', 'wifiRate', 'ambianceRate', 'productRate', 'method', 'promoRate', 'Id']

    def _define_target_attribute(self):
        return 'loyal'
