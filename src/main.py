from src.StarbucksCustomerSurvey import StarbucksCustomerSurvey
from src.EEGEyeTracking import EEGEyeTracking

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class Main:
    def __init__(self) -> None:
        # self.starbucks_customer_survey_ml()
        self.eeg_eye_tracking_test_ml()

    def eeg_eye_tracking_test_ml(self):
        EEGEyeTracking(verbose=True)

    def starbucks_customer_survey_ml(self):
        StarbucksCustomerSurvey(verbose=True)
