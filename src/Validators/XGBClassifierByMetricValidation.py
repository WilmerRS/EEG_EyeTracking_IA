# machine learning
from distutils.log import error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


class XGBClassifierByMetricValidation:

    _metric = None

    def __init__(self, metric):
        self._metric = metric

    def validate(self, train_X, val_X, train_y, val_y, estimators, seed=123, learning_rate=0.1,  **kwargs):
        model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=estimators,
            seed=seed,
            learning_rate=learning_rate,
            random_state=1
        )
        model.fit(train_X, train_y)
        predictions_val = model.predict(val_X)
        return self._metric(val_y, predictions_val.astype(int), **kwargs)

    def multiple_validate(self, train_X, val_X, train_y, val_y,estimators, seed=123, learning_rate=0.1 ):
        errors = []
        for estimator in estimators:
            error = self.validate(train_X, val_X, train_y, val_y, estimators=estimator, seed=seed, learning_rate=learning_rate)
            errors.append({
                f'{estimator}': error
            })

        return errors

    def cross_validation_score(self, X, y, estimators, seed=123, learning_rate=0.1, cv=5, scoring='accuracy'):
        model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=estimators,
            seed=seed,
            learning_rate=learning_rate,
            random_state=1
        )
        scores = cross_val_score(model, X, y,
                                 cv=cv, scoring=scoring)
        return scores, scores.mean()
