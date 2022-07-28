# machine learning
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score


class RandomForestRegressorByMetricValidation:

    _metric = None

    def __init__(self, metric):
        self._metric = metric

    def validate(self, train_X, val_X, train_y, val_y, n_estimators=100, **kwargs):
        model = RandomForestRegressor(
            random_state=1, n_estimators=n_estimators)
        model.fit(train_X, train_y)
        predictions_val = model.predict(val_X)
        return self._metric(val_y, predictions_val.astype(int), **kwargs)

    def multiple_validate(self,  train_X, val_X, train_y, val_y, estimators):
        errors = []
        for estimator in estimators:
            mae = self.validate(train_X, val_X, train_y,
                                val_y, n_estimators=estimator)
            errors.append({
                f'{estimator}': mae
            })

        return errors

    def cross_validation_score(self, X, y, estimator=100, cv=5, scoring='neg_mean_absolute_error'):
        model = RandomForestRegressor(
            random_state=1, n_estimators=estimator)
        scores = -1 * cross_val_score(model, X, y,
                                    cv=cv, scoring=scoring)
        return scores, scores.mean()