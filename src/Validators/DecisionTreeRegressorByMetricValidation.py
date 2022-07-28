# machine learning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

class DecisionTreeRegressorByMetricValidation:
    
    _metric = None
    def __init__(self, metric):
        self._metric = metric

    def validate(self, max_leaf_nodes, train_X, val_X, train_y, val_y, **kwargs):
        model = DecisionTreeRegressor(
            max_leaf_nodes=max_leaf_nodes, random_state=1)
        model.fit(train_X, train_y)
        predictions_val = model.predict(val_X)
        return self._metric(val_y, predictions_val.astype(int), **kwargs)

    def multiple_validate(self, leafs,  train_X, val_X, train_y, val_y, ):
        errors = []
        for max_leaf_nodes in leafs:
            mae = self.validate(max_leaf_nodes, train_X, val_X, train_y, val_y)
            errors.append({
                f'{max_leaf_nodes}': mae
            })

        return errors
    
    def cross_validation_score(self, X, y, max_leaf_nodes=100, cv=5, scoring='accuracy'):
        model = DecisionTreeRegressor(
            random_state=1, max_leaf_nodes=max_leaf_nodes)
        scores = cross_val_score(model, X, y,
                                    cv=cv, scoring=scoring)
        return scores, scores.mean()
