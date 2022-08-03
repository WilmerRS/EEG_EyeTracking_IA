from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import cross_val_score

class XGBOptimization:

    def space():
        return {
            'max_depth': hp.quniform("max_depth", 1, 70, 1),
            'gamma': hp.uniform('gamma', 0, 10),
            'reg_alpha': hp.quniform('reg_alpha', 0, 1, 0.01),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.quniform('min_child_weight', 0, 20, 1),
            'n_estimators': hp.quniform('n_estimators', 20, 2000, 1),
            'learning_rate': hp.quniform('learning_rate', 0, 30, 0.01),
            'scale_pos_weight': hp.quniform('scale_pos_weight', 0, 20, 0.01),
            'seed': 0
        }

    def objective(space_1, X, y):
        cv = RepeatedKFold(n_splits=5, n_repeats=4, random_state=1)
        def val(space):
            print(space)
            clf = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=float(space['scale_pos_weight']),
                learning_rate=float(space['learning_rate']),
                n_estimators=int(space['n_estimators']),
                reg_alpha=int(space['reg_alpha']),
                max_depth=int(space['max_depth']),
                gamma=space['gamma'], 
                min_child_weight=int(space['min_child_weight']),
                colsample_bytree=int(space['colsample_bytree']),
                seed=0,
            )
            f1 = cross_val_score(
                clf, X, y,
                cv=cv, scoring='f1'
            )
            print(f'F1:   {f1.mean()}', )

            return {'loss': -f1.mean(), 'status': STATUS_OK}
        
        return val

    def best_hyper_params(X, y):
        trials = Trials()
        best_hyper_params = fmin(
            fn=XGBOptimization.objective(XGBOptimization.space(), X, y),
            space=XGBOptimization.space(),
            algo=tpe.suggest,
            max_evals=150,
            trials=trials
        )
        return best_hyper_params
