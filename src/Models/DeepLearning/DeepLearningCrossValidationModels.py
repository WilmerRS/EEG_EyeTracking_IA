import abc

# deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from scikeras.wrappers import KerasClassifier

# machine learning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DeepLearningCrossValidationModels(metaclass=abc.ABCMeta):

    def _one_node_binary_crossentropy(
        self,
        input_shape,
        layers,
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
        drop_out_percentage=0.2
    ):
        model = Sequential()
        model.add(Dense(layers[0], input_shape=(
            input_shape,), activation='relu'))

        for layer in range(1, len(layers)):
            model.add(
                Dropout(drop_out_percentage)
            )
            model.add(
                Dense(
                    layers[layer],
                    activation='relu',
                    kernel_constraint=MaxNorm(3),
                )
            )

        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss=loss,
            optimizer='adam',
            metrics=metrics
        )
        return model

    def _dl_models(self, columns):
        names = [
            f'DL[{columns}]',
            f'DL[{columns} {columns}]',
            f'DL[{columns} {columns} {columns}]',
            f'DL[{columns} {columns} {columns} {columns}]',
            f'DL[{columns} 100]',
            f'DL[{columns} 100]',
            f'DL[{columns} 100 100]',
            f'DL[{columns} 1000]',
            f'DL[{columns} 20 30 20 30 20]',
            f'DL[{columns} 30 30 30]',
            f'DL[{columns} 50 70 50]',
            f'DL[{columns} 50 70 50 {columns}]',
            f'DL[{columns} 40 100 100 40 {columns}]',
            f'DL[{columns} 60]',
            f'DL[{columns} 60 60]',
        ]
        classifiers = [
            self._one_node_binary_crossentropy(
                layers=[columns], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, columns], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, columns, columns], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, columns, columns, columns], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 100], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 100, 100], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 1000], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 100, 100, 100], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 20, 30, 20, 30, 20], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 30, 30, 30], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 50, 70, 50], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 100, 100, 100, columns], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 60], input_shape=columns),
            self._one_node_binary_crossentropy(
                layers=[columns, 60, 60], input_shape=columns),
        ]

        return classifiers, names

    def _validate_dl_classifier_models(self, X, y):
        classifiers, names = self._dl_models(len(X.columns))
        cv = StratifiedKFold(n_splits=7)

        accuracies = []
        precision_s = []
        f1_s = []
        recalls = []

        index = 1
        for name, classifier in zip(names, classifiers):
            estimator = KerasClassifier(
                model=classifier,
                epochs=400,
                batch_size=-1,
                verbose=0,
                random_state=0,
            )

            accuracy = cross_val_score(
                estimator, X, y,
                cv=cv, scoring='accuracy'
            )
            precision = cross_val_score(
                estimator, X, y,
                cv=cv, scoring='precision'
            )
            recall = cross_val_score(
                estimator, X, y,
                cv=cv, scoring='recall'
            )
            f1 = cross_val_score(
                estimator, X, y,
                cv=cv, scoring='f1'
            )
            print(f'\n ** {index}.{name} ** \n')
            print(
                f"accuracy:   {accuracy.mean():.3f} (std) {accuracy.std():.3f}")
            print(
                f"precision:        {precision.mean():.3f} (std) {precision.std():.3f}")
            print(f"recall:     {recall.mean():.3f} (std) {recall.std():.3f}")
            print(f"f1:         {f1.mean():.3f} (std) {f1.std():.3f}")

            accuracies.append(accuracy.mean())
            precision_s.append(precision.mean())
            recalls.append(recall.mean())
            f1_s.append(f1.mean())

            index += 1

        return [
            accuracies,
            precision_s,
            recalls,
            f1_s,
        ], names

    def _validate_one_dl_classifier_models(self, X, y, xx, yy):
        estimator = KerasClassifier(
            model=self._one_node_binary_crossentropy(
                layers=[len(X.columns), 100, 100, 100, len(X.columns)], input_shape=len(X.columns)),
            epochs=400,
            batch_size=-1,
            verbose=0,
            random_state=0,
        )
        estimator.fit(X, y)

        v = estimator.predict(
            [xx.values]
        )
        print(f'stem: {v} real: {yy}')
