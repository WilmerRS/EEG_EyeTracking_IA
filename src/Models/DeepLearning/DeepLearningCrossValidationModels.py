import abc

# deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from scikeras.wrappers import KerasClassifier

# machine learning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DeepLearningCrossValidationModels(metaclass=abc.ABCMeta):

    def _one_node_binary_crossentropy(self, layers=[11], input_shape=11, loss='binary_crossentropy', metrics=['binary_accuracy']):
        model = Sequential()
        model.add(Dense(layers[0], input_shape=(
            input_shape,), activation='relu'))

        for layer in range(1, len(layers)):
            model.add(
                Dense(
                    layers[layer],
                    activation='relu'
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
            f'DL[{columns} 10 100]',
            f'DL[{columns} 10 100 100]',
            f'DL[{columns} 20 30 20]',
            f'DL[{columns} 30 30 30]',
            f'DL[{columns} 50 70 50]',
            f'DL[{columns} 50 70 50 {columns}]',
            f'DL[{columns} 60]',
            f'DL[{columns} 60 60]',
        ]
        classifiers = [
            self._one_node_binary_crossentropy(),
            self._one_node_binary_crossentropy(layers=[columns]),
            self._one_node_binary_crossentropy(layers=[columns, columns]),
            self._one_node_binary_crossentropy(
                layers=[columns, columns, columns]),
            self._one_node_binary_crossentropy(
                layers=[columns, columns, columns, columns]),
            self._one_node_binary_crossentropy(layers=[columns, 100]),
            self._one_node_binary_crossentropy(layers=[columns, 100, 100]),
            self._one_node_binary_crossentropy(
                layers=[columns, 100, 100, 100]),
            self._one_node_binary_crossentropy(layers=[columns, 20, 30, 20]),
            self._one_node_binary_crossentropy(layers=[columns, 30, 30, 30]),
            self._one_node_binary_crossentropy(layers=[columns, 50, 70, 50]),
            self._one_node_binary_crossentropy(
                layers=[columns, 50, 70, 50, columns]),
            self._one_node_binary_crossentropy(layers=[columns, 60]),
            self._one_node_binary_crossentropy(layers=[columns, 60, 60]),
        ]

        return classifiers, names

    def _validate_dl_classifier_models(self, X, y):
        classifiers, names = self._dl_models(len(X.columns))
        cv = StratifiedKFold(n_splits=10, shuffle=False)

        accuracies = []
        precision_s = []
        f1_s = []
        recalls = []

        index = 1
        for name, classifier in zip(names, classifiers):
            estimator = KerasClassifier(
                model=classifier,
                epochs=100,
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
            f1_s.append(f1.mean())
            recalls.append( recall.mean())

            index += 1

        return [
            accuracies,
            precision_s,
            recalls,
            f1_s,
        ], names
