import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ("svm", "knn", "rf")


class Classifier:
    def __init__(self, model="svm", debug=False):
        self.model_type = model if model in MODELS else "svm"
        self.debug = debug

        if self.model_type == "svm":
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced"),
                    ),
                ]
            )

        elif self.model_type == "knn":
            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
                ]
            )

        elif self.model_type == "rf":
            self.model = Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200, class_weight="balanced_subsample"
                        ),
                    ),
                ]
            )

    def classify(
        self, training_features, training_labels, testing_features, testing_labels
    ):
        if self.debug:
            self._debug()

        self._train(training_features, training_labels)
        self._evaluate(testing_features, testing_labels)

    def _train(self, features, labels):
        features = np.array(features)
        labels = np.array(labels)

        self.model.fit(features, labels)

    def _predict(self, features):
        features = np.array(features)
        return self.model.predict(features)

    def _evaluate(self, features, labels):
        predictions = self._predict(features)
        print(classification_report(labels, predictions, zero_division=0))
        print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")

    def _debug(self):
        print(f"Training and classifying using {self.model_type}.")
