import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.constants import CLASSIFIERS


class Classifier:
    def __init__(self, classifier_type="svm", debug=False):
        self.classifier_type = (
            classifier_type if classifier_type in CLASSIFIERS else "svm"
        )
        self.debug = debug

        if self.classifier_type == "svm":
            self.classifier = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced"),
                    ),
                ]
            )

        elif self.classifier_type == "knn":
            self.classifier = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
                ]
            )

        elif self.classifier_type == "rf":
            self.classifier = Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200, class_weight="balanced_subsample"
                        ),
                    ),
                ]
            )

    # TODO Add validation set optimization
    def train(self, train_features, train_labels, val_features, val_labels):
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        val_features = np.array(val_features)
        val_labels = np.array(val_labels)

        self.classifier.fit(train_features, train_labels)

    def save(self, path="model.joblib"):
        dump(self.classifier, path)
        print(f"Saved model as {path}")

    def load(self, path="model.joblib"):
        self.classifier = load(path)
        print(f"Loaded model from {path}")

    def predict(self, features):
        features = np.array(features)
        return self.classifier.predict(features)

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        print(classification_report(labels, predictions, zero_division=0))
        print(f"Accuracy: {(accuracy_score(labels, predictions) * 100):.2f}")

    def _debug(self):
        print(f"Training and classifying using {self.classifier_type}.")
