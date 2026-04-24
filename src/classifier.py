from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class Classifier:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test


    def classify(self):
        best_model, best_params = self._train()

        test_accuracy = self._evaluate(
            self.best_model,
            self.X_test,
            self.y_test,
            dataset_name="Test Set"
        )


    def _train(self):
        """
        Train an RBF SVM using the training set,
        optimize hyperparameters using the validation set,
        and evaluate final performance on the test set.

        Returns
        -------
        best_model : trained sklearn model
        best_params : dict
        test_accuracy : float
        """

        # -------------------------------------------------
        # Hyperparameter search space
        # -------------------------------------------------

        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]

        best_model = None
        best_params = None
        best_val_accuracy = 0

        # -------------------------------------------------
        # Validation-based hyperparameter tuning
        # -------------------------------------------------

        print("Starting validation-based hyperparameter search...\n")

        for C in C_values:
            for gamma in gamma_values:
                model = SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    random_state=42
                )

                # Train on training set
                model.fit(self.X_train, self.y_train)

                # Evaluate on validation set
                val_predictions = model.predict(self.X_val)
                val_accuracy = accuracy_score(
                    self.y_val,
                    val_predictions
                )

                print(
                    f"C={C:<6} gamma={gamma:<6} "
                    f"Validation Accuracy={val_accuracy:.4f}"
                )

                # Keep best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = model
                    best_params = {
                        "C": C,
                        "gamma": gamma
                    }

        print("\nBest Hyperparameters Found:")
        print(best_params)
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

        self.best_model = best_model
        self.best_params = best_params

        return best_model, best_params


    def _evaluate(self, model, X, y, dataset_name="Dataset"):
        """
        Evaluate a trained model on a given dataset.

        Parameters
        ----------
        model : sklearn model
            Trained classifier

        X : np.ndarray
            Feature vectors

        y : np.ndarray
            Ground truth labels

        dataset_name : str
            Name used for printing results

        Returns
        -------
        accuracy : float
        """

        y_predictions = model.predict(X)
        classes = np.unique(y)

        accuracy = accuracy_score(y, y_predictions)
        cr = classification_report(y, y_predictions, zero_division=0)
        cm = confusion_matrix(y, y_predictions)

        print(f"\n{dataset_name} Evaluation")
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(cr)

        print("\nConfusion Matrix:")
        print(cm)

        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(range(len(classes)), classes, rotation=90)
        plt.yticks(range(len(classes)), classes)
        plt.colorbar()
        plt.show()

        return accuracy
