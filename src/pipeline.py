from src.classifier import Classifier
from src.dataset_loader import DatasetLoader
from src.processor import Processor


class Pipeline:
    def __init__(
        self,
        dataset_path,
        save_model,
        load_model,
        classifier_type,
        debug=False,
    ):
        self.classifier = None
        self.classifier_type = classifier_type
        self.save_model = save_model
        self.debug = debug

        dataset = DatasetLoader(dataset_path=dataset_path, debug=self.debug)
        train_tuple, val_tuple, test_tuple = dataset.load_and_split()

        self.test_features, self.test_labels = self._extract_features_and_labels(
            test_tuple, "testing"
        )

        if load_model is not None:
            self.classifier = Classifier(debug=self.debug)
            self.classifier.load(load_model)
        else:
            self.train_features, self.train_labels = self._extract_features_and_labels(
                train_tuple, "training"
            )

            self.val_features, self.val_labels = self._extract_features_and_labels(
                val_tuple, "validation"
            )

    def execute(self):
        if self.classifier is None:
            self.classifier = Classifier(
                classifier_type=self.classifier_type, debug=self.debug
            )
            self.classifier.train(
                self.train_features,
                self.train_labels,
                self.val_features,
                self.val_labels,
            )

        self.classifier.evaluate(self.test_features, self.test_labels)

        if self.save_model is not None:
            self.classifier.save(self.save_model)

    def _extract_features_and_labels(self, dataset, dataset_type):
        processor = Processor(
            dataset=dataset, dataset_type=dataset_type, debug=self.debug
        )
        features, labels = processor.process()

        return features, labels
