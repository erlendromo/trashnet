from src.processing.feature_extractor import FeatureExtractor
from src.processing.preprocessor import Preprocessor


class Processor:
    def __init__(self, dataset=None, dataset_type=None, debug=False):
        if dataset is None:
            raise ValueError("dataset cannot be 'None'")

        self.dataset = dataset
        self.dataset_type = dataset_type if dataset_type is not None else "unknown"
        self.debug = debug

    def process(self):
        if self.debug:
            self._debug()

        features = []
        labels = []

        for image_path, label in self.dataset:
            preprocessor = Preprocessor(image_path=image_path)
            preprocessed_image = preprocessor.process()

            feature_extractor = FeatureExtractor(image=preprocessed_image)

            image_features = feature_extractor.extract()

            features.append(image_features)
            labels.append(label)

        self.features = features
        self.labels = labels

        return self.features, self.labels

    def _debug(self):
        print(f"Processing images in {self.dataset_type} set")
