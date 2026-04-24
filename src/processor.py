from src.processing.feature_extractor import FeatureExtractor
from src.processing.preprocessor import Preprocessor


class Processor:
    def __init__(self, dataset_tuple, config):
        if dataset_tuple is None:
            raise ValueError("dataset cannot be 'None'")

        if config is None:
            raise ValueError("config cannot be 'None'")

        self.dataset_tuple = dataset_tuple
        self.config = config


    def process(self):
        all_features = []
        all_labels = []

        for image_path, label in self.dataset_tuple:
            preprocessor = Preprocessor(
                image_path=image_path,
                noise=self.config.noise,
                segment=self.config.segment,
                visualize=self.config.visualize
            )

            preprocessed_image = preprocessor.process()

            feature_extractor = FeatureExtractor(
                image=preprocessed_image,
                lbp=self.config.lbp,
                glcm=self.config.glcm,
                hsv=self.config.hsv,
                gabor=self.config.gabor,
                sift=self.config.sift,
                hu=self.config.hu,
                hog=self.config.hog,
                superpixel=self.config.superpixel
            )

            features = feature_extractor.extract()

            all_features.append(features)
            all_labels.append(label)

        return all_features, all_labels
