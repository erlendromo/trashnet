from src.classifier import Classifier
from src.dataset_loader import DatasetLoader
from src.processor import Processor
import src.utils.feature_plotter as fp
import src.utils.scaler as sc


class Pipeline:
    def __init__(self, config=None):
        if config is None:
            raise ValueError("config cannot be 'None'")

        self.config = config


    def execute(self):
        dataset = DatasetLoader(dataset_path=self.config.dataset_path, test_size=self.config.test_size, val_size=self.config.val_size)
        train_tuple, val_tuple, test_tuple = dataset.load_and_split()

        print(f"Applying the following preprocessing configuration:\nNoise: {self.config.noise}\nSegmentation: {self.config.segment}\n")
        print(f"Extracting the following features:\nLBP: {self.config.lbp}\nGLCM: {self.config.glcm}\nHSV: {self.config.hsv}\nGabor: {self.config.gabor}\nSIFT: {self.config.sift}\nHu: {self.config.hu}\nHOG: {self.config.hog}\nSuperpixel: {self.config.superpixel}\n")
        print(f"-----\nThis may take a couple of minutes...\n-----\n")

        train_features, y_train = self._extract_features_and_labels(train_tuple)
        val_features, y_val = self._extract_features_and_labels(val_tuple)
        test_features, y_test = self._extract_features_and_labels(test_tuple)

        X_train, X_val, X_test, _ = sc.scale_feature_vectors(train_features, val_features, test_features)

        if self.config.visualize:
            fp.plot_pca_feature_space(X=X_train, y=y_train)
            fp.plot_tsne_feature_space(X=X_train, y=y_train)
            fp.plot_umap_feature_space(X=X_train, y=y_train)

        classifier = Classifier(X_train, y_train, X_val, y_val, X_test, y_test)
        classifier.classify()


    def _extract_features_and_labels(self, dataset_tuple):
        processor = Processor(dataset_tuple=dataset_tuple, config=self.config)
        features, labels = processor.process()

        return features, labels
