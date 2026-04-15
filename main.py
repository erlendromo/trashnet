from src.classifier import Classifier
from src.feature_extractor import FeatureExtractor
from src.preprocessor import Preprocessor

"""
Trashnet project entry point
"""


def main():
    preprocessor = Preprocessor()
    preprocessor.print()


if __name__ == "__main__":
    main()
