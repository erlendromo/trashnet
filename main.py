from os import getenv
from time import time

from cv2 import imread
from dotenv import load_dotenv

from src.classifier import Classifier
from src.dataset import Dataset
from src.feature_extractor import FeatureExtractor
from src.preprocessor import Preprocessor

# Load .env file
load_dotenv()


def debug_mode():
    debug = getenv("DEBUG", False)
    debug = True if debug == "True" else False
    if debug:
        print("Debug mode activated...\n")

    return debug


def elapsed(start):
    end = time()
    elapsed = end - start

    minutes = elapsed / 60 if elapsed >= 60 else 0
    seconds = elapsed % 60 if elapsed >= 60 else elapsed if elapsed < 60 else 0

    print(f"\nProgram completed in {minutes:.0f}min and {seconds:.0f}sec")


def extract_features(dataset, debug):
    features = []
    labels = []
    for path, label in dataset:
        image = imread(path)
        if image is None:
            continue

        processed_image = Preprocessor(image=image, debug=debug).process()
        image_features = FeatureExtractor(processed_image, debug=debug).extract()

        features.append(image_features)
        labels.append(label)

    return features, labels


def main():
    start = time()
    debug = debug_mode()

    dataset = Dataset(debug=debug)
    training_set, validation_set, testing_set = dataset.load_and_split()

    print(
        "Dataset size: ", (len(training_set) + len(validation_set) + len(testing_set))
    )

    training_features, training_labels = extract_features(
        dataset=training_set, debug=debug
    )
    validation_features, validation_labels = extract_features(
        dataset=validation_set, debug=debug
    )
    testing_features, testing_labels = extract_features(
        dataset=testing_set, debug=debug
    )

    classifier = Classifier(model="svm", debug=debug)
    classifier.classify(
        training_features,
        training_labels,
        testing_features,
        testing_labels,
    )

    elapsed(start)


if __name__ == "__main__":
    main()
