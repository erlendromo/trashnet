import os

from sklearn.model_selection import train_test_split

from src.utils.constants import ACCEPTED_IMAGE_FORMATS, ALL_LABELS, RECOMMENDED_LABELS


class DatasetLoader:
    def __init__(
        self,
        labels=RECOMMENDED_LABELS,
        dataset_path="./dataset",
        test_size=0.15,
        val_size=0.10,
        debug=False,
    ):
        self.dataset_path = dataset_path
        self.labels = labels
        self.test_size = max(0.1, min(test_size, 0.3))
        self.val_size = max(0.0, min(val_size, 0.15))
        self.debug = debug

    def load_and_split(self):
        if self.debug:
            self._debug()

        self._load()
        self._split()

        return self.train, self.val, self.test

    def _load(self):
        dataset = []

        for label in self.labels:
            label_path = os.path.join(self.dataset_path, label)

            if not os.path.exists(label_path):
                continue

            for image_name in os.listdir(label_path):
                if not image_name.lower().endswith(ACCEPTED_IMAGE_FORMATS):
                    continue

                image_path = os.path.join(label_path, image_name)

                if not os.path.isfile(image_path):
                    continue

                dataset.append((image_path, label))

        self.dataset = dataset

    def _split(self, seed=42):
        if not self.dataset:
            raise ValueError("dataset is empty, check dataset path and structure")

        paths = [sample[0] for sample in self.dataset]
        labels = [sample[1] for sample in self.dataset]

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths,
            labels,
            test_size=self.test_size + self.val_size,
            random_state=seed,
            stratify=labels,
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=seed,
            stratify=temp_labels,
        )

        self.train = list(zip(train_paths, train_labels))
        self.val = list(zip(val_paths, val_labels))
        self.test = list(zip(test_paths, test_labels))

    def _debug(self):
        train = f"{(1.0 - self.val_size - self.test_size) * 100}%"
        val = f"{self.val_size * 100}%"
        test = f"{self.test_size * 100}%"

        print(
            f"Loading dataset: splitting into {train} training, {val} validation and {test} testing"
        )
