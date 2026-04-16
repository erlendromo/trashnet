import os

from sklearn.model_selection import train_test_split

LABELS = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

ACCEPTED_FORMATS = (".jpeg", ".jpg", ".png")


class Dataset:
    def __init__(self, labels=LABELS, test_size=0.15, val_size=0.10):
        self.dataset_path = os.getenv("DATASET_PATH", "./dataset")
        self.labels = labels
        self.test_size = max(0.1, min(test_size, 0.3))
        self.val_size = max(0.05, min(val_size, 0.15))

    def load_and_split(self):
        self._load()
        self._split()

        return self.train, self.val, self.test

    def _load(self):
        self.dataset = []

        for label in self.labels:
            label_path = os.path.join(self.dataset_path, label)

            if not os.path.exists(label_path):
                continue

            for file in os.listdir(label_path):
                if not file.lower().endswith(ACCEPTED_FORMATS):
                    continue

                file_path = os.path.join(label_path, file)

                if not os.path.isfile(file_path):
                    continue

                self.dataset.append((file_path, label))

        if not self.dataset:
            raise ValueError("Dataset is empty. Check dataset path and structure.")

        return self.dataset

    def _split(self, seed=42):
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
