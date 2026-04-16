import os
import tempfile

import cv2
import numpy as np
import pytest

from src.dataset import Dataset


def create_fake_dataset(base_path, labels=("glass", "metal"), samples_per_label=10):
    for label in labels:
        label_dir = os.path.join(base_path, label)
        os.makedirs(label_dir, exist_ok=True)

        for i in range(samples_per_label):
            img = np.zeros((10, 10, 3), dtype=np.uint8)

            # Make filenames unique and explicit
            file_path = os.path.join(label_dir, f"{label}_{i}.jpg")

            cv2.imwrite(file_path, img)

    return list(labels)


def test_dataset_load():
    with tempfile.TemporaryDirectory() as tmp:
        create_fake_dataset(tmp)

        dataset = Dataset(labels=("glass", "metal"))
        dataset.dataset_path = tmp

        data = dataset._load()

        assert len(data) == 20
        assert all(len(item) == 2 for item in data)


def test_file_format_filtering():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "glass"))

        # valid image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp, "glass/1.jpg"), img)

        # invalid file
        with open(os.path.join(tmp, "glass/1.txt"), "w") as f:
            f.write("not an image")

        dataset = Dataset(labels=("glass",))
        dataset.dataset_path = tmp

        data = dataset._load()

        assert len(data) == 1


def test_split_sizes():
    with tempfile.TemporaryDirectory() as tmp:
        create_fake_dataset(tmp)

        dataset = Dataset(labels=("glass", "metal"))
        dataset.dataset_path = tmp

        train, val, test = dataset.load_and_split()

        total = len(train) + len(val) + len(test)

        assert total == 20


def test_empty_dataset_raises():
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Dataset(labels=("glass",))
        dataset.dataset_path = tmp

        with pytest.raises(ValueError):
            dataset.load_and_split()


def test_labels_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        create_fake_dataset(tmp)

        dataset = Dataset(labels=("glass", "metal"))
        dataset.dataset_path = tmp

        train, val, test = dataset.load_and_split()

        all_labels = {label for _, label in train + val + test}

        assert all_labels == {"glass", "metal"}
