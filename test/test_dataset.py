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
            image = np.zeros((10, 10, 3), dtype=np.uint8)

            file_path = os.path.join(label_dir, f"{label}_{i}.jpg")

            cv2.imwrite(file_path, image)

    return list(labels)


def test_dataset_load():
    labels = ("glass", "metal")
    samples_per_label = 10
    total_samples = len(labels) * samples_per_label

    with tempfile.TemporaryDirectory() as temp:
        create_fake_dataset(temp, labels, samples_per_label)

        dataset = Dataset(labels=labels)
        dataset.dataset_path = temp

        data = dataset._load()

        assert len(data) == total_samples
        assert all(len(item) == 2 for item in data)


def test_file_format_filtering():
    labels = ("glass",)

    with tempfile.TemporaryDirectory() as temp:
        os.makedirs(os.path.join(temp, labels[0]))

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(temp, f"{labels[0]}/1.jpg"), image)

        with open(os.path.join(temp, f"{labels[0]}/1.txt"), "w") as f:
            f.write("not an image")

        dataset = Dataset(labels=labels)
        dataset.dataset_path = temp

        data = dataset._load()

        assert len(data) == 1


def test_split_sizes():
    labels = ("glass", "metal")
    samples_per_label = 10
    total_samples = len(labels) * samples_per_label

    with tempfile.TemporaryDirectory() as temp:
        create_fake_dataset(temp, labels, samples_per_label)

        dataset = Dataset(labels=labels)
        dataset.dataset_path = temp

        train, val, test = dataset.load_and_split()

        total = len(train) + len(val) + len(test)

        assert total == total_samples


def test_empty_dataset_raises():
    labels = ("glass",)

    with tempfile.TemporaryDirectory() as temp:
        dataset = Dataset(labels=labels)
        dataset.dataset_path = temp

        with pytest.raises(ValueError):
            dataset.load_and_split()


def test_labels_preserved():
    labels = ("glass", "metal")
    samples_per_label = 10

    with tempfile.TemporaryDirectory() as temp:
        create_fake_dataset(temp, labels, samples_per_label)

        dataset = Dataset(labels=labels)
        dataset.dataset_path = temp

        train, val, test = dataset.load_and_split()

        all_labels = {label for _, label in train + val + test}

        assert all_labels == {"glass", "metal"}
