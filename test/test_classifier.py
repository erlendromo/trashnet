import pytest

from src.classifier import Classifier


@pytest.fixture
def classifier():
    classifier = Classifier()

    return classifier


def test_xxx(classifier):
    assert 1 >= 0
