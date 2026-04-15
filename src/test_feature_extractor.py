import pytest

from src.feature_extractor import FeatureExtractor


@pytest.fixture
def feature_extractor():
    feature_extractor = FeatureExtractor()

    return feature_extractor


def test_xxx(feature_extractor):
    assert # Something
