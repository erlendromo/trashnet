import cv2
import numpy as np
import pytest

from src.feature_extractor import FeatureExtractor


def create_test_image():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = 50  # B
    image[:, :, 1] = 100  # G
    image[:, :, 2] = 150  # R

    return image


def test_init_creates_images():
    image = create_test_image()
    feature_extractor = FeatureExtractor(image)

    assert feature_extractor.bgr_image.shape == (128, 128, 3)
    assert feature_extractor.color_image is not None
    assert feature_extractor.gray_image is not None


def test_color_features():
    image = create_test_image()
    feature_extractor = FeatureExtractor(image)

    feature_extractor._color_features()

    assert isinstance(feature_extractor.color_features, list)
    assert len(feature_extractor.color_features) > 0
    assert not np.isnan(feature_extractor.color_features).any()


def test_hog_features():
    image = create_test_image()
    feature_extractor = FeatureExtractor(image)

    feature_extractor._hog_features()

    assert isinstance(feature_extractor.hog_features, list)
    assert len(feature_extractor.hog_features) > 0
    assert not np.isnan(feature_extractor.hog_features).any()


def test_lbp_features_length():
    image = create_test_image()
    feature_extractor = FeatureExtractor(image)

    feature_extractor._lbp_features()

    assert len(feature_extractor.lbp_features) == 10

    array = np.array(feature_extractor.lbp_features)
    assert np.isclose(array.sum(), 1.0, atol=1e-5)


def test_superpixel_features_length():
    image = create_test_image()
    feature_extractor = FeatureExtractor(image)

    feature_extractor._superpixel_features()

    expected = 50 * 3

    assert len(feature_extractor.superpixel_features) == expected
    assert not np.isnan(feature_extractor.superpixel_features).any()


def test_extract_returns_vector():
    image = create_test_image()
    feature_extractor = FeatureExtractor(image)

    features = feature_extractor.extract()

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.ndim == 1
    assert len(features) > 0
    assert not np.isnan(features).any()


def test_extract_is_deterministic():
    image = create_test_image()

    feature_extractor1 = FeatureExtractor(image)
    feature_extractor2 = FeatureExtractor(image)

    features1 = feature_extractor1.extract()
    features2 = feature_extractor2.extract()

    np.testing.assert_allclose(features1, features2)


def test_feature_vector_size_consistency():
    image = create_test_image()

    feature_extractor = FeatureExtractor(image)
    features = feature_extractor.extract()

    assert features.shape == (len(features),)
