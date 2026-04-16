import numpy as np
import pytest
from cv2 import COLOR_BGR2GRAY, COLOR_BGR2HSV

from src.preprocessor import Preprocessor


def test_init_raises_on_none_image():
    with pytest.raises(ValueError):
        Preprocessor(image=None)


def test_resize_output_shape():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    p = Preprocessor(image=img, size=200)
    output = p.preprocess()

    assert output.shape[0] == 200
    assert output.shape[1] == 200


def test_normalization_range():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    p = Preprocessor(image=img)
    output = p.preprocess()

    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_output_dtype():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    p = Preprocessor(image=img)
    output = p.preprocess()

    assert output.dtype == np.float32


def test_grayscale_has_channel_dimension():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    p = Preprocessor(image=img, color=COLOR_BGR2GRAY)
    output = p.preprocess()

    assert len(output.shape) == 3  # [width, height, channels]
    assert output.shape[2] == 1  # Only gray channel


def test_invalid_color_defaults_to_hsv():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    invalid_color = -1
    p = Preprocessor(image=img, color=invalid_color)

    assert p.color == COLOR_BGR2HSV


def test_noise_changes_image():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    p_no_noise = Preprocessor(image=img, noise=False)
    out1 = p_no_noise.preprocess()

    p_noise = Preprocessor(image=img, noise=True)
    out2 = p_noise.preprocess()

    # They should not be identical
    assert not np.allclose(out1, out2)


def test_size_clamping():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    p_small = Preprocessor(image=img, size=50)
    assert p_small.size == 128

    p_large = Preprocessor(image=img, size=1000)
    assert p_large.size == 512
