import pytest

from src.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    preprocessor = Preprocessor()

    return preprocessor


def test_xxx(preprocessor):
    assert # Something
