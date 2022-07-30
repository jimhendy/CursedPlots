import numpy as np
import pytest

from utils import data_utils


class TestXYToData:
    @staticmethod
    @pytest.fixture
    def array() -> np.ndarray:
        return np.array(range(50))

    @staticmethod
    @pytest.fixture
    def x() -> np.array:
        return np.array([0, 1, 2])

    @staticmethod
    @pytest.fixture
    def y() -> np.array:
        return np.array([3, 4, 5])

    @staticmethod
    @pytest.fixture
    def data(x, y) -> np.array:
        return data_utils.xy_to_data(x, y)

    @staticmethod
    def test_stack_direction(x, y) -> None:
        expected = np.array([[xi, yi] for xi, yi in zip(x, y)])
        actual = data_utils.xy_to_data(x, y)
        np.testing.assert_equal(actual, expected)

    @staticmethod
    def test_stack_direction_large(array: np.ndarray) -> None:
        expected = np.array([[xi, yi] for xi, yi in zip(array, array)])
        actual = data_utils.xy_to_data(array, array)
        np.testing.assert_equal(actual, expected)

    @staticmethod
    def test_extracted_dimensions(data, x, y) -> None:
        np.testing.assert_equal(data_utils.data_x(data), x)
        np.testing.assert_equal(data_utils.data_y(data), y)
