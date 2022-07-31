from typing import List, Optional, Union

import numpy as np


def _ensure_array(
    x: Union[List, np.ndarray], assert_dimensions: Optional[int] = None
) -> np.ndarray:
    if isinstance(x, (list, tuple, set)):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise ValueError(f"Cannot convert arg to numpy array, {x}")

    if assert_dimensions is not None:
        assert len(x.shape) == assert_dimensions

    return x


def xy_to_data(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Combine `x` and `y` data into a single numpy array of the correct shape
    """
    x = _ensure_array(x, 1)
    y = _ensure_array(y, 1)
    assert y.shape[0] == x.shape[0]
    return np.vstack((x, y)).T


def data_x(data: np.ndarray) -> np.ndarray:
    """
    Extract the x coordinates from the data array
    """
    return data[:, 0]


def data_y(data: np.ndarray) -> np.ndarray:
    """
    Extract the y coordinates from the data array
    """
    return data[:, 1]


def data_len(data: np.ndarray) -> int:
    """
    Find the length(/number of occurances) of the supplied data
    """
    return data.shape[0]


def data_num_dimensions(data: np.ndarray) -> int:
    """
    Find the number of spatial dimentsion of the supplied data
    Normally to ensure we are in 2 dimensions
    """
    return data.shape[1]


def assert_data_shape(data: np.ndarray) -> None:
    """
    Test the supplied data are in the expected shape for plotting
    """
    # Ensure correct data shape (occurances and dimensions)
    assert len(data.shape) == 2
    # Ensure only 2 dimensions (x and y)
    assert data_num_dimensions(data) == 2
