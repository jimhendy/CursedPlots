import itertools
from typing import Callable, Optional, Tuple

import numpy as np
import numpy_indexed as npi

from utils import data_utils

SQRT_TWO = np.sqrt(2)


def _get_extreme(
    lims: Optional[Tuple[float, float]],
    extreme_func: Callable,
    series: np.ndarray,
    index: int,
    buffer: float,
) -> float:
    """
    Find the max/min of a series or use the supplied limit if available
    """
    if lims:
        assert len(lims) == 2, f"Limits must be of length 2: {lims}"
        assert (
            lims[1] > lims[0]
        ), f"Second limit must be larger than first (min, max): {lims}"
        extreme = lims[index]
    else:
        extreme = extreme_func(series)
        if buffer:
            offset = np.ptp(series) * buffer
            extreme += offset if extreme_func is np.max else -offset
    return extreme


def _get_max(
    lims: Optional[Tuple[float, float]], series: np.ndarray, buffer: float = 0
) -> float:
    return _get_extreme(
        lims=lims, series=series, extreme_func=np.max, index=1, buffer=buffer
    )


def _get_min(
    lims: Optional[Tuple[float, float]], series: np.ndarray, buffer: float = 0
) -> float:
    return _get_extreme(
        lims=lims, series=series, extreme_func=np.min, index=0, buffer=buffer
    )


def _translate_data_to_grid(
    data: np.ndarray[float, float],
    grid_maxs: Tuple[int, ...],
    x_lims: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Tuple[float, float]] = None,
    buffer: float = 0.05,
) -> np.ndarray[float, float]:
    """
    Translate plot data (x,y) to coordinates on a grid of shape `grid_shape`.

    Optionally axis limits can be passed via the args: TODO
    If no limits are passed, the data extremes are used as limits considering the `buffer`.

    Args:
        data:
            2d array of floats. The input data in (x,y) coordinate space.
            `x` = data_utils.data_x(data), `y` = data_utils.data_y(data).
            `len(data.shape) == 2`
            `data.shape[1] == 2`

        grid_maxs:
            2-tuple of ints of the maximum value allowed in the grid.
            The grid is assumed to start at 0 in both dimensions.
            These are domain of the output data.

        x_lims:
            (Optional) 2-tuple of x-axis min, max. If not supplied (`None`) the
            extremes of the data are used.

        y_lims:
            (Optional) As for `x_lims` but for the y-axis.

        buffer:
            Fraction of extra space to add between the data extremes and edge of the grid.
            If `x_lims` or `y_lims` are supplied, buffer is **NOT** used.
            E.g. if `buffer=0.1`, 10% the maximum x value will be at
            `grid_shape[0] - buffer * (max(data[0]) - min(data[0]))`
    """
    # assert shapes are correct
    data_utils.assert_data_shape(data)

    x = data_utils.data_x(data)
    y = data_utils.data_y(data)

    # get realised maxs n mins
    maxs = np.array((_get_max(x_lims, x, buffer), _get_max(y_lims, y, buffer)))
    mins = np.array((_get_min(x_lims, x, buffer), _get_min(y_lims, y, buffer)))

    # do translation
    return np.multiply(
        np.divide(np.subtract(data, mins), np.subtract(maxs, mins)), grid_maxs
    )


def _interpolate_regularly(data: np.ndarray[float, float]) -> np.ndarray[float, float]:
    """
    Ensure the supplied `data` are sampled once per integer value ()
    """
    x = data_utils.data_x(data)
    y = data_utils.data_y(data)

    num_points = int(np.ptp(x)) + 1
    interp_x = np.linspace(x.min(), x.max(), num_points)
    assert interp_x.min() == x.min()
    assert interp_x.max() == x.max()

    interp_y = np.interp(x=interp_x, xp=x, fp=y)

    return data_utils.xy_to_data(interp_x, interp_y)


def _inverse_distance_to_vertex(
    data: np.ndarray, increase_x: bool, increase_y: bool
) -> Tuple[np.ndarray, np.ndarray]:
    vertex = data.astype(np.int64)  # Round down to bottom left vertex
    if increase_x:
        vertex[:, 0] += 1
    if increase_y:
        vertex[:, 1] += 1
    distance = np.power(np.power(np.subtract(data, vertex), 2).sum(axis=1), 0.5)
    weight = (SQRT_TWO - distance) / SQRT_TWO
    return vertex, weight


def _anti_aliased_data(data: np.ndarray[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap the `data` coordinates to integers and calculate a weight for each interger vertex
    using the distance from the input `data`.

    E.g.

    data = np.array([[0.75, 0.5]])

    Distance to integer point (0,0) is ((0.75-0)**2 + (0.5-0)**2 )**0.5 so weight assigned to (0,0)
    is sqrt(2) - <distance> (sqrt(2) is the maximum distance in the unit square hypotenuse)

    Distance to integer point (1,0) is ((0.75-1)**2 + (0.5-0)**2 )**0.5 so weight assigned to (1,0)
    is sqrt(2) - <distance>

    Similarly for (0,1) and (1,1)
    """
    bools = (False, True)
    weighted_points = [
        _inverse_distance_to_vertex(data=data, increase_x=x, increase_y=y)
        for x, y in itertools.product(bools, bools)
    ]

    vertices = np.concatenate([i[0] for i in weighted_points])
    weights = np.concatenate([i[1] for i in weighted_points])

    locs, weights = npi.group_by(vertices).sum(weights)

    weights = np.clip(weights, a_max=1, a_min=0)

    return locs, weights


def anti_alias(
    data: np.ndarray, grid_maxs: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate the supplied data onto the grid and smooth via anti-aliasing
    """
    grid_data = _translate_data_to_grid(data, grid_maxs=grid_maxs)
    interp_data = _interpolate_regularly(grid_data)
    return _anti_aliased_data(interp_data)
