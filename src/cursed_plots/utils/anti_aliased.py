import itertools
from typing import Tuple

import numpy as np
import numpy_indexed as npi

from ..utils import data_utils

SQRT_TWO = np.sqrt(2)


def interpolate_regularly(data: np.ndarray[float, float]) -> np.ndarray[float, float]:
    """
    Ensure the supplied `data` are sampled once per integer value ()
    """
    x = data_utils.data_x(data)
    y = data_utils.data_y(data)

    if x.min() == x.max():
        interp_x = [x.min(), x.max()]
        interp_y = [y.min(), y.max()]
    else:
        num_points = int(np.ptp(x)) + 1
        interp_x = np.linspace(x.min(), x.max(), num=num_points)
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


def anti_aliased_data(data: np.ndarray[float, float]) -> Tuple[np.ndarray, np.ndarray]:
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
