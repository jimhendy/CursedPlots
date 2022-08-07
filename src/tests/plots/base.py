from cursed_plots.plots.base import Plot
from cursed_plots.utils import data_utils

import pytest
from test_utils.window import MockWindow
import numpy as np

class TestTranslateDataToGrid:
    """
    Test the _translate_data_to_grid function
    """

    @staticmethod
    @pytest.fixture(name="BasePlot")
    def base_plot() -> Plot:
        return Plot(screen=MockWindow())

    @staticmethod
    def test_direct_translate(base_plot):
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        data = data_utils.xy_to_data(x, y)
        grid_maxs = [1, 1]
        actual = base_plot._translate_data_to_grid(
            data=data, grid_maxs=grid_maxs, buffer=0
        )
        np.testing.assert_equal(data, actual)

    @staticmethod
    def test_translate_scale_up(base_plot):
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        data = data_utils.xy_to_data(x, y)
        grid_maxs = [10, 10]
        actual = base_plot._translate_data_to_grid(
            data=data, grid_maxs=grid_maxs, buffer=0
        )
        expected = data_utils.xy_to_data(
            [i * grid_maxs[0] for i in x], [i * grid_maxs[1] for i in y]
        )
        np.testing.assert_equal(expected, actual)

    @staticmethod
    def test_translate_scale_up_unequal(base_plot):
        x = [0, 1]
        y = [0, 1]
        data = data_utils.xy_to_data(x, y)
        grid_maxs = [10, 5]
        actual = base_plot._translate_data_to_grid(
            data=data, grid_maxs=grid_maxs, buffer=0
        )
        expected = data_utils.xy_to_data(
            [i * grid_maxs[0] for i in x], [i * grid_maxs[1] for i in y]
        )
        np.testing.assert_equal(expected, actual)

    @staticmethod
    def test_translate_scale_down_unequal(base_plot):
        x = np.linspace(0, 100)
        y = np.linspace(0, 200)
        data = data_utils.xy_to_data(x, y)
        grid_maxs = [10, 2]
        actual = base_plot._translate_data_to_grid(
            data=data, grid_maxs=grid_maxs, buffer=0
        )
        expected = data_utils.xy_to_data(
            [i * grid_maxs[0] / x.max() for i in x],
            [i * grid_maxs[1] / y.max() for i in y],
        )
        np.testing.assert_almost_equal(expected, actual)

    @staticmethod
    def test_translate_scale_up_unequal_min_nonzero(base_plot):
        x = [0.5, 0.75, 1]
        y = x[:]
        scale = lambda i: (i - min(x)) / (max(x) - min(x))
        data = data_utils.xy_to_data(x, y)
        grid_maxs = [10, 5]
        actual = base_plot._translate_data_to_grid(
            data=data, grid_maxs=grid_maxs, buffer=0
        )
        expected = data_utils.xy_to_data(
            [scale(i) * grid_maxs[0] for i in x], [scale(i) * grid_maxs[1] for i in y]
        )
        np.testing.assert_equal(expected, actual)