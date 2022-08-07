import numpy as np
import pytest

from cursed_plots.plots.base import Plot
from cursed_plots.utils import data_utils

from ..test_utils.basic_plot import BasicPlot, get_mock_plot
from ..test_utils.window import MockWindow


class TestTranslateDataToGrid:
    """
    Test the _translate_data_to_grid function
    """

    @staticmethod
    def test_direct_translate():
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        data = data_utils.xy_to_data(x, y)
        plot = get_mock_plot(grid_maxs=(1, 1))
        actual = plot._translate_data_to_grid(data=data)
        np.testing.assert_equal(data, actual)

    @staticmethod
    def test_translate_scale_up():
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]
        data = data_utils.xy_to_data(x, y)
        plot = get_mock_plot(grid_maxs=(10, 10))
        actual = plot._translate_data_to_grid(data=data)
        expected = data_utils.xy_to_data(
            [i * plot.grid_maxs[0] for i in x], [i * plot.grid_maxs[1] for i in y]
        )
        np.testing.assert_equal(expected, actual)

    @staticmethod
    def test_translate_scale_up_unequal():
        x = [0, 1]
        y = [0, 1]
        data = data_utils.xy_to_data(x, y)
        plot = get_mock_plot(grid_maxs=(10, 5))
        actual = plot._translate_data_to_grid(data=data)
        expected = data_utils.xy_to_data(
            [i * plot.grid_maxs[0] for i in x], [i * plot.grid_maxs[1] for i in y]
        )
        np.testing.assert_equal(expected, actual)

    @staticmethod
    def test_translate_scale_down_unequal():
        x = np.linspace(0, 100)
        y = np.linspace(0, 200)
        data = data_utils.xy_to_data(x, y)
        plot = get_mock_plot(grid_maxs=(10, 2))
        actual = plot._translate_data_to_grid(data=data)
        expected = data_utils.xy_to_data(
            [i * plot.grid_maxs[0] / x.max() for i in x],
            [i * plot.grid_maxs[1] / y.max() for i in y],
        )
        np.testing.assert_almost_equal(expected, actual)

    @staticmethod
    def test_translate_scale_up_unequal_min_nonzero():
        x = [0.5, 0.75, 1]
        y = x[:]
        scale = lambda i: (i - min(x)) / (max(x) - min(x))
        data = data_utils.xy_to_data(x, y)
        plot = get_mock_plot(grid_maxs=(10, 5))
        actual = plot._translate_data_to_grid(data=data)
        expected = data_utils.xy_to_data(
            [scale(i) * plot.grid_maxs[0] for i in x],
            [scale(i) * plot.grid_maxs[1] for i in y],
        )
        np.testing.assert_equal(expected, actual)


class TestGetExtreme:
    @staticmethod
    def test_max_no_lims():
        data = list(range(10))
        expected = max(data)
        actual = Plot._get_max(lims=None, series=data)
        assert expected == actual

    @staticmethod
    def test_min_no_lims():
        data = list(range(10))
        expected = min(data)
        actual = Plot._get_min(lims=None, series=data)
        assert expected == actual

    @staticmethod
    def test_max_with_lims():
        data = list(range(10))
        lims = [0, 5]
        expected = lims[1]
        actual = Plot._get_max(lims=lims, series=data)
        assert expected == actual

    @staticmethod
    def test_min_with_lims():
        data = list(range(10))
        lims = [-100, 5]
        expected = lims[0]
        actual = Plot._get_min(lims=lims, series=data)
        assert expected == actual

    @staticmethod
    def test_max_offset():
        data = list(range(10))
        buffer = 0.5
        expected = max(data) + buffer * np.ptp(data)
        actual = Plot._get_max(lims=None, series=data, buffer=buffer)
        assert actual == expected
