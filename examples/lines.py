import curses

import _curses
import numpy as np
from _utils import ensure_module_available

ensure_module_available()
from cursed_plots import LinePlot, data_utils  # pylint: disable=wrong-import-position

X_LIMS = [-5, 5]
Y_LIMS = [-1.5, 1.5]
X_POINTS = 50
X = np.linspace(start=X_LIMS[0], stop=X_LIMS[1], num=X_POINTS)


def sin_func(time_: int) -> np.ndarray:
    """Example sin function"""
    y = (np.arange(len(X)) + 1) * np.sin(X + time_ * 0.1) / X_POINTS
    return data_utils.xy_to_data(X, y)


def cos_func(time_: int) -> np.ndarray:
    """Example cos function"""
    y = np.cos(X * 3 + time_) * (np.arange(X_POINTS) + 1)[::-1] / X_POINTS
    return data_utils.xy_to_data(X, y)


def tan_func(time_: int) -> np.ndarray:  # pylint: disable=unused-argument
    """Example power function"""
    y = np.tan(X)
    return data_utils.xy_to_data(X, y)


def main(stdscr: _curses.window) -> None:
    """
    Execution function for plotting example line plot
    """
    plot = LinePlot(
        screen=stdscr,
        x_lims=X_LIMS,
        y_lims=Y_LIMS,
        functions=[sin_func, cos_func, tan_func],
    )
    plot.plot()


if __name__ == "__main__":
    curses.wrapper(main)
