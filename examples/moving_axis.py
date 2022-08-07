from _utils import ensure_module_available

ensure_module_available()

import curses

import _curses
import numpy as np

from cursed_plots import LinePlot, data_utils

START = -5
X_SPAN = 10
X_POINTS = 50


def sin_func(time_: int) -> np.ndarray:
    time_ *= 0.1
    x = np.linspace(time_ + START, time_ + START + X_SPAN, num=X_POINTS)
    y = np.sin(x)
    return data_utils.xy_to_data(x=x, y=y)


def main(stdscr: _curses.window) -> None:
    """
    Execution function for plotting example line plot
    """
    plot = LinePlot(
        screen=stdscr,
        functions=[sin_func],
        step_delay=1e-2,
        y_lims=[-1.5, 1.5],
    )
    plot.plot()


if __name__ == "__main__":
    curses.wrapper(main)
