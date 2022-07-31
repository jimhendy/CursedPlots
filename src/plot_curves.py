import curses

import _curses
import numpy as np

from plots import LinePlot
from utils import data_utils


def get_data(time_: int) -> np.ndarray:
    """Example sin function"""
    x_span = 10
    x_points = 50
    x = np.linspace(start=time_, stop=time_ + x_span, num=x_points)
    y = (np.arange(len(x)) + 1) * np.sin(x + time_)
    return data_utils.xy_to_data(x, y)


def get_data_2(time_: int) -> np.ndarray:
    """Example cos function"""
    x_span = 10
    x_points = 100
    x = np.linspace(start=time_, stop=time_ + x_span, num=x_points)
    y = np.cos(x * 3) * (np.arange(len(x)) + 1)[::-1]
    return data_utils.xy_to_data(x, y)


def get_data_3(time_: int) -> np.ndarray:
    """Example power function"""
    x_span = 10
    x_points = 100
    x = np.linspace(start=1, stop=time_ + x_span, num=x_points)
    y = np.power(x[:], 2)
    return data_utils.xy_to_data(x, y)


def main(stdscr: _curses.window) -> None:
    """
    Execution function for plotting example line plot
    """
    plot = LinePlot(screen=stdscr, functions=[get_data, get_data_2, get_data_3])
    plot.plot(iterations=500)


if __name__ == "__main__":
    curses.wrapper(main)
