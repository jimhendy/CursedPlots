import curses
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import _curses
import numpy as np
from loguru import logger

from utils import anti_aliased, data_utils


class PlottingException(Exception):
    ...


class Plot(ABC):

    CHARACTERS = list(
        r"""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. """
    )[::-1]

    def __init__(self, screen: _curses.window) -> None:
        self.screen = screen
        self.n_characters = len(Plot.CHARACTERS)
        self.screen_size = self._fetch_screen_size()

        curses.curs_set(False)
        curses.start_color()
        curses.use_default_colors()
        for i in range(curses.COLORS):
            curses.init_pair(i + 1, i, -1)
        self.screen.clear()

    def _fetch_screen_size(self):
        rows, cols = self.screen.getmaxyx()
        return rows - 1, cols

    @property
    def rows(self):
        return self.screen_size[0]

    @property
    def columns(self):
        return self.screen_size[1]

    def clear(self) -> None:
        window_size = self.screen.getmaxyx() # Use directly for full clean
        empty_string = " " * window_size[0] * (window_size[1] - 1)
        self.screen.addstr(0, 0, empty_string)

    def _character_from_alpha(self, alpha: float) -> str:
        assert 0 <= alpha <= 1
        return Plot.CHARACTERS[int(alpha * (self.n_characters - 0.5))]

    def set_char(
        self, row_num: int, col_num: int, char: str, color_num: int = 0
    ) -> None:
        if row_num >= self.rows:
            raise PlottingException(
                f"Cannot add to row {row_num} as screen is only {self.rows} rows"
            )
        if col_num >= self.columns:
            raise PlottingException(
                f"Cannot add to column {col_num} as screen is only {self.columns} columns"
            )
        self.screen.addch(row_num, col_num, char, curses.color_pair(color_num + 2))

    def refresh(self):
        self.screen.refresh()

    @abstractmethod
    def plot(self):
        ...


def get_data(time: int):
    x_span = 10
    x_points = 50
    x = np.linspace(start=time, stop=time + x_span, num=x_points)
    y = (np.arange(len(x)) + 1) * np.sin(x + time)
    return data_utils.xy_to_data(x, y)


def get_data_2(time: int):
    x_span = 10
    x_points = 100
    x = np.linspace(start=time, stop=time + x_span, num=x_points)
    y = np.cos(x * 3) * (np.arange(len(x)) + 1)[::-1]
    return data_utils.xy_to_data(x, y)


def get_data_3(time: int):
    x_span = 10
    x_points = 100
    x = np.linspace(start=1, stop=time + x_span, num=x_points)
    y = np.power(x[:], 2)
    return data_utils.xy_to_data(x, y)


class LinePlot(Plot):
    def __init__(self, screen: _curses.window, functions: List[Callable]) -> None:
        super().__init__(screen=screen)
        self.functions = functions

    def _fill_grid(self, data: np.ndarray, color_num: int) -> None:

        grid_maxs = [i - 2 for i in self.screen_size[::-1]]
        locations, weights = anti_aliased.anti_alias(data, grid_maxs)

        for point, alpha in zip(locations, weights):
            self.set_char(
                row_num=point[1],
                col_num=point[0],
                char=self._character_from_alpha(alpha),
                color_num=color_num,
            )

    def plot(self, iterations: Optional[int] = None):

        for t in range(iterations or int(10e10)):
            self.clear()
            for i, func in enumerate(self.functions):
                data = func(t * 0.1)
                self._fill_grid(data=data, color_num=i)

            self.refresh()
            t += 1
            time.sleep(6e-2)

        time.sleep(4)


def main(stdscr: _curses.window):
    plot = LinePlot(
        screen=stdscr, functions=[get_data , get_data_2, get_data_3])
    plot.plot(iterations=500)


if __name__ == "__main__":
    curses.wrapper(main)
