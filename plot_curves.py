import curses
import re
import time
from typing import Optional, Tuple

import _curses
import numpy as np

from utils import anti_aliased, data_utils
from utils.xiaolin_wu import get_points

np.random.seed(123)

CHARACTERS = list(" .:#")
"""
CHARACTERS = [
    " ",
    "\u2581",
    "\u2582",
    "\u2583",
    "\u2584",
    "\u2585",
    "\u2586",
    "\u2587",
    "\u2588",
    "\u2589",
]
"""


def _character_from_alpha(alpha: float):
    assert 0 <= alpha <= 1
    n_characters = len(CHARACTERS)
    return CHARACTERS[int(alpha * (n_characters - 0.5))]


# def show(stdscr: _curses.window, grid: np.ndarray):
#     [
#         set_char(stdscr, row_num, col_num, char, 0)
#         for row_num, row in enumerate(grid)
#         for col_num, char in enumerate(row)
#     ]
#     stdscr.refresh()


def set_char(stdscr, row_num, col_num, char, color_num):
    stdscr.addch(row_num, col_num, char, curses.color_pair(color_num + 2))


def get_data(time: int):
    x_span = 10
    x_points = 50
    x = np.linspace(start=time, stop=time + x_span, num=x_points)
    y = np.sin(x)
    return data_utils.xy_to_data(x, y)


def get_data_2(time: int):
    x_span = 10
    x_points = 100
    x = np.linspace(start=time, stop=time + x_span, num=x_points)
    y = 0.2 * np.cos(x * 3)
    return data_utils.xy_to_data(x, y)


def get_data_3(time: int):
    x_span = 10
    x_points = 100
    x = np.linspace(start=1, stop=time + x_span, num=x_points)
    y = np.power(x[:], 2)
    return data_utils.xy_to_data(x, y)


def reset_grid(stdscr: _curses.window, grid: np.ndarray):
    # grid[:] = CHARACTERS[0]
    char = CHARACTERS[0]
    [
        set_char(stdscr, row_num, col_num, char, 0)
        for row_num in range(grid.shape[0])
        for col_num in range(grid.shape[1])
    ]


def fill_grid(
    data: np.ndarray, grid: np.ndarray, color_num: int, stdscr: _curses.window
) -> None:

    grid_maxs = [i - 2 for i in grid.shape[::-1]]
    weights = anti_aliased.anti_alias(data, grid_maxs)

    for point, alpha in weights.items():
        set_char(
            stdscr=stdscr,
            row_num=point[1],
            col_num=point[0],
            char=_character_from_alpha(alpha),
            color_num=color_num,
        )


def plot(stdscr: _curses.window, iterations: Optional[int] = None):
    curses.curs_set(False)

    curses.start_color()
    curses.use_default_colors()
    for i in range(curses.COLORS):
        curses.init_pair(i + 1, i, -1)
    stdscr.clear()

    size: Tuple[int, int] = tuple(i - 1 for i in stdscr.getmaxyx())
    board = np.zeros(size, dtype=str)
    iterations = iterations or int(10e10)

    funcs = (get_data, get_data_2, get_data_3)

    for t in range(iterations):

        reset_grid(stdscr=stdscr, grid=board)
        for i, func in enumerate(funcs):
            data = func(t * 0.1)
            fill_grid(data=data, grid=board, color_num=i, stdscr=stdscr)

        # show(stdscr=stdscr, grid=board)
        stdscr.refresh()
        t += 1
        time.sleep(2e-2)

    time.sleep(4)


if __name__ == "__main__":
    curses.wrapper(plot)  # , iterations=1_000)
