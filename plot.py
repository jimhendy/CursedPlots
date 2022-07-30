import curses
import time
from typing import Optional, Tuple

import _curses
import bresenham
import numpy as np

from utils.xiaolin_wu import get_points

np.random.seed(123)

characters = (" ", "#")

# Borrowing from: http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/


def show(stdscr: _curses.window, grid: np.ndarray):
    [
        stdscr.addch(row_num, col_num, characters[char])
        for row_num, row in enumerate(grid)
        for col_num, char in enumerate(row)
    ]
    stdscr.refresh()


def get_data(time: int):
    x_span = 10
    x_points = 100
    x = np.linspace(start=time, stop=time + x_span, num=x_points)
    y = np.sin(x)
    return x, y


def data_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> None:
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape == y.shape

    rows, cols = grid.shape

    # coord -> (coord + coord_min) / coord_max * grid_max

    to_grid = lambda coords, grid_max: np.round(
        (coords - min(coords)) / (max(coords) - min(coords)) * grid_max
    ).astype(int)
    x_grid = to_grid(x, cols - 1)
    y_grid = to_grid(y, rows - 1)

    grid[:] = 0
    for x1, y1, x2, y2 in zip(x_grid[:-1], y_grid[:-1], x_grid[1:], y_grid[1:]):
        points = list(bresenham.bresenham(x1, y1, x2, y2))
        x = get_points(x1, y1, x2, y2)
        for p in points:
            grid[p[1], p[0]] = 1


def plot(stdscr: _curses.window, iterations: Optional[int] = None):
    curses.curs_set(False)
    stdscr.clear()

    size: Tuple[int, int] = tuple(i - 1 for i in stdscr.getmaxyx())
    board = np.zeros(size, dtype=np.uint8)
    iterations = iterations or int(10e10)

    for t in range(iterations):
        x, y = get_data(t * 0.1)
        data_to_grid(x=x, y=y, grid=board)
        show(stdscr=stdscr, grid=board)
        t += 1
        time.sleep(7e-2)

    time.sleep(2)


if __name__ == "__main__":
    curses.wrapper(plot, iterations=1_000)
