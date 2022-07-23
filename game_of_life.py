import curses
import time
from typing import Optional, Tuple

import _curses
import numpy as np
from numpy.lib.stride_tricks import as_strided

np.random.seed(123)

characters = (" ", "\u2665")

# Borrowing from: http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/


def show_game(stdscr: _curses.window, grid: np.ndarray):
    [
        stdscr.addch(row_num, col_num, characters[char])
        for row_num, row in enumerate(grid)
        for col_num, char in enumerate(row)
    ]
    stdscr.refresh()


def game_of_life(stdscr: _curses.window, iterations: Optional[int] = None):
    # Clear screen
    curses.curs_set(False)  # Hide the cursor
    stdscr.clear()  #  Clear the window

    board_size: Tuple[int, int] = tuple(i - 1 for i in stdscr.getmaxyx())
    full_size: Tuple[int, int] = tuple(i + 2 for i in board_size)

    full_board = np.zeros(full_size, dtype=np.uint8)

    n_dims: int = 2
    assert len(board_size) == n_dims

    visible_board_slice = (slice(1, -1),) * n_dims
    board = full_board[visible_board_slice]
    board[:] = np.random.choice(a=[0, 1], size=board.shape, p=[0.7, 0.3])

    iterations = iterations or int(10e10)

    sumOver = tuple(-(i + 1) for i in range(n_dims))

    # index is number of neighbors alive
    ruleOfLifeAlive = np.zeros(8 + 1, np.uint8)  # default all to dead
    ruleOfLifeAlive[[2, 3]] = 1  # alive stays alive <=> 2 or 3 neighbors

    ruleOfLifeDead = np.zeros(8 + 1, np.uint8)  # default all to dead
    ruleOfLifeDead[3] = 1  # dead switches to living <=> 3 neighbors

    newShape = [_len - 2 for _len in full_board.shape]
    newShape.extend([3] * n_dims)
    newStrides = full_board.strides + full_board.strides

    show_game(stdscr, board)
    for _ in range(iterations):
        time.sleep(7e-2)

        neighborhoods = as_strided(full_board, shape=newShape, strides=newStrides)
        neighborCt = np.sum(neighborhoods, sumOver) - board
        board[:] = np.where(
            board, ruleOfLifeAlive[neighborCt], ruleOfLifeDead[neighborCt]
        )

        show_game(stdscr, board)

    time.sleep(2)


if __name__ == "__main__":
    curses.wrapper(game_of_life, iterations=1_000)
