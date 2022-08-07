from typing import Any, Optional, Tuple

from cursed_plots.plots.base import Plot

from .window import MockWindow


class BasicPlot(Plot):
    def __init__(self, buffer: float = 0, **kwargs):
        super().__init__(screen=MockWindow(), buffer=buffer, **kwargs)

    def plot(self):
        ...

    def _init_curses(self):
        ...


def get_mock_plot(
    screen_size: Optional[Tuple[int, int]] = None,
    grid_maxs: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
):
    plot = BasicPlot(**kwargs)
    if screen_size:
        plot._fetch_screen_size = lambda *args, **kwargs: screen_size
        plot.screen_size = plot._fetch_screen_size()
    if grid_maxs:
        plot.screen_size = [i + 2 for i in grid_maxs[::-1]]
    return plot
