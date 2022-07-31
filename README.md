# CursedPlots

Using the [`curses`](https://docs.python.org/3/howto/curses.html) module ([docs](https://docs.python.org/3/library/curses.html)) to plot animations things on the terminal.Designed using `python 3.10`.

![Example plotting animation](curves.gif)

## ToDos

 - [x] Curves are currently inverted in y-axis - swap
 - [ ] A few extra characters on the bottom row - suspect this is caused by incorrect terminal size
 - [ ] Add axis if max(data) * min(data) < 0
 - [ ] Each curve currently on distinct axis - allow common limits
 - [ ] Add legend
 - [ ] Add rich for tooltips and legend clicking to remove trace (as for plotly)
 - [ ] Use numba to speed up anti-aliasing
 - [ ] Additional plot types:
     - [ ] Histogram
     - [ ] Scatter (simplified line)
 - [ ] Mutliple plots should work out of the box as each `screen` in curses is only aware of it's own bounds. Check this works
 - [ ] Visual testing using playwrgiht
 - [x] Pre-commit hooks
 - [x] CI Testing
 - [x] Don't allow master commits
