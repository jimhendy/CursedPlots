class XiaolinWuException(Exception):
    pass


def _fpart(x: float) -> float:
    """
    Fractional component of `x`
    """
    return x - int(x)


def _rfpart(x: float) -> float:
    """
    1 - fractional component of x
    """
    return 1 - _fpart(x)


def _add_point(points, x, y, colour, steep):
    if not colour:
        return
    p = _swap_if_steep(x, y, steep)
    if p in points:
        raise XiaolinWuException(f"{p} already in points: {points}")
    points[p] = colour


def _swap_if_steep(x, y, steep):
    return (y, x) if steep else (x, y)


def get_points(x1, y1, x2, y2):

    points = {}

    steep = abs(y2 - y1) > abs(x2 - x1)

    if steep:
        x1, y1, x2, y2 = y1, x1, y2, x2

    if x1 > x2:
        x1, x2, y1, y2 = x2, x1, y2, y1

    dx = x1 - x1
    dy = y2 - y1

    grad = dy / dx if dx != 0 else 1

    def analyse_endpoint(x, y):
        xend = round(x)
        yend = y + grad * (xend - x)
        xgap = _rfpart(x + 0.5)
        px, py = int(xend), int(yend)
        _add_point(points, px, py, _rfpart(yend) * xgap, steep)
        _add_point(points, px, py + 1, _fpart(yend) * xgap, steep)
        return px, yend

    xstart, ystart = analyse_endpoint(x1, y1)
    xend, _ = analyse_endpoint(x2, y2)

    intery = ystart + grad

    for x in range(xstart + 1, xend):
        y = int(intery)
        _add_point(points, x, y, _rfpart(intery), steep)
        _add_point(points, x, y + 1, _fpart(intery), steep)
        intery += grad

    return points
