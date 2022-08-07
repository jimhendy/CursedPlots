class MockWindow:  # pylint: disable=too-few-public-methods
    """
    Dummy _curses.window for use in tests
    """

    def __init__(self):
        ...

    def getmaxyx(self):
        return [5, 10]
