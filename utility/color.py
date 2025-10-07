import random


class Painter:
    """
    A class for coloring strings.
    """

    def __init__(self):
        """
        Initialize the painter with an array containing strings of all available colors.
        """
        self.colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]

    def black(self, string: str):
        """
        Color the input string black.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[30m{string}\033[0m"

    def red(self, string: str):
        """
        Color the input string red.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;31m{string}\033[0m"

    def green(self, string: str):
        """
        Color the input string green.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;32m{string}\033[0m"

    def yellow(self, string: str):
        """
        Color the input string yellow.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;33m{string}\033[0m"

    def blue(self, string: str):
        """
        Color the input string blue.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;34m{string}\033[0m"

    def magenta(self, string: str):
        """
        Color the input string magenta.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;35m{string}\033[0m"

    def cyan(self, string: str):
        """
        Color the input string cyan.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;36m{string}\033[0m"

    def white(self, string: str):
        """
        Color the input string white.
        :param string: The string to color.
        :return: The colored string.
        """
        return f"\033[1;37m{string}\033[0m"

    def rainbow(self, string: str):
        """
        Color the input string in random colors.
        :param string: The string to color.
        :return: The colored string.
        """
        rainbow: str = ""
        for char in string:
            color = random.choice(self.colors)
            if color == "black":
                rainbow += self.black(char)
            elif color == "red":
                rainbow += self.red(char)
            elif color == "green":
                rainbow += self.green(char)
            elif color == "yellow":
                rainbow += self.yellow(char)
            elif color == "blue":
                rainbow += self.blue(char)
            elif color == "magenta":
                rainbow += self.magenta(char)
            elif color == "cyan":
                rainbow += self.cyan(char)
            elif color == "white":
                rainbow += self.white(char)
        return rainbow
