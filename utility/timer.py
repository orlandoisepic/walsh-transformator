import time
from utility.color import Painter


class Timer:
    """
    The timer class represents a handy way of starting, lapping and stopping time.
    It can calculate timeframes between stopped time-points and
    return strings of the duration with a color indicating its length.
    """

    def __init__(self, round: int = 2) -> None:
        """
        Initialize the timer. This does not set the starting point yet.
        :param round: The number of decimal points a result will be rounded to.
        """
        self.round: int = round
        self.time_points: list[float] = []
        self.pearl: Painter = Painter()
        self.started: bool = False
        self.start_time: float = -1
        self.end_time: float = -1
        self.stopped: bool = False

    def start(self) -> None:
        """
        "Starts" the timer and appends the time to the time-points list.
        """
        self.started = True
        self.start_time = time.time()
        self.time_points.append(self.start_time)

    def lap(self) -> None:
        """
        Saves the current time-point to the time-points list. Works only if the time has been started and not yet stopped.
        """
        if self.started and not self.stopped:
            current_time = time.time()
            self.time_points.append(current_time)
        else:
            print("Timer has already stopped. Cannot create new time-points.")

    def stop(self) -> None:
        """
        "Stops" the timer and appends the time to the time-points list.
        """
        self.end_time = time.time()
        self.time_points.append(self.end_time)
        self.stopped = True

    def get_last_interval(self) -> float:
        """
        Returns the duration of the interval stopped last.
        :return: The duration of the interval.
        """
        if len(self.time_points) >= 1:
            return (self.time_points[-1] - self.time_points[-2]).__round__(self.round)
        else:
            print("Not enough time-points to calculate interval.")
            return -1

    def get_last_interval_string(self, base_duration: int = 30) -> str:
        """
        Return the last interval as a colored string, based on its duration.
        Based on the base_duration value, the color-thresholds are calculated as follows:\n
        green: base_duration * 1\n
        blue: base_duration * 2\n
        yellow: base_duration * 4\n
        magenta: base_duration * 10\n
        red: anything larger than base_duration * 10\n
        :param base_duration: The duration in seconds that will lead to a green-colored output.
        :return: A colored string.
        """
        interval = self.get_last_interval()
        if interval == -1:
            return ""
        else:
            if interval <= base_duration:
                return self.pearl.green(str(interval))
            elif interval <= 2 * base_duration:
                return self.pearl.blue(str(interval))
            elif interval <= 4 * base_duration:
                return self.pearl.yellow(str(interval))
            elif interval <= 10 * base_duration:
                return self.pearl.magenta(str(interval))
            else:
                return self.pearl.red(str(interval))

    def get_total_time(self) -> float:
        """
        Returns the total time the timer has been used, from calling start() to calling stop().
        :return:
        """
        if self.start_time != -1 and self.end_time != -1:
            return (self.end_time - self.start_time).__round__(self.round)
        else:
            print("Timer is not done yet. Cannot calculate total time.")
            return -1

    def get_total_time_string(self, base_duration: int = 300) -> str:
        """
        Return the total time of the timer as a colored string, based on its duration from start to end.
        Based on the base_duration value, the color-thresholds are calculated as follows:\n
        green: base_duration * 1\n
        blue: base_duration * 2\n
        yellow: base_duration * 4\n
        magenta: base_duration * 12\n
        red: anything larger than base_duration * 12\n
        :param base_duration: The duration in seconds that will lead to a green-colored output.
        :return: A colored string.
        """
        interval = self.get_total_time()
        if interval == -1:
            return ""
        else:
            if interval <= base_duration:
                return self.pearl.green(str(interval))
            elif interval <= 2 * base_duration:
                return self.pearl.blue(str(interval))
            elif interval <= 4 * base_duration:
                return self.pearl.yellow(str(interval))
            elif interval <= 12 * base_duration:
                return self.pearl.magenta(str(interval))
            else:
                return self.pearl.red(str(interval))
