from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from utility import utils as u


class DiscreteBaseFunction(ABC):
    """
    This class represents an abstract base class for a discrete base function,
    i.e., a (class) of functions to serve as a basis for a discrete transformation of functions.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the base function. This automatically sets the font used in plots.
        """
        u.latex_font()
        pass

    @abstractmethod
    def plot(self) -> None:
        """
        Plot the base function in the interval [0,1]ᵈ.
        This opens a window with the plot.
        """
        pass

    @abstractmethod
    def evaluate(self, *point):
        """
        Evaluate the base function at point x ∈ [0,1]ᵈ.
        Since the base function is discrete, only the interval containing the point has to be determined.
        This method is largely unused, which is why the intervals are calculated again, instead of storing them.
        :param point: The point to evaluate at.
        :return: The value of the base function at the point.
        """
        pass

    @abstractmethod
    def evaluate_vec(self, *points: np.ndarray) -> np.ndarray:
        """
        Evaluate the base function at a vector of points x = (x₁,...,xₙ)ᵈ, where xᵢ ∈ [0,1]ᵈ ∀i
        IMPORTANT: This assumes x to be ordered uniformly from 0 to 1,
        e.g., with np.linspace(0, 1, samples), where samples is 2ⁿ for some n.
        :param x: The vector of points to evaluate the function at.
        :return: A vector containing the values of the Walsh function at each xᵢ.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the base function, i.e., Walsh, cosine, Gandalf, et cetera.
        :return: The name of the base function as a string.
        """
        pass

    @property
    @abstractmethod
    def plot_title(self) -> str:
        """
        The title which should be displayed on the plot.
        Depending on the base function, different titles make more sense.
        :return: The title of the plot as a string.
        """
        pass

    @property
    @abstractmethod
    def values(self) -> list[float] | list[int] | list[list[float]] | list[list[int]]:
        """
        The values of the base function in each interval.
        As the base function is discrete, storing the values this way does not lose any information.
        :return: A list containing the values of the base function in each interval.
        """
        pass

    @property
    @abstractmethod
    def max_scale(self) -> float:
        """
        The scale of the plot, i.e., a little larger than the maximum value of the function.
        :return: The scale of the plot.
        """
        pass

    @property
    @abstractmethod
    def square_integral(self) -> float:
        """
        The integral of the squared base function on the interval [0,1]ᵈ.
        This is useful for calculating the base functions coefficient.
        :return: The value of the integral of the squared base function on the interval [0,1]ᵈ.
        """
        pass


class DiscreteBaseFunction1D(DiscreteBaseFunction):
    """
    This class represents a one-dimensional base class for a discrete function.
    To implement this class, only the __init__ method has to be implemented
    and the properties: values, intervals, name, plot_title and max_scale have to be assigned.
    """

    @abstractmethod
    def __init__(self, order: int, n: int):
        """
        Initialize the base function.
        :param order: The order of the function; i.e., if all base functions up to 2^n were enumerated, this base function would be the order-th.
        :param n: The number of base functions and number of intervals is 2^n.
        """
        super().__init__()
        if order < 0 or order >= 2 ** n:
            raise ValueError("Order must be between 0 and 2^n -1.")
        elif n <= 0:
            raise ValueError("n must be greater than 0.")
        level, shift = u.get_level_shift(order)
        self.level = level
        self.shift = shift
        self.order = order
        self.n = n
        pass

    def evaluate(self, x: float) -> int | float:
        """
        Evaluate the base function at point x ∈ [0,1].
        Since the base function is discrete, only the interval containing x has to be determined.
        This method is largely unused, which is why the intervals are calculated again, instead of storing them.
        :param x: The point to evaluate at.
        :return: The value of the base function at point x.
        """
        if x < 0 or x > 1:
            raise ValueError("x must be between 0 and 1.")
        h: float = 1 / (2 ** self.n)
        for i in range(2 ** self.n):
            if i * h <= x <= (i + 1) * h:
                return self.values[i]
        return 0

    def evaluate_vec(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the base function at a vector of points x = (x₁,...,xₙ), where xᵢ ∈ [0,1] ∀i
        IMPORTANT: This assumes x to be ordered uniformly from 0 to 1,
        e.g., with np.linspace(0, 1, samples), where samples is 2ⁿ for some n.
        :param x: The vector of points to evaluate the function at.
        :return: A vector containing the values of the base function at each xᵢ.
        """
        ratio = x.shape[0] / len(self.values)

        if int(ratio) - ratio != 0.0:
            # If the ration is not an integer, then the samples are not correct
            raise ValueError("Number of samples must be a power of two and larger or equal to 2^n.")
        # Repeat the values of the base function, so that it fits the 2ⁿ sample points
        long_values = np.repeat(self.values, ratio)
        return long_values

    def plot(self) -> None:
        """
        Plot the base function in the interval [0,1].
        This opens a window with the plot.
        """
        # Fixate the plot size to avoid resizing, especially for k=0.
        plt.xlim(0, 1)
        plt.ylim(-self.max_scale, self.max_scale)
        # {self.name} function {self.order} of {2 ** self.n}
        plt.title(f"{self.plot_title}")

        x = np.linspace(0, 1, 1000)
        y = []
        for v in x:
            y.append(self.evaluate(v))
        plt.plot(x, y, color="blue")
        plt.show()

    @property
    def square_integral(self) -> float:
        return float(np.sum(np.array(self.values) ** 2)) / (2 ** self.n)


class DiscreteBaseFunction2D(DiscreteBaseFunction):
    """
    This class represents a two-dimensional base class for a discrete function.
    To implement this class, only the __init__ method has to be implemented
    and the properties: values, intervals, name and plot_title have to be assigned.
    """

    @abstractmethod
    def __init__(self, order_x: int, order_y: int, n: int):
        """
        Initializes the base function.
        :param order_x: The order of the function;
        i.e., if all base functions up to 2^n per dimension were enumerated, this base function would be the order_x-th.
        :param order_y: The order of the function in y-direction.
        :param n: The number of base functions and number of intervals is (2^n)² = 4^n.
        """
        super().__init__()
        if order_x < 0 or order_x >= 2 ** n or order_y < 0 or order_y >= 2 ** n:
            raise ValueError("Order must be between 0 and 2^n -1.")
        elif n <= 0:
            raise ValueError("n must be greater than 0.")

        self.order_x = order_x
        self.order_y = order_y
        self.n = n
        pass

    @property
    def values(self):
        """
        The values of the base function in each interval.
        As the base function is discrete, storing the values this way does not lose any information.

        IMPORTANT: This method recomputes the matrix of values each time,
        so that the matrix does not need to be stored continuously.
        If this property is often accessed, it might be better to store the matrix instead.
        :return: A list containing the values of the base function in each interval.
        """
        return list(np.outer(np.array(self.values_x), np.array(self.values_y)).T)

    @property
    @abstractmethod
    def values_x(self) -> list[int | float]:
        """
        The values of the base function in x-direction,
        i.e., only the values of the one-dimensional base function in x-direction.
        :return: The values of the base function in x-direction.
        """
        pass

    @property
    @abstractmethod
    def values_y(self) -> list[int | float]:
        """
        The values of the base function in y-direction,
        i.e., only the values of the one-dimensional base function in y-direction.
        :return: The values of the base function in y-direction.
        """
        pass

    @property
    def square_integral(self) -> float:
        return float(np.sum(np.array(self.values) ** 2)) / (2 ** (2 * self.n))

    def evaluate(self, x: float, y: float) -> int | float:
        """
        Evaluate the base function at the point (x,y) ∈ [0,1]².
        Since the base function is discrete, only the interval containing x has to be determined.
        This method is largely unused, which is why the intervals are calculated again, instead of storing them.
        :param x: The x coordinate of the point to evaluate at.
        :param y: The y coordinate of the point to evaluate at.
        :return: The value of the base function at point x.
        """
        h: float = 1 / (2 ** self.n)
        for i in range(2 ** self.n):
            if i * h <= x <= (i + 1) * h:
                x_index = i
            if i * h <= y <= (i + 1) * h:
                y_index = i
        return self.values[x_index][y_index]

    def evaluate_vec(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate the base function at a vector of points, both in x- and y-direction.
        IMPORTANT: This assumes x to be ordered uniformly from 0 to 1,
        e.g., with np.linspace(0, 1, samples), where samples is 2ⁿ for some n.
        :param x: The vector of x-coordinates to evaluate the function at.
        :param y: The vector of y-coordinates to evaluate the function at.
        :return: A vector containing the values of the base function at each (xᵢ,yᵢ).
        """
        if x.shape != y.shape:
            raise ValueError("Shape of x and y must be the same.")
        ratio = x.shape[0] / len(self.values)

        if int(ratio) - ratio != 0.0:
            # If the ration is not an integer, then the samples are not correct
            raise ValueError("Number of samples must be a power of two and larger or equal to 2^n")
        long_values = np.repeat(np.repeat(self.values, ratio, axis=1), ratio, axis=0)
        return long_values

    def plot(self) -> None:
        """
        Plots the 2-dimensional base function on the unit square as a heatmap.
        This opens a window with the plot.
        """
        if self.order_x == 0 == self.order_y:
            plt.imshow(self.values, cmap="gray", interpolation="none", extent=[0, 1, 0, 1], vmin=-self.max_scale,
                       vmax=self.max_scale)
        else:
            plt.imshow(self.values, cmap="gray", interpolation="none", extent=[0, 1, 0, 1], vmin=-self.max_scale,
                       vmax=self.max_scale)
        plt.colorbar(label="Value")
        # Not advisable for n > 3
        # plt.xticks(np.linspace(0, 1, 2 ** self.n + 1), labels=self.intervals)
        # plt.yticks(np.linspace(0, 1, 2 ** self.n + 1), labels=self.intervals)
        plt.title(self.plot_title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()
