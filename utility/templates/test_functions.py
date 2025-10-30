from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import brentq

from utility import utils as u


class TestFunctionType(Enum):
    """
    Define the type of the test function to improve outputs.
    """
    FUNCTION = "function"
    IMAGE = "image"


class TestFunction(ABC):
    """
    This class represents a parent class for both 1D and 2D test functions to be used for the Walsh transform.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the function, i.e., f(x) = x^2. This is a property of the object.
        This uses LaTeX syntax to allow for better display in plots.
        """
        pass

    @property
    def name_cli(self) -> str:
        """
        Name of the function, i.e., f(x) = x^2. This is a property of the object.
        This does not use LaTeX syntax to allow display in the command line.
        """
        return self.name

    @abstractmethod
    def evaluate(self, *point) -> float | np.ndarray:
        """
        Evaluate the n-dimensional test function at an n-dimensional point, or an array of n-dimensional points.
        :param point: The n-dimensional point(s) to evaluate the test function at.
        :return: The value of the function at the given point(s), so f(point(s)).
        """
        pass

    @abstractmethod
    def evaluate_integral(self, *point: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluates the indefinite integral of the function at point x ∈ [0,1].
        :param point: The n-dimensional point to evaluate at.
        :return: The value of the integral of the function at x.
        """
        pass

    @abstractmethod
    def get_equidistant_data_points(self, n: int) -> list[tuple[float, float]] | list[list[tuple[float, float, float]]]:
        """
        Get n higher-dimensional equidistant data points per dimension.
        :param n: The number of data points.
        :return: n higher-dimensional equidistant data points.
        """
        pass

    @abstractmethod
    def sample(self, samples: int = 0) -> np.ndarray:
        """
        Sample the n-dimensional function on an n-dimensional regular grid to generate a matrix representation of it.
        This representation can be used to plot the function.
        :param samples: The number of samples to generate per dimension.
        :return: Values of the function on the n-dimensional grid.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Plot the n-dimensional test function on the unit interval [0,1]ⁿ.
        This opens a window with a plot.
        """
        pass

    @abstractmethod
    def evaluate_integral_squared(self, *point: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the indefinite integral of the squared function at x ∈ [0,1].
        :param x: The point at which to evaluate the integral of the squared function at.
        :return: The value of the integral of the squared function at x.
        """
        pass

    @abstractmethod
    def l2_norm_square(self):
        """
        Calculate the squared L² norm of the function, i.e., the L² scalar product ⟨f,f⟩=∫ f^2 of the function with itself.
        :return: The squared L² norm of the function.
        """
        pass


class TestFunction1D(TestFunction):
    """
    Represents a 1D test function to test in Walsh transformation.
    A 1D test function needs to implement the evaluate() method and set the name property.
    Then it can be used to generate data points, at which the function is evaluated.
    """

    @abstractmethod
    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluates the function at point x ∈ [0,1].
        :param x: The point to evaluate at.
        :return: The value of the function at x.
        """
        pass

    @abstractmethod
    def evaluate_integral(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluates the integral of the function at point x ∈ [0,1].
        :param x: The point to evaluate at.
        :return: The value of the integral of the function at x.
        """
        pass

    def get_zero_crossings(self, n: int, c: float = 0, a: float = 0, b: float = 1) -> list[float]:
        """
        Find zero crossings of the function - c in the interval [a,b], i.e., f(x) - c = 0, x ∈ [a,b].
        :param n: The number of samples per interval.
        :param c: A constant to be subtracted from f.
        :param a: The start of the interval.
        :param b: The end of the interval.
        :return: A list of zero crossings.
        """
        xs: np.array = np.linspace(a, b, n + 1)
        f: type[callable] = lambda x: self.evaluate(x) - c
        zero_crossings: list[float] = []
        ys = self.evaluate(xs) - c
        for i in range(len(xs) - 1):
            if np.isclose(ys[i], 0.0, atol=1e-12, rtol=1e-9):
                zero_crossings.append(xs[i])
            if ys[i] * ys[i + 1] < 0:  # A sign change occurred
                zero_crossings.append(brentq(f, xs[i], xs[i + 1]))
        # Check last entry separately
        if np.isclose(ys[-1], 0.0, atol=1e-12, rtol=1e-9):
            zero_crossings.append(xs[-1])

        zeros = np.asarray(zero_crossings, dtype=float)
        zeros = zeros[(zeros > a + 1e-14) & (zeros < b - 1e-14)]
        if zeros.size == 0:
            return []
        zeros.sort()

        unique: list[float] = [float(zero_crossings[0])]
        for zero in zeros[1:]:
            if not np.isclose(zero, unique[-1], atol=1e-12, rtol=1e-9):
                unique.append(float(zero))

        return unique

    def get_equidistant_data_points(self, n: int) -> list[tuple[float, float]]:
        """
        Create n equidistant data points in the interval [0,1].
        A data point consists of its coordinate and its value, i.e., (x, f(x)).
        FOR ONLY Y = f(X) VALUES USE ``sample()``.
        :param n: The number of data points.
        :return: A list of data points [(x_i, f(x_i)].
        """
        data_points = []
        for i in range(n):
            data_points.append((i / n, self.evaluate(i / n)))
        return data_points

    def sample(self, samples: int = 1024) -> np.ndarray:
        """
        Sample the function on the interval [0,1], with the given number of samples.
        :param samples: The number of samples. Default: 1024
        :return: Values of the function on the unit interval.
        """
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)

        x = np.linspace(0, 1, samples)
        f = self.evaluate(x)
        return f

    def plot(self, samples: int = 256) -> None:
        x = np.linspace(0, 1, samples)
        title: str = f"$\\displaystyle {self.name}$"
        # fig = plt.figure(figsize=(6,6))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, self.evaluate(x), color="orange")
        plt.title(title)
        # plt.savefig("exp1d.pdf", dpi=400,bbox_inches="tight",pad_inches=0)
        plt.show()

    def l2_norm_square(self):
        return self.evaluate_integral_squared(1) - self.evaluate_integral_squared(0)


class TestFunction2D(TestFunction):
    """
    Represents a 2D test function to test in Walsh transformation.
    A 2D test function needs to implement the ``evaluate()`` method and set the ``name`` property.
    Then it can be used to generate data points, at which the function is evaluated.
    """

    @abstractmethod
    def evaluate(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluates the function at point(s) (x,y) ∈ [0,1]².
        :param x: The x-coordinate of the point(s) to evaluate at.
        :param y: The y-coordinate of the point(s) to evaluate at.
        :return: The value of the function at (x,y).
        """
        pass

    @abstractmethod
    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluates the integral of the function at point x ∈ [0,1].
        :param x: The x-coordinate of the point(s) to evaluate at.
        :param y: The y-coordinate of the point(s) to evaluate at.
        :return: The value of the integral of the function at (x,y).
        """
        pass

    def get_equidistant_data_points(self, n: int) -> list[list[tuple[float, float, float]]]:
        """
        Create an equidistant grid of n² data points in the unit square [0,1].
        A data point consists of its coordinates and its value, i.e., (x,y,f(x,y)).
        :param n: The number of data points per direction.
        :return: A list of data points [(x_i, y_i, f(x_i, y_i))].
        """
        # n × n grid of data points
        data_points = [[0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                data_points[i][j] = (i / n, j / n, self.evaluate(i / n, j / n))
        return data_points

    def sample(self, samples: int = 256) -> np.ndarray:
        """
        Sample the function on an equidistant grid on the unit square [0,1], with the given number of samples per dimension.
        :param samples: The number of samples per dimension. Default: 256
        :return: The value of the function on the equidistant grid.
        """
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)

        x = np.linspace(0, 1, samples)
        y = np.linspace(0, 1, samples)
        X, Y = np.meshgrid(x, y)

        return self.evaluate(X, Y)

    def plot(self, samples: int = 256) -> None:
        x = np.linspace(0, 1, samples)
        y = np.linspace(0, 1, samples)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, Y, self.evaluate(X, Y), cmap='viridis')
        ax.view_init(elev=30, azim=-135)
        plt.title(f"$\\displaystyle {self.name}$")
        # plt.savefig("x22d.pdf", dpi=400, bbox_inches="tight", pad_inches=0)
        plt.show()

    def l2_norm_square(self) -> float:
        return (self.evaluate_integral_squared(1, 1)
                - self.evaluate_integral_squared(1, 0)
                - self.evaluate_integral_squared(0, 1)
                + self.evaluate_integral_squared(0, 0))


class Image(TestFunction):
    """
    This class represents an image.
    The image is sampled to form a square matrix with entries denoting the greyscale pixel value.
    """

    def __init__(self, path: str, resolution: int) -> None:
        """
        Initializes an Image. The path is used to sample the image and define its name.
        :param path: The path to the image relative to the working directory.
        :param resolution: The resolution of the image per dimension, i.e., resolution = 256 for a 256 × 256 image.
        """
        self.path = path
        self._name = path.split("/")[-1]
        self.resolution = resolution

    @property
    def name(self) -> str:
        """
        Name of the image as given by the path ending.
        :return: The name of the image.
        """
        return self._name

    def evaluate(self):
        """
        Not implemented. Image is sampled directly in ``get_data_points()``.
        :raise NotImplementedError:
        """
        raise NotImplementedError("Images do not have to be evaluated.")

    def evaluate_integral(self):
        """
        Not implemented. Image does not have an integral.
        :return: NotImplemented Error:
        """
        raise NotImplementedError("Images do not have to be evaluated.")

    def get_equidistant_data_points(self, n) -> np.ndarray:
        """
        Returns an n × n matrix with entries denoting the greyscale pixel value.
        :param n: The number of data points per direction.
        :return: A matrix representing the image. The origin is in the upper-left corner.
        """
        return u.sample_image(self.path, n)

    def sample(self, samples: int = -1) -> np.ndarray:
        """
        Samples the image with the given number of samples per dimension.
        :param samples: The number of samples per dimension. Default: the resolution of the image.
        :return:
        """
        if samples == -1:
            samples = self.resolution
        if samples < 0:
            raise ValueError("The number of samples must be positive.")
        return np.array(u.sample_image(self.path, samples))

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.sample(self.resolution), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"${self.name}$")
        plt.show()

    def evaluate_integral_squared(self):
        """
        Not implemented. Image does not have an integral.
        To calculate the L² norm, the function l2_norm() is still available.
        """
        raise NotImplementedError("Images do not have to be evaluated.")

    def l2_norm_square(self):
        """
        Calculate the squared L² norm of the function, i.e., the L² scalar product ⟨f,f⟩=∫ f^2 of the function with itself.
        Even though images do not have an integral which can be evaluated,
        the L² norm can still be calculated for the image as a step function.
        :return: The squared L² norm of the function.
        """
        return np.sum(self.sample(samples=self.resolution) ** 2) / (self.resolution ** 2)
