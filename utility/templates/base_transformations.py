from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utility import utils as u
from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.test_functions import TestFunction, Image, TestFunctionType, TestFunction2D, TestFunction1D


class Transformation(ABC):
    """
    This class represents a base for transformations, both in 1D and 2D.
    """

    @abstractmethod
    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        """
        Initializes a transformation for the given test function by initializing 2^n base functions.
        :param n: 2^n is the number of Walsh functions.
        :param function: The test function to transform.
        """
        if n <= 0:
            raise ValueError("n must be greater than 0.")
        self.n = n
        self.function = function
        u.latex_font()

        if boundary_n == -1 or boundary_n < n:
            self.boundary_n = n
        else:
            self.boundary_n = boundary_n
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the transformation, i.e., Wavelet-, Walsh- or Werewolf-transformation, et cetera.
        :return: The name of the transformation.
        """
        pass

    @property
    @abstractmethod
    def base_functions(self) -> list[DiscreteBaseFunction1D] | list[list[DiscreteBaseFunction2D]]:
        """
        A list containing all base functions for the transformation. For n-dimensions, they are always stored in an n-dimensional grid-like list.
        :return: The list of base functions.
        """
        pass

    @property
    @abstractmethod
    def base_square_integrals(self):
        """
        A list containing a value for each base function,
        where the value is equal to the integral of the squared base function in the unit square [0,1]ᵈ.
        :return: A list containing the value of the squared base function integral.
        """

    @abstractmethod
    def plot_base_matrix(self, cli: bool = False, fig: Figure = None, index: int = -1) -> None:
        """
        Plot a matrix representation of all base functions.
        This opens a window with the plot.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the parameters 'fig' and 'index' need to be given as well.
        :param fig: The figure in which the base matrix is plotted.
        :param index: The index where in the plot the base matrix is plotted.
        """
        pass

    @abstractmethod
    def get_coefficients_integration_orthonormal(self) -> np.ndarray:
        """
        Transform the test-function to a linear combination of base functions by calculating the coefficients.
        This is achieved by integrating the test function and multiplying with the base functions.
        IMPORTANT: This assumes the base functions to be orthonormal.
        For non-orthonormal base functions, use get_coefficients_integration().
        :return: The coefficients of the base functions.
        """
        pass

    def get_coefficients_integration(self) -> np.ndarray:
        """
        Transform the test-function to a linear combination of base functions by calculating the coefficients.
        This is achieved by integrating the test function and multiplying with the base functions,
        as well as dividing by the square integral of the base functions.
        :return: The coefficients of the base functions.
        """
        coefficients = self.get_coefficients_integration_orthonormal()
        coefficients /= np.array(self.base_square_integrals).T.flatten()
        return coefficients

    @abstractmethod
    def sample_transform(self, coefficients: np.ndarray, samples: int = 256) -> np.ndarray:
        """
        Sample the transformed function to generate samples for a plot.
        :param coefficients: The coefficients of the base functions.
        :param samples: The number of samples to generate per dimension.
        :return: The values of the transformed function at the sample points.
        """
        pass

    def discard_coefficients_absolute(self, coefficients: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Discard all coefficients smaller than some epsilon.
        All coefficients that are smaller than epsilon are set to 0.
        :param coefficients: The coefficients to update.
        :param epsilon: The threshold.
        :return: The updated coefficients.
        """
        modified_coefficients = np.copy(coefficients)
        # All coefficients smaller than epsilon are set to zero, the rest are left unchanged.
        modified_coefficients[abs(coefficients) < epsilon] = 0
        return modified_coefficients

    @abstractmethod
    def discard_coefficients_sparse_grid(self, coefficients: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Discard all coefficients whose base function's level sum across all dimensions is larger than epsilon.
        This corresponds to the sparse grid selection of coefficients. In 1-D, this does nothing.
        :param coefficients: The coefficients to update.
        :param epsilon: The maximum sum of levels above which the coefficients will be discarded.
        :return: The updated coefficients.
        """
        pass

    def discard_coefficients_relative(self, coefficients: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Discard all coefficients smaller than some epsilon, relative to the maximum coefficient.
        All coefficients that are smaller than epsilon times the largest coefficient are set to 0.
        :param coefficients: The coefficients to update.
        :param epsilon: The threshold factor.
        :return: The updated coefficients.
        """
        modified_coefficients = np.copy(coefficients)
        # Determine maximum coefficient
        max_coefficient = np.max(np.abs(coefficients))
        # All coefficients that are smaller than ε * max_coef are set to zero
        modified_coefficients[abs(coefficients) < epsilon * max_coefficient] = 0
        return modified_coefficients

    def discard_coefficients_percentage(self, coefficients: np.ndarray, epsilon: float, cli: bool = False) -> (
            tuple[np.ndarray, float] | np.ndarray):
        """
        Discard the smallest epsilon percentage of the coefficients.
        The discarded coefficients are set to 0.
        :param coefficients: The coefficients to update.
        :param epsilon: The percentage of coefficients to discard.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the threshold will be returned as well. Defaults to False.
        :return: The updated coefficients. If cli=True, also return the threshold of discarded coefficients.
        """
        if epsilon >= 100:
            if cli:
                return np.zeros_like(coefficients), coefficients.max()
            else:
                return np.zeros_like(coefficients)
        elif epsilon < 0:
            raise ValueError("The percentage of coefficients to discard is negative.")

        modified_coefficients = np.copy(coefficients)
        # Absolute number of coefficients to keep, cast to integer
        n_keep = int(len(coefficients) * (epsilon / 100))

        # Scale coefficient
        scale_coefficient = coefficients[0]

        # Get the value of the last element to keep / first to discard
        threshold: float = np.partition(abs(coefficients), n_keep)[n_keep]
        modified_coefficients[abs(coefficients) < threshold] = 0

        # Keep scale coefficient
        modified_coefficients[0] = scale_coefficient
        if cli:
            return modified_coefficients, threshold
        else:
            return modified_coefficients

    @abstractmethod
    def plot_error_absolute(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "", cli: bool = False, fig: Figure = None, index: int = -1,
                            vmin: float = -1, vmax: float = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        """
        Plots the absolute error of the transformation with the given coefficients and the test function.
        This opens a window with the plot.
        :param transform_values: The values of the transformation at the sample points.
        :param function_values: The values of the test function at the sample points.
        :param subtitle: A subtitle to be displayed on the plot,
        e.g., for information about the transformation or the test function, et cetera.
        By default, no subtitle is displayed.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the parameters 'fig', 'index', 'vmin' and 'vmax' need to be given as well.
        :param fig: The figure in which the error is plotted.
        :param index: The index where in the plot the error is plotted.
        :param vmin: The minimum value of the plot.
        :param vmax: The maximum value of the plot.
        :return: None, if not shown in cli;
        an AxesImage and Axes object, if shown in the cli, to be able to create a colorbar
        """
        pass

    @abstractmethod
    def plot_transformation(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "", original: bool = False,
                            cli: bool = False, fig: Figure = None, index: int = -1,
                            vmin: float = -1, vmax: float = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        """
        Plots the transformation of a function to a linear combination of base functions with the given values.
        This opens a window with the plot.
        :param transform_values: The values of the base transformation at the sample points.
        :param function_values: The values of the test function at the sample points.
        :param subtitle: A subtitle to be displayed on the plot,
        e.g., for information about the transformation or the test function, et cetera.
        By default, no subtitle is displayed.
        :param original: A boolean indicating whether to display the original function or not.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the parameters 'fig', 'index', 'vmin' and 'vmax' need to be given as well.
        :param fig: The figure in which the coefficients are plotted.
        :param index: The index where in the plot the coefficients are plotted.
        :param vmin: The minimum value of the plot.
        :param vmax: The maximum value of the plot.
        :return: None, if not used in cli;
        an AxesImage and Axes object, if shown in the cli, to be able to create a colorbar
        """
        pass

    @abstractmethod
    def evaluate(self, coefficients: np.ndarray, *point: float | tuple[float, float]) -> float | np.ndarray:
        """
        Evaluate the transformation defined by the given coefficients at a given point x.
        :param point: The point to evaluate the transformation at; either one- or two-dimensional.
        :param coefficients: The coefficients of the base functions.
        :return: The value of the transformed function at x.
        """
        pass

    @abstractmethod
    def plot_coefficients(self, coefficients: np.ndarray, subtitle: str = "", sorted: bool = False,
                          first_n: int = 0, cli: bool = False, fig: Figure = None, index: int = -1,
                          vmin: float = -1, vmax: float = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        """
        Plot the coefficients of the transformation, with double logarithmic scaling.
        This opens a window with the plot.
        :param coefficients: The coefficients of the base functions.
        :param subtitle: A subtitle to be displayed on the plot,
        e.g., for information about the transformation or the test function, et cetera.
        By default, no subtitle is displayed.
        :param sorted: A boolean indicating whether the coefficients should be sorted. Defaults to False.
        :param first_n: If a value larger than 0 is assigned, only the first n coefficients will be displayed.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the parameters 'fig', 'index', 'vmin' and 'vmax' need to be given as well.
        :param fig: The figure in which the coefficients are plotted.
        :param index: The index where in the plot the coefficients are plotted.
        :param vmin: The minimum value of the plot.
        :param vmax: The maximum value of the plot.
        :return: None, if not shown in cli;
        an AxesImage and Axes object, if shown in the cli, to be able to create a colorbar.
        """
        pass

    def get_squared_l2_error(self, coefficients: np.ndarray) -> float:
        """
        Calculate the squared L² norm of the error of the approximation given by the coefficients.
        The L² norm of the error (for orthonormal transformations) is defined as: ‖E‖²= ‖f‖² - ∑α²
        :param coefficients: The coefficients of the base functions.
        :return: The squared L² norm of the error.
        """
        f_norm = self.function.l2_norm_square()
        coef_sum = np.sum(coefficients ** 2)
        return f_norm - coef_sum

    @abstractmethod
    def get_l1_error(self, coefficients: np.ndarray, function_values: np.ndarray = None,
                     transformation_values: np.ndarray = None) -> float:
        """
        Calculate the L1 norm of the transformation of the approximation given by the coefficients.
        The L¹ norm of the error is defined as ‖E‖¹ = ∫ |f-∑αφ|, which can be split into multiple integrals,
        for each of which ∑αφ = c is constant.
        The function values are only needed if the transformed function is an image.
        By default, transformation_values and function_values will be calculated if needed.
        However, to improve performance, it might be beneficial to pass them as an argument.
        Note that the transformation should be sampled exactly 2^n times per dimension,
        so that no information is lost or redundant computation is performed.
        Additionally, the shapes of the function values and the transformation values need to match.
        :param coefficients: The coefficients of the base functions.
        :param function_values: The values of the test function at the sample points.
        :param transformation_values: The values of the transformation at the sample points.
        :return: The L¹ norm of the error.
        """
        pass

    @abstractmethod
    def get_linf_error(self, coefficients: np.ndarray, function_values: np.ndarray = None,
                       transformation_values: np.ndarray = None, samples: int = 1024) -> float:
        """
        Approximate the L∞ norm of the error of the approximation given by the coefficients.
        It is L∞ = ‖E‖∞ = sup |f-∑αφ|. By sampling |f-∑αφ| samples times per dimension, a good estimate will be found.
        :param coefficients: The coefficients of the base functions.
        :param function_values: The values of the test function at the sample points.
        :param transformation_values: The values of the transformation at the sample points.
        :param samples: The number of samples to use.
        :return: The L∞ norm of the error.
        """
        pass


class Transformation1D(Transformation):
    """
    This class represents an abstract base class for one-dimensional transformations.
    Methods such as calculating coefficients, plotting errors et cetera are implemented.
    To implement a transformation, the base functions need to be orthonormal to each other.
    Then, simply the __init__ method needs to be implemented.
    The properties name, base_functions and base_values have to be defined as well.
    """

    @abstractmethod
    def __init__(self, n: int, function: TestFunction1D, boundary_n: int = -1):
        """
        Initialize the transformation for the given TestFunktion with 2^n base functions.
        The optional parameter boundary_n is only effective for Walsh transformations
        and defines up to which 2^boundary_n the boundaries will be calculated,
        after which the used base functions are determined.
        :param n: 2^n is the number of base functions.
        :param function: The function to be transformed.
        :param boundary_n: 2^boundary_n is the number of base functions that will be considered
        to be used in the transformation. Then, the 2^n best base functions are actually used.
        This is only effective for Walsh transformations.
        """
        super().__init__(n, function, boundary_n)
        pass

    @property
    def base_square_integrals(self):
        return [self.base_functions[i].square_integral for i in range(2 ** self.n)]

    def plot_base_matrix(self, cli: bool = False, fig: Figure = None, index: int = -1,
                         outer_grid: GridSpec = None) -> None:
        """
        Plots the base-function-matrix, i.e., the values of every base function in the interval [0,1].
        The entries in line i represent one base-function function of order i.
        This opens a window with the plot.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the parameters 'fig' and 'index' need to be given as well.
        :param fig: The figure in which the base matrix is plotted.
        :param index: The index where in the plot the base matrix is plotted.
        :param outer_grid: This parameter is not used.
        """
        title: str = f"{self.name}-functions for $n={self.n}$"
        y = [self.base_functions[i].values for i in range(2 ** self.n)]
        if not cli:
            plt.figure(figsize=(6, 6))
            plt.imshow(y, cmap="gray", aspect="auto")
            plt.title(title)
            plt.show()
        # cli
        else:
            ax = fig.add_subplot(1, 3, index)
            ax.imshow(y, cmap="gray", aspect="auto")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

    def get_coefficients_integration_orthonormal(self) -> np.ndarray:
        # It is αᵢ = ∫ gfᵢ / ∫ f²ᵢ
        # However, for an orthonormal function fᵢ, the integral of its square is equal to one.
        # This means that the coefficient is equal to ∫ gfᵢ
        coefficients: list[float] = []
        integrals: list[float] = []

        # This might differ per transformation,
        # depending on whether it uses base functions with 2^boundary_n or 2^n intervals.
        length: int = 2 ** max(self.n, self.boundary_n)
        # Since the values of the base function differ in each interval, split the integral up into all intervals
        # Recomputing the interval borders is probably better than storing them forever
        for i in range(length):
            integral_i = (self.function.evaluate_integral((i + 1) / length)
                          - self.function.evaluate_integral(i / length))
            integrals.append(integral_i)

        # Select base function. There must only be 2^n base functions, so this cannot change per transformation.
        for i in range(2 ** self.n):
            base_integral = np.sum(np.array(self.base_functions[i].values) * np.array(integrals))
            coefficients.append(base_integral)

        return np.array(coefficients)

    def sample_transform(self, coefficients: np.ndarray, samples: int = 1024) -> np.ndarray:
        """
        Samples the transformation, which is defined by the given coefficients and the base functions.
        :param coefficients: The coefficients of the base functions.
        :param samples: The number of samples to use. By default, 1024 samples are used.
        For fast evaluation, the number of samples will always be set to a power of two.
        If more base functions than 1024 are used, the number of samples will be set to the number of base functions.

        :return: An array containing the values of the transformation.
        """
        # If samples is not a power of two, then eval_vec will not work correctly
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)
        # If samples is less than 2ⁿ, then eval_vec will throw an error
        # Increase it to 2ⁿ to avoid this
        if samples < 2 ** self.boundary_n:
            samples = 2 ** self.boundary_n
        # Depending on the resolution of the transformation, this can be different per transformation.
        # Thus, this case has to be checked as well.
        elif samples < len(self.base_functions[0].values):
            samples = len(self.base_functions[0].values)
        x = np.linspace(0, 1, samples)
        y: np.array = np.zeros_like(x)
        # Amass the values of the transformation, by multiplying each coefficient by its base function and its values
        for i, phi in enumerate(self.base_functions):
            y += coefficients[i] * phi.evaluate_vec(x)

        return np.array(y)

    def plot_error_absolute(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "", cli: bool = False, fig: Figure = None, index: int = -1,
                            vmin: int = -1, vmax: int = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        # Only append a new line if the subtitle is not empty.
        subtitle = "\n" + subtitle if subtitle else ""
        title: str = (f"Absolute error of {self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$") + subtitle

        if not cli:
            plt.figure(figsize=(6, 6))
            x = np.linspace(0, 1, transform_values.size)
            y = abs(transform_values - function_values)
            plt.plot(x, y, color="blue")
            plt.title(title)
            plt.xlim(0, 1)
            plt.show()
        # cli
        else:
            ax = fig.add_subplot(1, 3, index)
            plt.ylim(vmin, vmax)
            x = np.linspace(0, 1, transform_values.size)
            y = abs(transform_values - function_values)
            im = ax.plot(x, y, color="blue")
            plt.title(title)
            plt.xlim(0, 1)
            return im, ax
        return None, None

    def plot_transformation(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "", original: bool = False,
                            cli: bool = False, fig: Figure = None, index: int = -1,
                            vmin: float = -1, vmax: float = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        # Only append a new line if the subtitle is not empty.
        subtitle = "\n" + subtitle if subtitle else ""
        title: str = (f"{self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$") + subtitle
        if not cli:
            plt.figure(figsize=(4.5, 4.5), constrained_layout=True)
            x = np.linspace(0, 1, transform_values.size)
            plt.plot(x, transform_values, color="blue")
            plt.plot(x, function_values, color="orange")
            plt.title(title)
            plt.show()
        # cli
        else:
            ax = fig.add_subplot(1, 3, index)
            x = np.linspace(0, 1, transform_values.size)
            im = ax.plot(x, transform_values, color="blue")
            im = ax.plot(x, function_values, color="orange")
            ax.set_title(title)
            return im, ax
        return None, None

    def evaluate(self, coefficients: np.ndarray, *point: float | tuple[float, float]) -> float | np.ndarray:
        if len(point) != 1:
            raise ValueError("Point must be one-dimensional")
        y_i = 0
        x = point[0]
        for i, f in enumerate(self.base_functions):
            y_i += coefficients[i] * f.evaluate(x)
        return y_i

    def plot_coefficients(self, coefficients: np.ndarray, subtitle: str = "", sorted: bool = False,
                          first_n: int = 0, cli: bool = False, fig: Figure = None, index: int = -1,
                          vmin: float = -1, vmax: float = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        subtitle = "\n" + subtitle if subtitle else ""
        title: str = (f"Coefficients of {self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$\n"
                      f"with {2 ** self.n} {self.name} functions" + subtitle)
        # Do not update coefficients[] object
        coefficients_to_plot = coefficients.copy()
        if not cli:
            plt.figure(figsize=(4.5, 4.5), constrained_layout=True)
            # Plot only the first n coefficients
            if first_n > 0:
                coefficients_to_plot = coefficients_to_plot[:first_n]
                subtitle += f"\nshowing {first_n} coefficients"
            if sorted:
                # Sort coefficients in descending order by absolute value
                coefficients_to_plot[::-1] = np.sort(abs(np.array(coefficients_to_plot)))
                subtitle += "\nsorted in descending order"

            plt.plot(range(1, coefficients_to_plot.shape[0] + 1), abs(coefficients_to_plot), "--",
                     color="blue", marker="o")

            plt.yscale("log")
            plt.xscale("log")
            plt.title(title)
            plt.show()
        # cli
        else:
            ax = fig.add_subplot(1, 3, index)
            if first_n > 0:
                coefficients_to_plot = coefficients_to_plot[:first_n]
                subtitle += f"\nshowing {first_n} coefficients per dimension"
            if sorted:
                coefficients_to_plot[::-1] = np.sort(abs(np.array(coefficients_to_plot)))
                subtitle += "\nsorted in descending order"
            im = ax.plot(range(1, coefficients_to_plot.shape[0] + 1), abs(coefficients_to_plot),
                         color="blue", marker="o", markersize=5)
            plt.ylim(vmin, vmax)
            plt.xscale("log")
            plt.yscale("log")

            title: str = (f"Coefficients of {self.name} transformation of\n"
                          f"$\\displaystyle {self.function.name}$\n"
                          f"with {2 ** self.n} {self.name} functions per dimension" + subtitle)
            ax.set_title(title)

            return im, ax
        return None, None

    def get_l1_error(self, coefficients: np.ndarray, function_values: np.ndarray = None,
                     transformation_values: np.ndarray = None) -> float:
        if transformation_values is None:
            transformation_values = self.sample_transform(coefficients, samples=2 ** self.boundary_n)
        h: float = 1 / 2 ** self.boundary_n  # Interval width
        total: float = 0
        samples: int = u.determine_samples(self.boundary_n)
        for i in range(2 ** self.boundary_n):
            c: float = transformation_values[i]
            a: float = i * h
            b: float = (i + 1) * h
            zeros = self.function.get_zero_crossings(samples, c=c, a=a, b=b)

            if len(zeros) > 1:
                total += abs(self.function.evaluate_integral(zeros[0])
                             - self.function.evaluate_integral(a)
                             - c * (zeros[0] - a))
                for j in range(len(zeros) - 1):
                    total += abs(self.function.evaluate_integral(zeros[j + 1])
                                 - self.function.evaluate_integral(zeros[j])
                                 - c * (zeros[j + 1] - zeros[j]))
                total += abs(self.function.evaluate_integral(b)
                             - self.function.evaluate_integral(zeros[-1])
                             - c * (b - zeros[-1]))
            elif len(zeros) == 1:
                total += abs(self.function.evaluate_integral(b)
                             - self.function.evaluate_integral(zeros[0])
                             - c * (b - zeros[0]))
                total += abs(self.function.evaluate_integral(zeros[0])
                             - self.function.evaluate_integral(a)
                             - c * (zeros[0] - a))
            else:
                total += abs(self.function.evaluate_integral(b)
                             - self.function.evaluate_integral(a)
                             - c * h)
        return total

    def get_linf_error(self, coefficients: np.ndarray, function_values: np.ndarray = None,
                       transformation_values: np.ndarray = None, samples: int = 1024) -> float:
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)
        elif samples < 2 ** self.boundary_n:
            samples = 2 ** self.boundary_n
        x = np.linspace(0, 1, samples)
        if transformation_values is None:
            transformation_values = self.sample_transform(coefficients, samples=samples)
        if function_values is None:
            function_values = self.function.evaluate(x)
        if transformation_values.shape != function_values.shape:
            transformation_values = self.sample_transform(coefficients, samples=samples)
            function_values = self.function.evaluate(x)

        difference = abs(transformation_values - function_values)
        return max(difference)

    def discard_coefficients_sparse_grid(self, coefficients: np.ndarray, epsilon: float) -> np.ndarray:
        return coefficients


class Transformation2D(Transformation):
    """
    An abstract base class for two-dimensional transformations.
    """

    @abstractmethod
    def __init__(self, n: int, function: TestFunction2D | Image, boundary_n: int = -1):
        super().__init__(n, function, boundary_n)
        if isinstance(function, Image):
            self.type = TestFunctionType.IMAGE
        elif isinstance(function, TestFunction2D):
            self.type = TestFunctionType.FUNCTION
        else:
            raise NotImplementedError(f"Trying to instantiate a {self.name} transformation for "
                                      f"{function.__class__.name}, which is not supported.")
        pass

    @property
    def base_values(self) -> list[list[float]] | list[list[int]] | list[list[list[float]]] | list[list[list[int]]]:
        return [[self.base_functions[i][j].values for j in range(2 ** self.n)]
                for i in range(2 ** self.n)]

    @property
    def base_square_integrals(self):
        return [[self.base_functions[i][j].square_integral for j in range(2 ** self.n)]
                for i in range(2 ** self.n)]

    def plot_base_matrix(self, cli: bool = False, fig: Figure = None, index: int = -1,
                         outer_grid: GridSpec = None) -> None:
        """
        Plots the base-matrix, i.e., a matrix containing "small" matrices of each base-functions values in the unit square.
        The functions grow first in y-direction, before growing in x-direction.
        This opens a window with the plot.
        :param cli: A boolean indicating whether the command is used in the cli.
        If so, then the parameters 'fig', 'index' and 'outer_grid' need to be given as well.
        :param fig: The figure in which the base matrix is plotted.
        :param index: The index where in the plot the base matrix is plotted.
        :param outer_grid: The outer grid of the plot, i.e., hte .
        :return: None
        """
        title: str = f"{self.name}-functions for $n={self.n}$ per dimension."
        if not cli:
            # figsize(x,y) are dimensions of the figure in inches.
            fig, axes = plt.subplots(2 ** self.n, 2 ** self.n, figsize=(10, 10))

            for i in range(2 ** self.n):
                for j in range(2 ** self.n):
                    vmax = self.base_functions[i][j].max_scale
                    vmin = -vmax
                    if i == j == 0:
                        vmin = -1
                        vmax = 1
                    ax = axes[j][i]
                    ax.imshow(self.base_values[i][j], cmap="gray", vmin=vmin, vmax=vmax)
                    # No ticks to improve readability
                    ax.set_xticks([])
                    ax.set_yticks([])
            # Maybe add a colorbar? The bar is not that interesting when there are only two values, though.
            plt.title(title)
            plt.tight_layout()
            plt.show()
        # cli
        else:
            if self.n > 3:
                print("This method is not advisable for n > 3.")
                return
            # First row, index-th column
            spec = outer_grid[0, index - 1]
            inner_grid = spec.subgridspec(2 ** self.n, 2 ** self.n, wspace=0.01, hspace=0.01)
            for i in range(2 ** self.n):
                for j in range(2 ** self.n):
                    ax_ij = fig.add_subplot(inner_grid[j, i])
                    scale = self.base_functions[i][j].max_scale
                    ax_ij.imshow(self.base_values[i][j], cmap="gray", vmin=-scale, vmax=scale)
                    ax_ij.set_xticks([])
                    ax_ij.set_yticks([])

    def get_function_integral_matrix(self) -> list[list[float]]:
        integrals: list[list[float]] = []
        # When using boundary_n, a base function can have more values
        length: int = 2 ** max(self.n, self.boundary_n)

        if self.type == TestFunctionType.FUNCTION:
            # Calculate all integrals first
            for i in range(length):
                integrals_i: list[float] = []
                for j in range(length):
                    integrals_i.append((
                            self.function.evaluate_integral((j + 1) / length, (i + 1) / length)
                            - self.function.evaluate_integral(j / length, (i + 1) / length)
                            - self.function.evaluate_integral((j + 1) / length, i / length)
                            + self.function.evaluate_integral(j / length, i / length)))
                integrals.append(integrals_i)

        elif self.type == TestFunctionType.IMAGE:
            # Images are not affected by boundary_n.
            # For an image, each subintegral is equal to the function value, times the interval area
            # Since the integrals are always from c * 1/2^n to (c+1) * 1/2^n, the interval size (area) is (1/2^n)^2 = 1/4^n
            # This means that scaling all image values by 1/4^n is equal to their integral
            function_values = self.function.get_equidistant_data_points(2 ** self.n)
            integrals = function_values * (1 / (4 ** self.n))

        # else-case cannot happen, since initializing such a function will throw an error
        return integrals

    def get_coefficients_integration_orthonormal(self) -> np.ndarray:
        """
        Get a matrix representing the integrals of f on the unit square, which is divided into 2ⁿ intervals per dimension.
        :return: A matrix representing the integrals of f on the unit square.
        """
        coefficients = []
        integrals: list[list[float]] = self.get_function_integral_matrix()

        # Assign each subinterval the correct sign / value, by multiplying with the base functions value
        for i in range(2 ** self.n):
            for j in range(2 ** self.n):
                coefficients.append(np.sum(np.array(self.base_functions[i][j].values) * np.array(integrals)))
        return np.array(coefficients)

    def sample_transform(self, coefficients: np.ndarray, samples: int = 256) -> np.ndarray:
        if self.type == TestFunctionType.IMAGE:
            # Sampling the image more often than its resolution will do no good
            resolution: int = self.function.resolution
            x = np.linspace(0, 1, resolution)
            y = np.linspace(0, 1, resolution)
        elif self.type == TestFunctionType.FUNCTION:
            # Grid of equidistant sample points
            x = np.linspace(0, 1, samples)
            y = np.linspace(0, 1, samples)
        # else-case cannot happen, since initializing such a function will throw an error

        approx = np.zeros((len(x), len(y)))
        for i, phi in enumerate(list(np.array(self.base_functions).flatten())):
            approx += coefficients[i] * phi.evaluate_vec(x, y)
        return approx

    def plot_transformation(self, transformation_values: np.ndarray, function_values: np.ndarray = None,
                            subtitle: str = "", original: bool = False,
                            cli: bool = False, fig: Figure = None, index: int = -1, vmin: int = -1,
                            vmax: int = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        subtitle = "\n" + subtitle if subtitle else ""
        title: str = (f"{self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$\n"
                      f"with {2 ** self.n} {self.name} functions per dimension." + subtitle)
        if not cli:
            if self.type == TestFunctionType.IMAGE:
                # Show image as "heatmap"
                plt.imshow(transformation_values, cmap="gray", vmin=0, vmax=255)
                plt.colorbar(label="Value")
                plt.axis('off')
                plt.title(title)
                plt.tight_layout()
                plt.show()

                # True image
                if original:
                    if function_values is None:
                        samples = transformation_values.shape[0]
                        function_values = self.function.sample(samples=samples)
                    plt.imshow(function_values, cmap="gray", vmin=0, vmax=255)
                    plt.colorbar(label="Value")
                    plt.title(f"True image ${self.function.name}$")
                    plt.tight_layout()
                    plt.show()

            elif self.type == TestFunctionType.FUNCTION:
                # Display transformation on an equidistant grid
                x = np.linspace(0, 1, transformation_values.shape[0])
                y = np.linspace(0, 1, transformation_values.shape[1])
                X, Y = np.meshgrid(x, y)

                # Create a 3D plot for transformation
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, transformation_values, cmap='inferno')
                # Rotate origin to face viewer
                ax.view_init(elev=30, azim=-135)
                plt.title(title)
                plt.show()

                if original:
                    if function_values is None:
                        samples = transformation_values.shape[0]
                        function_values = self.function.sample(samples=samples)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(X, Y, function_values, cmap='viridis')
                    ax.view_init(elev=30, azim=-135)
                    plt.title(f"Original function $\\displaystyle {self.function.name}$")
                    plt.show()
        # cli
        else:
            if self.type == TestFunctionType.IMAGE:
                ax = fig.add_subplot(1, 3, index)
                im = ax.imshow(transformation_values, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title)
            elif self.type == TestFunctionType.FUNCTION:
                x = np.linspace(0, 1, transformation_values.shape[0])
                y = np.linspace(0, 1, transformation_values.shape[1])
                X, Y = np.meshgrid(x, y)

                ax = fig.add_subplot(1, 3, index, projection='3d')
                im = ax.plot_surface(X, Y, transformation_values, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.view_init(elev=30, azim=-135)
                ax.set_title(title)
            return im, ax
        return None, None

    def plot_coefficients(self, coefficients: np.ndarray,
                          subtitle: str = "", sorted: bool = False, first_n: int = 0,
                          cli: bool = False, fig: Figure = None, index: int = -1,
                          vmin: int = -1, vmax: int = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        subtitle = "\n" + subtitle if subtitle else ""
        title: str = (f"Coefficients of {self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$\n"
                      f"with {2 ** self.n} {self.name} functions per dimension" + subtitle)
        coefficients_to_plot = coefficients.copy()  # do not modify the original coefficients[]
        if not cli:
            fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=False)

            if sorted:
                coefficients_to_plot[::-1] = np.sort(abs(np.array(coefficients_to_plot)))
                if first_n > 0:
                    coefficients_to_plot = coefficients_to_plot[:first_n]
                    subtitle += f"\nshowing {first_n} coefficients per dimension"
                plt.plot(abs(coefficients_to_plot), color="blue")
                plt.xscale("log")
                plt.yscale("log")
                subtitle += "\nsorted in descending order"
            else:
                coefficients_to_plot = np.reshape(coefficients_to_plot, (2 ** self.n, 2 ** self.n))
                if first_n > 0:
                    coefficients_to_plot = coefficients_to_plot[:first_n, :first_n]
                    subtitle += f"\nshowing {first_n} coefficients per dimension"
                sparse_matrix = u.create_levelsum_matrix(self.n)
                mask = sparse_matrix > 12
                segments = u.boundary_segments_from_mask(mask)
                lc = LineCollection(segments, colors='black', linewidths=1.5, zorder=5)
                ax.add_collection(lc)
                im = ax.imshow(abs(coefficients_to_plot.T), cmap="inferno", norm=LogNorm())
                # This is for transparent discarded sparse coef
                if "sparse" in subtitle:
                    original_coefficients = abs(
                        self.get_coefficients_integration_orthonormal().reshape(2 ** self.n, 2 ** self.n)).T
                    original_coefficients[~mask] = np.nan
                    cmap = plt.get_cmap("inferno").copy()
                    cmap.set_bad(alpha=0.0)
                    ax.imshow(original_coefficients, cmap=cmap, alpha=0.65, norm=LogNorm())
                ax.set_xlabel("$x$")
                ax.set_ylabel("$y$")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="7.5%", pad=0.13)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=12)
                fig.subplots_adjust(right=0.89)

            fig.suptitle(title)
            plt.savefig(f"wav_coef{subtitle}.pdf", dpi=400, pad_inches=0.01)
            plt.show()
        # cli
        else:
            ax = fig.add_subplot(1, 3, index)
            if sorted:
                coefficients_to_plot[::-1] = np.sort(abs(np.array(coefficients_to_plot)))
                if first_n > 0:
                    coefficients_to_plot = coefficients_to_plot[:first_n]
                    subtitle += f"\nshowing {first_n} coefficients per dimension"
                im = ax.plot(abs(coefficients_to_plot), color="blue")
                plt.ylim(vmin, vmax)
                plt.xscale("log")
                plt.yscale("log")
                subtitle += "\nsorted in descending order"
            else:
                coefficients_to_plot = np.reshape(coefficients_to_plot, (2 ** self.n, 2 ** self.n))
                if first_n > 0:
                    coefficients_to_plot = coefficients_to_plot[:first_n, :first_n]
                    subtitle += f"\nshowing {first_n} coefficients per dimension"
                    # Here, .T is needed to show coefficients properly
                im = ax.imshow(abs(coefficients_to_plot.T), cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))

            ax.set_title(title)

            return im, ax
        return None, None

    def plot_error_absolute(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "", cli: bool = False, fig: Figure = None, index: int = -1,
                            vmin: int = 0, vmax: int = -1) -> tuple[AxesImage, Axes] | tuple[None, None]:
        title: str = (f"Absolute error of {self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$\n"
                      f"with {2 ** self.n} {self.name} functions per dimension." + subtitle)
        if not cli:
            if self.type == TestFunctionType.IMAGE:
                # Show image as "heatmap"
                if transform_values.shape != function_values.shape:
                    # Ratio of the shape difference, e.g., ratio = 2 = 32 / 16
                    ratio = int(function_values.shape[0] / transform_values.shape[0])
                    if ratio > 1:
                        transform_values = np.repeat(np.repeat(transform_values, ratio, axis=0), ratio, axis=1)
                        print(f"Upsampling image by factor {ratio}.")
                    else:
                        # Take only every 1 / ratio element, e.g., ratio = 0.5 = 16 / 32 => only every second value
                        transform_values = transform_values[::int(1 / ratio), ::int(1 / ratio)]
                        print(f"Downsampling image by factor {int(1 / ratio)}.")
                plt.imshow(abs(transform_values - function_values), cmap="gray", vmin=0, vmax=255)
                plt.colorbar(label="Value")
                plt.title(title)
                plt.tight_layout()
                plt.show()

            elif self.type == TestFunctionType.FUNCTION:
                x = np.linspace(0, 1, transform_values.shape[0])
                y = np.linspace(0, 1, transform_values.shape[0])
                X, Y = np.meshgrid(x, y)

                Z = abs(transform_values - function_values)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='inferno')
                # Sets the origin to appear at the front
                ax.view_init(elev=30, azim=-135)
                plt.title(title)
                plt.tight_layout()
                plt.show()
        # cli
        else:
            if self.type == TestFunctionType.IMAGE:
                # Show image as "heatmap"
                if transform_values.shape != function_values.shape:
                    # Ratio of the shape difference, e.g., ratio = 2 = 32 / 16
                    ratio = int(function_values.shape[0] / transform_values.shape[0])
                    if ratio > 1:
                        transform_values = np.repeat(np.repeat(transform_values, ratio, axis=0), ratio, axis=1)
                        print(f"Upsampling image by factor {ratio}")
                    else:
                        # Take only every 1 / ratio element, e.g., ratio = 0.5 = 16 / 32 => only every second value
                        transform_values = transform_values[::int(1 / ratio), ::int(1 / ratio)]
                        print(f"Downsampling image by factor {int(1 / ratio)}")
                ax = fig.add_subplot(1, 3, index)
                im = ax.imshow(abs(transform_values - function_values), cmap="gray", vmin=vmin, vmax=255)
                ax.set_title(title)
                return im, ax

            elif self.type == TestFunctionType.FUNCTION:
                x = np.linspace(0, 1, transform_values.shape[0])
                y = np.linspace(0, 1, transform_values.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = abs(transform_values - function_values)

                ax = fig.add_subplot(1, 3, index, projection='3d')
                im = ax.plot_surface(X, Y, Z, cmap='inferno', vmin=0, vmax=vmax)
                # Sets the origin to appear at the front
                ax.view_init(elev=30, azim=-135)
                ax.set_title(title)
                return im, ax
        return None, None

    def evaluate(self, coefficients: np.ndarray, *point: float | tuple[float, float]) -> float | np.ndarray:
        if len(point) != 2:
            raise ValueError("Point must be two-dimensional")
        y_i = 0
        x, y = point
        for coefficient, base_function in zip(coefficients.flatten(), np.array(self.base_functions).flatten()):
            y_i += coefficient * base_function.evaluate(x, y)
        return y_i

    def get_l1_error(self, coefficients: np.ndarray,
                     function_values: np.ndarray = None, transformation_values: np.ndarray = None) -> float:
        if transformation_values is None:
            transformation_values = self.sample_transform(coefficients, samples=2 ** self.boundary_n)
        if function_values is None:
            function_values = np.array(self.function.get_equidistant_data_points(2 ** self.boundary_n))
        if function_values.shape != transformation_values.shape:
            transformation_values = self.sample_transform(coefficients, samples=2 ** self.boundary_n)
            function_values = np.array(self.function.get_equidistant_data_points(2 ** self.boundary_n))

        if self.type == TestFunctionType.FUNCTION:
            total: float = 0

            h: float = 1 / (2 ** self.boundary_n)
            scale: int = u.get_scale(self.boundary_n)  # A larger scale means more accurate approximation
            min_area: float = (h * h) / (2 * scale)
            max_depth: int = int(np.log2(scale) / 2)
            epsilon: float = (np.max(function_values) - np.min(function_values)) * 1e-6

            for x in range(2 ** self.boundary_n):
                x_0 = x * h
                x_1 = (x + 1) * h
                for y in range(2 ** self.boundary_n):
                    y_0 = y * h
                    y_1 = (y + 1) * h
                    c: float = transformation_values[y, x]
                    total += self.adaptive_rect_l1_fast(x_0, x_1, y_0, y_1, c,
                                                        epsilon=epsilon, min_area=min_area, max_depth=max_depth)
        elif self.type == TestFunctionType.IMAGE:
            # For an image, subtract image values from function values and take the absolute value
            total: float = np.sum(np.abs(transformation_values - function_values)) / (2 ** (2 * self.n))
        # else-case cannot happen

        return total

    def adaptive_rect_l1_fast(self, x0: float, x1: float, y0: float, y1: float, c: float,
                              epsilon: float = 1e-7, min_area: float = 1e-7, max_depth: int = 7) -> float:
        """
        Calculates the integral_fine of |f(x,y)-c| over rectangle [x0,x1] × [y0,y1].
        This is done by recursively subdividing a rectangle given by [x0,x1] × [y0,y1],
        while there are sign changes occurring in it.
        :param x0: The x coordinate of the lower left corner of the rectangle.
        :param x1: The x coordinate of the upper right corner of the rectangle.
        :param y0: The y coordinate of the lower left corner of the rectangle.
        :param y1: The y coordinate of the upper right corner of the rectangle.
        :param c: The constant value to subtract from the true function.
        :param epsilon: The maximum absolute error to allow.
        :param min_area: The minimum area of a rectangle's subdivision.
        :param max_depth: The maximum depth of the recursion.
        :return: The approximate integral of the function - c on the specified interval.
        """
        # Two dicts for storing integral- and function-evaluations at corner points
        integral_eval_cache: dict[tuple[float, float], float] = {}
        f_eval_cache: dict[tuple[float, float], float] = {}

        def integral(x, y):
            """
            Evaluate the integral of the used function at the given coordinates.
            The used function is the function for which the transformation was defined.
            :param x: The x coordinate.
            :param y: The y coordinate.
            :return: The value of the integral.
            """
            key = (float(x), float(y))
            if key not in integral_eval_cache:
                integral_eval_cache[key] = self.function.evaluate_integral(x, y)
            return integral_eval_cache[key]

        def function(x, y):
            """
            Evaluate the used function at the given coordinates.
            The used function is the function for which the transformation was defined.
            :param x: The x coordinate.
            :param y: The y coordinate.
            :return: The value of the function.
            """
            key: tuple[float, float] = (float(x), float(y))
            if key not in f_eval_cache:
                f_eval_cache[key] = self.function.evaluate(x, y)
            return f_eval_cache[key]

        total: float = 0
        # initial corner values
        corners: dict[tuple[float, float], float] = {
            (x0, y0): function(x0, y0),
            (x1, y0): function(x1, y0),
            (x0, y1): function(x0, y1),
            (x1, y1): function(x1, y1)
        }

        stack = [(x0, x1, y0, y1, 0, corners)]
        while stack:

            xa, xb, ya, yb, depth, corn = stack.pop()
            area: float = (xb - xa) * (yb - ya)

            # quick area/depth stop
            if area <= min_area or depth >= max_depth:
                integral_f = (integral(xb, yb) - integral(xa, yb) - integral(xb, ya) + integral(xa, ya))
                total += abs(integral_f - c * area)
                continue

            xm: float = 0.5 * (xa + xb)
            ym: float = 0.5 * (ya + yb)
            f_center: float = function(xm, ym) - c

            # The values at the corners - the approximated value c
            f_corner: list = [corn[(xa, ya)] - c, corn[(xb, ya)] - c, corn[(xa, yb)] - c,
                              corn[(xb, yb)] - c, f_center]
            # unanimous sign check (with eps)
            non_zeros = [v for v in f_corner if abs(v) > 1e-12]
            if len(non_zeros) == 0 or all(v > 0 for v in non_zeros) or all(v < 0 for v in non_zeros):
                integral_f = (integral(xb, yb) - integral(xa, yb) - integral(xb, ya) + integral(xa, ya))
                total += abs(integral_f - c * area)
                continue

            rectangle_test_values = f_corner + [f_center]
            variation: float = max(rectangle_test_values) - min(rectangle_test_values)
            local_scale = max(1.0, max(abs(v) for v in rectangle_test_values))
            if variation <= epsilon or variation <= local_scale * epsilon:
                integral_f = (integral(xb, yb) - integral(xa, yb) - integral(xb, ya) + integral(xa, ya))
                total += abs(integral_f - c * area)
                continue

            # compute mid-edge values once and reuse for children
            p_xm_ym = function(xm, ym)  # center (already computed above) but cached P will handle
            p_xm_ya = function(xm, ya)
            p_xb_ym = function(xb, ym)
            p_xa_ym = function(xa, ym)
            p_xm_yb = function(xm, yb)

            # build child corner dicts reusing values
            corn1 = {(xa, ya): corn[(xa, ya)], (xm, ya): p_xm_ya, (xa, ym): p_xa_ym, (xm, ym): p_xm_ym}
            corn2 = {(xm, ya): p_xm_ya, (xb, ya): corn[(xb, ya)], (xm, ym): p_xm_ym, (xb, ym): p_xb_ym}
            corn3 = {(xa, ym): p_xa_ym, (xm, ym): p_xm_ym, (xa, yb): corn[(xa, yb)], (xm, yb): p_xm_yb}
            corn4 = {(xm, ym): p_xm_ym, (xb, ym): p_xb_ym, (xm, yb): p_xm_yb, (xb, yb): corn[(xb, yb)]}

            # push children
            stack.append((xm, xb, ym, yb, depth + 1, corn4))
            stack.append((xa, xm, ym, yb, depth + 1, corn3))
            stack.append((xm, xb, ya, ym, depth + 1, corn2))
            stack.append((xa, xm, ya, ym, depth + 1, corn1))

        return total

    def get_linf_error(self, coefficients: np.ndarray, function_values: np.ndarray = None,
                       transformation_values: np.ndarray = None, samples: int = 256) -> float:
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)
        if samples < 2 ** (self.boundary_n + 2):
            samples = 2 ** (self.boundary_n + 2)
        if isinstance(self.function, TestFunction2D):
            if transformation_values is None:
                transformation_values = self.sample_transform(coefficients, samples=samples)
            if function_values is None:
                function_values = self.function.sample(samples=samples)
            if transformation_values.shape != function_values.shape:
                transformation_values = self.sample_transform(coefficients, samples=samples)
                function_values = self.function.sample(samples=samples)

            difference = abs(transformation_values - function_values)
            return np.max(difference)
        elif isinstance(self.function, Image):
            if transformation_values is None:
                transformation_values = self.sample_transform(coefficients, samples=2 ** self.n)
            if function_values is None:
                function_values = self.function.sample(samples=2 ** self.n)
            if (transformation_values.shape != function_values.shape
                    != (self.function.resolution, self.function.resolution)):
                transformation_values = self.sample_transform(coefficients, samples=samples)
                function_values = self.function.sample(samples=2 ** self.boundary_n)
            difference = abs(transformation_values - function_values)
            return np.max(difference)  # np.max looks at the entire matrix, python max sees only sub-arrays
        else:  # This cannot happen
            return None

    def discard_coefficients_sparse_grid(self, coefficients: np.ndarray, epsilon: int = -1) -> np.ndarray:
        modified_coefficients = coefficients.copy().reshape(2 ** self.n, 2 ** self.n)
        epsilon = self.n if epsilon < 0 else epsilon

        level_sum: np.ndarray = u.create_levelsum_matrix(self.n)

        mask = level_sum > epsilon
        modified_coefficients[mask] = 0

        return modified_coefficients.flatten()
