import numpy as np
import matplotlib.pyplot as plt

import utility.utils as u
from transformations.walsh_transformation_1d.experimental_walsh_function import ExperimentalWalshFunction
from utility import test_functions_1d
from utility.templates.base_transformations import Transformation
from walsh_function_1d import WalshFunction
from utility.templates.test_functions import TestFunction
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation

u.latex_font()


class ExperimentalWalshTransformation:
    """
    This class implements a Walsh transformation for a given function.
    """

    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        """
        Initializes a Walsh transformation by initializing 2^n Walsh functions.
        :param n: 2^n is the number of Walsh functions.
        """
        self.n = n
        self.function = function
        if boundary_n != -1 and boundary_n > n:
            self.boundary_n = boundary_n
        else:
            self.boundary_n = n

        # calculate more boundaries, use only the first 2^n
        self.exp_boundaries = u.calculate_boundaries_exp(self.boundary_n)
        self.boundary_order_exp: np.ndarray = u.get_order_from_bounds(self.exp_boundaries)
        # print(len(self.boundary_order_exp))
        # print(u.is_sorted_descending(self.exp_boundaries))

        self.cos_boundaries = u.calculate_boundaries_cos(n)
        self.boundary_order_cos: np.ndarray = u.get_order_from_bounds(self.cos_boundaries)

        self.cos2pi_boundaries = u.calculate_boundaries_cos2pi(n)
        self.boundary_order_cos2pi: np.ndarray = u.get_order_from_bounds(self.cos2pi_boundaries)

        self.dyadic_order: list[int] = u.sequency_to_dyadic(n)

        self.function_values = function.get_equidistant_data_points(2 ** self.boundary_n)

        intervals, intervals_b = u.walsh_function_setup_helper(n)

        walsh_functions: list[ExperimentalWalshFunction] = []
        # Add Walsh functions to matrix => Create Walsh-Matrix
        # Doing the trick
        if self.boundary_n > self.n:
            for i in range(2 ** n):
                walsh_functions.append(WalshFunction(self.boundary_order_exp[i], self.boundary_n))
        else:
            for i in range(2 ** n):
                # walsh_functions.append(WalshFunction(self.boundary_order_exp[i], n))
                walsh_functions.append(WalshFunction(self.dyadic_order[i], n))
        self.walsh_functions = walsh_functions
        # A Matrix of Walsh functions is a Hadamard matrix
        self.walsh_values = [f.values for f in self.walsh_functions]

    def get_coefficients_interpolation(self) -> np.ndarray:
        """
        Calculate the coefficients of the Walsh transformation by solving a linear system of equations
        of function evaluations.
        :return: The coefficients of the Walsh functions.
        """
        # Convert to numpy-array
        function = np.array(self.function_values)
        # Use only f(x) values
        vector = function[:, 1]

        matrix = np.array(self.walsh_values)
        # Since 1/2^n * Walsh-Matrix is an orthonormal matrix, its inverse is equal to its transpose.
        # It is coefficients = 1/2^n * Walsh-Matrix * function-values
        return 1 / (2 ** self.n) * matrix @ vector

    def get_coefficients_integration(self) -> np.ndarray:
        """
        Calculate the coefficients of the Walsh transformation by integrating the Walsh functions and the test function.
        :return: The coefficients of the Walsh functions.
        """
        # It is αᵢ = ∫ gfᵢ / ∫ f²ᵢ
        # However, for a Walsh function fᵢ, its square is equal to one, as it only takes values -1 and 1.
        # This means that the coefficient is equal to ∫ gfᵢ
        coef: list[float] = []
        integral_s: list[float] = []
        for i in range(2 ** self.boundary_n):
            integral_i = (self.function.evaluate_integral((i + 1) / (2 ** self.boundary_n))
                          - self.function.evaluate_integral(i / (2 ** self.boundary_n)))
            integral_s.append(integral_i)

        # Select Walsh function
        for i in range(2 ** self.n):
            # integral = 0
            # Select interval to calculate integral for
            # for j in range(2 ** self.n):
            #     # The Walsh function is only a factor in each interval (doesn't contribute to integral itself)
            #     factor = self.walsh_functions[i].values[j]
            #     # Exact integral
            #     integral += factor * (self.function.evaluate_integral((j + 1) / (2 ** self.n))
            #                           - self.function.evaluate_integral(j / (2 ** self.n)))
            walsh_integral = np.sum(np.array(self.walsh_functions[i].values) * np.array(integral_s))
            coef.append(walsh_integral)
        for i, c in enumerate(coef):
            # print(i, abs(c), self.exp_boundaries[self.boundary_order_exp[i]])
            pass
        # print("Using", len(walcoef), "coefficients.")
        return np.array(coef)

    def plot_walsh_matrix(self) -> None:
        """
        Plots the Walsh-Matrix, i.e., the values of every Walsh function in the interval [0,1].
        The entries in line i represent one Walsh function of order i.
        This opens a window with the plot.
        :return: None
        """
        y = self.walsh_values

        plt.figure(figsize=(6, 6))
        plt.imshow(y, cmap="grey", aspect="auto")
        # plt.colorbar(label="Value")
        # plt.title(f"Walsh-Functions for $n={self.n}$")
        plt.show()

    def plot_transformation(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "") -> None:
        # Only append a new line if the subtitle is not empty.
        subtitle = "\n" + subtitle if subtitle else ""
        plt.figure(figsize=(4.5, 4.5), constrained_layout=True)
        x = np.linspace(0, 1, transform_values.size)
        if function_values.shape != transform_values.shape:
            function_values = self.function.sample(transform_values.size)
        plt.plot(x, transform_values, color="blue")
        plt.plot(x, function_values, color="orange")
        # plt.title(f"Walsh transformation of $\\displaystyle {self.function.name}$" + subtitle)
        plt.savefig("exptrick.pdf", dpi=400, pad_inches=0.01)
        plt.show()

    def sample_transform(self, coefficients: np.ndarray, samples: int = 1024) -> np.ndarray:
        # If samples is not a power of two, then eval_vec will not work correctly
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)
        # If samples is less than 2ⁿ, then eval_vec will throw an error
        # Increase it to 2ⁿ to avoid this
        if samples < 2 ** self.boundary_n:
            samples = 2 ** self.boundary_n
        x = np.linspace(0, 1, samples)
        y = 0
        for i, phi in enumerate(self.walsh_functions):
            y += coefficients[i] * phi.evaluate_vec(x)

        # Multiplying the Walsh matrix with the coefficients only yields the values of the transformation
        # at the data points and is thus not helpful when determining the error.
        return np.array(y)

    def plot_transformation_steps_rad(self, coefficients: np.ndarray, function_values: np.ndarray,
                                      samples: int = 1024) -> None:
        fmax = function_values.max() + 0.25
        fmin = function_values.min() - 0.25
        start = 0
        stop = start + 0.4
        yoffset = function_values.max() - function_values.max() / 10
        yrange = (function_values.max() - function_values.min()) / 10

        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)
        # If samples is less than 2ⁿ, then eval_vec will throw an error
        # Increase it to 2ⁿ to avoid this
        if samples < 2 ** self.boundary_n:
            samples = 2 ** self.boundary_n

        coef_to_plot = coefficients.copy()

        x = np.linspace(0, 1, samples)
        y = 0
        i = 0
        odd_weight: int = 0
        abs_odd_weight: int = 0
        even_weight: int = 0
        abs_even_weight: int = 0
        rad_weight: int = 0
        abs_rad_weight: int = 0
        for c, walsh in zip(coef_to_plot, self.walsh_functions):
            if u.is_power_of_two(i + 1) or i == 0:
                # print(">>>", i)
                # print("   ", c)
                # print("   ", odd_weight)
                # print("   ", abs_odd_weight)
                plt.figure(figsize=(6, 6))
                y += c * walsh.evaluate_vec(x)
                k = np.sign(c) * yrange if np.sign(c) != 0 else yrange
                walsh_vals = k * np.array(walsh.values) + yoffset
                plt.step(np.linspace(start, stop, len(walsh_vals)), walsh_vals, color="gray", where="mid")
                plt.plot(x, y, color="blue")
                plt.plot(x, function_values, color="orange")
                plt.ylim(fmin, fmax)
                plt.title(f"fkt, nr: {i}")
                plt.tight_layout()
                plt.show()
                if i != 0:
                    rad_weight += c
                    abs_rad_weight += abs(c)
                    odd_weight += c
                    abs_odd_weight += abs(c)
                coef_to_plot[i] = 0
            i += 1
        # print("Approx: Abstand min und max:", y.max() - y.min())
        # print("Exakt: Abstand min und max:", function_values.max() - function_values.min())
        print("Approx: Abstand 0,1:", abs(y[0] - y[-1]))
        print("Exakt: Abstand 0,1:", abs(function_values[0] - function_values[-1]))
        print("========== rad fertig ==========")
        i = 0
        unevens = np.zeros_like(self.walsh_functions[0].values)
        for c, walsh in zip(coef_to_plot, self.walsh_functions):
            if u.is_power_of_two(i + 1) or i == 0:
                i += 1
                continue
            # print(">>>", i)
            if walsh.values[0] != walsh.values[-1]:
                # print("   ", c)
                # print("   ", odd_weight)
                # print("   ", abs_odd_weight)
                # print(np.isclose(c, 0))

                unevens += np.array(walsh.values)
                odd_weight += c
                abs_odd_weight += abs(c)
                pass
            else:
                even_weight += c
                abs_even_weight += abs(c)
                # print(np.isclose(c, 0))
                pass
            plt.figure(figsize=(6, 6))
            y += c * walsh.evaluate_vec(x)
            k = np.sign(c) * yrange if np.sign(c) != 0 else yrange
            walsh_vals = k * np.array(walsh.values) + yoffset
            plt.step(np.linspace(start, stop, len(walsh_vals)), walsh_vals, color="gray", where="mid")
            plt.plot(x, y, color="blue")
            plt.plot(x, function_values, color="orange")
            plt.ylim(fmin, fmax)
            i += 1
            plt.title(f"fkt, nr: {walsh.order}")
            plt.tight_layout()
            plt.show()
        plt.figure(figsize=(6, 6))

        # plt.step(np.linspace(0, 1, len(unevens)), unevens, color="blue")
        # plt.title("summe ungerade nicht-rademacher")
        # plt.tight_layout()
        # plt.show()
        # print("Approx: Abstand min und max:", y.max() - y.min())
        # print("Exakt: Abstand min und max:", function_values.max() - function_values.min())
        print("Approx: Abstand 0,1:", abs(y[0] - y[-1]))
        print("Exakt: Abstand 0,1:", abs(function_values[0] - function_values[-1]))
        print("Gewicht der ungeraden:", odd_weight)
        print("Absolutes Gewicht der ungeraden:", abs_odd_weight)
        print("Gewicht der Rademacher:", rad_weight)
        print("Absolutes Gewicht der Rademacher + skala:", abs_rad_weight)
        print("Absolutes Gewicht der ungeraden ohne Rademacher und skala:", abs_odd_weight - abs_rad_weight)
        print("Gewicht der geraden:", even_weight)
        print("Absolutes Gewicht der geraden:", abs_even_weight)
        print("Summe aller absoluten Koeffizienten:", np.abs(coefficients).sum())
        print("Summe aller absoluten Koeffizienten - skala:", np.abs(coefficients).sum() - coefficients[0])

    def plot_transformation_steps_abs(self, coefficients: np.ndarray, function_values: np.ndarray,
                                      samples: int = 1024) -> None:
        fmax = function_values.max() + 0.25
        fmin = function_values.min() - 0.25
        start = 0
        stop = start + 0.4
        if not u.is_power_of_two(samples):
            samples = u.increase_to_next_power_of_two(samples)
        # If samples is less than 2ⁿ, then eval_vec will throw an error
        # Increase it to 2ⁿ to avoid this
        if samples < 2 ** self.boundary_n:
            samples = 2 ** self.boundary_n

        coef_to_plot = coefficients.copy()

        sorted_indices = np.argsort(abs(coefficients))[::-1]
        x = np.linspace(0, 1, samples)
        y = 0
        i = 0

        sorted_coefficients = coef_to_plot[sorted_indices]
        sorted_walshs = np.array(self.walsh_functions)[sorted_indices]
        for c, walsh in zip(sorted_coefficients, sorted_walshs):
            plt.figure(figsize=(6, 6))
            y += c * walsh.evaluate_vec(x)
            c = np.sign(c) * 0.1
            if i == 0 or i == 1:
                walsh_vals = c * np.array(walsh.values) + 1
            else:
                walsh_vals = c * np.array(walsh.values) + 1
            plt.plot(x, y, color="blue")
            plt.plot(x, function_values, color="orange")
            plt.ylim(fmin, fmax)
            plt.step(np.linspace(start, stop, len(walsh_vals)), walsh_vals, color="gray", where="mid")
            plt.title(f"{i + 1}/{2 ** self.n} fkt, nr: {sorted_indices[i]}")
            i += 1
            plt.tight_layout()
            plt.show()

    def plot_error_absolute(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "") -> None:
        # Only append a new line if the subtitle is not empty.
        subtitle = "\n" + subtitle if subtitle else ""
        plt.figure(figsize=(6, 6))
        x = np.linspace(0, 1, transform_values.size)
        y = abs(transform_values - function_values)
        plt.plot(x, y, color="blue")
        plt.title(
            f"Absolute error of Walsh transformation of $\\displaystyle {self.function.name}$." + subtitle)
        plt.xlim(0, 1)
        plt.show()

    def plot_error_relative(self, transform_values: np.ndarray, function_values: np.ndarray,
                            subtitle: str = "") -> None:
        # Only append a new line if the subtitle is not empty.
        subtitle = "\n" + subtitle if subtitle else ""
        plt.figure(figsize=(6, 6))
        x = np.linspace(0, 1, transform_values.size)
        y = abs((transform_values - function_values) + 1e-16) / abs(function_values + 1e-16)
        plt.plot(x, y)
        plt.yscale("log")
        plt.title(f"Relative error Walsh transformation of $\\displaystyle {self.function.name}$." + subtitle)
        plt.show()

    def plot_coefficients(self, coefficients: np.ndarray, subtitle: str = "", sorted: bool = False,
                          first_n: int = 0) -> None:
        # Do not update coefficients[] object
        coefficients_to_plot = coefficients.copy()
        subtitle = "\n" + subtitle if subtitle else ""

        # Plot only the first n coefficients
        if first_n > 0:
            new_coefficients = []
            for i in range(len(coefficients_to_plot)):
                new_coefficients.append(coefficients_to_plot[i])
                if i > first_n:
                    break
            coefficients_to_plot = new_coefficients
            subtitle += f"\nshowing {first_n} coefficients"
        if sorted:
            # Sort coefficients in descending order by absolute value
            coefficients_to_plot[::-1] = np.sort(abs(np.array(coefficients_to_plot)))
            subtitle += "\nsorted in descending order"
        plt.figure(figsize=(4.5, 4.5), constrained_layout=True)

        x = np.linspace(1, coefficients_to_plot.shape[0] + 1, coefficients_to_plot.shape[0], endpoint=False)
        y = (np.e -1) / x
        plt.plot(x, y, color="gray")

        x = np.linspace(1, coefficients_to_plot.shape[0] + 1, coefficients_to_plot.shape[0], endpoint=False)
        y = (np.e -1) / (x ** 2)
        plt.plot(x, y, color="gray")

        x = np.linspace(1, coefficients_to_plot.shape[0] + 1, coefficients_to_plot.shape[0], endpoint=False)
        y = (np.e -1) / (x ** 3)
        plt.plot(x, y, color="gray")

        # cos2pi_ordered: list[float] = [float(coefficients[self.boundary_order_cos2pi[i]]) for i in
        #                                range(len(coefficients))]
        # cos_ordered: list[float] = [float(coefficients[self.boundary_order_cos[i]]) for i in range(len(coefficients))]
        # exp_ordered: list[float] = [float(coefficients[self.boundary_order_exp[i]]) for i in range(len(coefficients))]

        # cofficients ordered according to exponential order
        # exp_ordered: list[float] = [float(coefficients[self.boundary_order_exp[i]]) for i in range(len(coefficients))]
        # plt.plot(range(1, len(exp_ordered) + 1), abs(np.array(exp_ordered)), color="green", label="Coef in exp order")

        # exponential boundaries ordered according to exponential order
        # exp_boundaries_exp_ordered = [self.exp_boundaries[self.boundary_order_exp[i]] for i in
        #                               range(2 ** self.boundary_n)]
        # exp_boundaries_dyad_ordered = [self.exp_boundaries[self.dyadic_order[i]] for i in
        #                                range(2 ** self.boundary_n)]
        # plt.plot(range(1, len(exp_boundaries_exp_ordered) + 1), exp_boundaries_exp_ordered, color="lightgreen",
        #          label="Exp boundaries in exp order")

        # dyad_ordered: list[float] = [float(coefficients[self.dyadic_order[i]]) for i in range(len(coefficients))]
        # actual_order: list[float] = [float(coefficients[i]) for i in range(len(coefficients))]
        # print("actual ordered", actual_order)
        # print("cos order", self.boundary_order_cos)
        # print("cos ordered", list(cos_ordered))
        # print("dyadic order", self.dyadic_order)
        # print("dyadic ordered", list(dyad_ordered))
        # plt.plot(range(1, coefficients_to_plot.shape[0] + 1), abs(np.array(cos_ordered)), color="red", label="Cosine order")
        # plt.plot(range(1, coefficients_to_plot.shape[0] + 1), abs(np.array(cos2pi_ordered)), color="orange",
        #          label="Cosine 2pi order")
        # plt.plot(np.sort(abs(np.array(coefficients_to_plot)))[::-1], label="Walsh coefficients sorted", color="blue")
        # plt.plot(np.sort(self.exp_boundaries)[::-1], label="Boundaries of $e^x$", color="green")
        # plt.plot(np.sort(self.cos_boundaries)[::-1],label="Boundaries of $\\cos x$", color="red")

        # plt.plot(range(1, coefficients_to_plot.shape[0] + 1), exp_boundaries_exp_ordered, color="green",
        #          label="Schranke")

        plt.plot(range(1, coefficients_to_plot.shape[0] + 1), abs(coefficients_to_plot), "--", color="blue",
                 marker="o", label="Koeffizienten")

        # plt.plot(self.exp_boundaries, label="Boundaries of $e^x$", color="green")
        # plt.plot(self.cos_boundaries, label="Boundaries of $\\cos x$", color="red")
        plt.yscale("log")
        plt.xscale("log")
        # plt.legend(loc="lower left")
        # plt.xlim(8.5e-1, 36)
        plt.ylim(9.14823772291129e-14 - 6e-14, 1.718281828459045 + 3)
        # plt.title(f"Coefficients of Walsh transformation of $\\displaystyle {self.function.name}$ "
        #          f"with {2 ** self.n} Walsh functions" + subtitle)

        plt.savefig(f"expcoeftrick.pdf", dpi=400, pad_inches=0.01)
        plt.show()

    def evaluate(self, coefficients: np.ndarray, *point: float) -> float | np.ndarray:
        if len(point) != 1:
            raise ValueError("Point must be one-dimensional")
        y_i = 0
        x = point[0]
        for i, f in enumerate(self.walsh_functions):
            y_i += coefficients[i] * f.evaluate(x)
        return y_i

    def test_coefficient_boundaries(self, coefficients: np.ndarray) -> None:
        """
        Test the boundaries of coefficients given by
        'Formulas for the Walsh coefficients of smooth functions and their application to bounds on the Walsh coefficients'
        :param coefficients:
        :param subtitle:
        :param first_n:
        :return:
        """
        m_2 = 2
        D = self.function.D
        r = self.function.r
        if D == -1:
            print("Operation not yet supported.")
            return
        b = 2  # Base two
        C_2 = 2
        bounds: np.ndarray = np.zeros_like(coefficients)
        for i in range(len(coefficients)):
            coef = coefficients[i]
            walter = self.walsh_functions[i]
            mu = walter.mu
            nu = walter.nu
            bounds[i] = D * (b ** -mu) * ((r / m_2) ** nu) * (C_2 ** min(1, nu))
            print(f"sequency={i},dyadic={walter.dyadic_order} D={D}, mu={mu}, nu={nu}")
            print(
                f"Coefficient {str(i).rjust(3)}: {str(abs(coef).__round__(6)).rjust(6)},  bound: {str(bounds[i].__round__(6)).ljust(6)}, <= ? {abs(coef) <= bounds[i]}\n"
                f">>> Their difference is: {str(abs(abs(coef) - bounds[i]).__round__(6)).ljust(10)} <<<>>> their ratio is: {str((bounds[i] / abs(coef + 1e-16)).__round__(6)).ljust(10)} <<<"
            )
            if bounds[i] / abs(coef + 1e-16) < 1.5:
                print(
                    "")
            if bounds[i] < abs(coef):
                print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                print(f"    Thats not good. The difference is: {abs(abs(coef) - bounds[i])} <<<")
                print(f"    {coef} <== actual value ")
                print(
                    "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        bounds_s: np.ndarray = np.sort(bounds)[::-1]
        index_s: np.ndarray = np.argsort(bounds)[::-1]
        print("===========================================================================")
        for index, bound in zip(index_s, bounds_s):
            print(f"index={index}, bound={bound}")
        x = np.linspace(0, len(bounds), len(bounds))
        plt.plot(x, bounds_s)
        plt.yscale("log")
        plt.xscale("log")
        plt.show()


f = test_functions_1d.Exponential()
# f = test_functions_1d.TaylorCosine()
# f= test_functions_1d.Cosine()
# f = test_functions_1d.Cosine()
# f = test_functions_1d.Quadratic()
# f = test_functions_1d.Cube()
# f= test_functions_1d.Line()
# f = test_functions_1d.HighPolynomial()
# f = test_functions_1d.ExponentialSine()

# n = 8
# b_n = n + 8

## =================== trick mit boundary_n ===================
# walt = ExperimentalWalshTransformation(n, f, boundary_n=b_n)
# # walt.plot_walsh_matrix()
# walcoef = walt.get_coefficients_integration()
# tvals = walt.sample_transform(walcoef, samples=2 ** b_n)
# fvals = f.sample(samples=2 ** b_n)
# walt.plot_coefficients(walcoef)
# walt.plot_transformation(tvals, fvals, subtitle="$n=8$, exp schranke bis $n=16$")
# walt.plot_error_absolute(tvals, fvals, subtitle="$n=5$,exp schranke bis $n=10$")
# print(np.abs(walcoef).min(), np.abs(walcoef).max())


# violet = ExperimentalWalshTransformation(n, f)
# walcoef = violet.get_coefficients_integration()
# fvals = f.sample(samples=2 ** n)
# tvals = violet.sample_transform(walcoef, samples=2 ** n)
# violet.plot_coefficients(walcoef)
#violet.plot_transformation(tvals, fvals, subtitle="paley $n=5$")
#violet.plot_error_absolute(tvals, fvals, subtitle="paley $n=5$")
#print(np.abs(walcoef).min(), np.abs(walcoef).max())
