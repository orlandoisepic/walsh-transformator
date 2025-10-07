import numpy as np
import scipy.special as spc
from scipy.integrate import quad

from utility.templates.test_functions import TestFunction1D

"""
Contains various 1D test functions for the Walsh transform. 
Test functions are chosen to represent multiple different scenarios, 
such as large derivatives, symmetric functions, et cetera. 
"""


class Cosine(TestFunction1D):
    name = "\\cos\\pi x"
    name_cli = "cos(pi x)"

    def evaluate(self, x: float) -> float:
        return np.cos(x * np.pi)

    def evaluate_integral(self, x: float | np.ndarray) -> float | np.ndarray:
        return (1 / np.pi) * np.sin(x * np.pi)

    def evaluate_integral_squared(self, x: float | np.ndarray) -> float | np.ndarray:
        # WolframAlpha
        return 1 / 2 * (x + np.sin(x) * np.cos(x))


class Sine(TestFunction1D):
    name = "\\sin2\\pi x"
    name_cli = "sin(2 pi x)"

    def evaluate(self, x: float) -> float:
        return np.sin(2 * np.pi * x)

    def evaluate_integral(self, x: float) -> float:
        return - np.cos(2 * np.pi * x) / (2 * np.pi)

    def evaluate_integral_squared(self, x: float) -> float:
        return -(np.sin(4 * np.pi * x) - 4 * np.pi * x) / (8 * np.pi)


class CubicPolynomialSymmetric(TestFunction1D):
    name = "6x^3 - 9x^2 + 3x"
    name_cli = name

    def evaluate(self, x: float) -> float:
        return 6 * (x ** 3) - 9 * (x ** 2) + 3 * x

    def evaluate_integral(self, x: float) -> float:
        return (6 / 4) * (x ** 4) - (9 / 3) * (x ** 3) + (3 / 2) * (x ** 2)

    def evaluate_integral_squared(self, x: float) -> float:
        return (36 * (x ** 7)) / 7 - 18 * (x ** 6) + (117 * (x ** 5)) / 5 - (27 * (x ** 4)) / 2 + 3 * (x ** 3)


class CubicPolynomial(TestFunction1D):
    name = "x^3 - x^2"
    name_cli = name

    def evaluate(self, x: float) -> float:
        return (x ** 3) - (x ** 2)

    def evaluate_integral(self, x: float) -> float:
        return (1 / 4) * (x ** 4) - (1 / 3) * (x ** 3)

    def evaluate_integral_squared(self, x: float) -> float:
        return (x ** 7) / 7 - (x ** 6) / 3 + (x ** 5) / 5


class Exponential(TestFunction1D):
    name = "e^x"
    name_cli = "e^x"
    # with (e-1)/2, the first condition ||f^(n)|| ≤ D r^n is NOT fulfilled
    D = (np.e - 1)
    r = 1

    def evaluate(self, x: float) -> float:
        return np.exp(x)

    def evaluate_integral(self, x: float) -> float:
        return np.exp(x)

    def evaluate_integral_squared(self, x: float) -> float:
        return np.exp(2 * x) / 2


class Rational(TestFunction1D):
    name = "\\frac{1}{(x - 0.25)^2 + 0.1}"
    name_cli = "1/((x - 0.25)^2 + 0.1)"

    def evaluate(self, x: float) -> float:
        return 1 / ((x - 0.25) ** 2 + 0.1)

    def evaluate_integral(self, x: float) -> float:
        # WolframAlpha
        return np.sqrt(10) * np.arctan(np.sqrt(10) * (x - 1 / 4))

    def evaluate_integral_squared(self, x: float) -> float:
        # WolframAlpha
        return (50 * (x - 1 / 4)) / (10 * ((x - 1 / 4) ** 2) + 1) + 5 * np.sqrt(10) * np.arctan(
            np.sqrt(10) * (x - 1 / 4))


class SineLog(TestFunction1D):
    name = "\\sin(5\\ln(x + 0.1))"
    name_cli = "sin(5 ln(x + 0.1))"

    def evaluate(self, x: float) -> float:
        return np.sin(5 * np.log(x + 0.1))

    def evaluate_integral(self, x: float) -> float:
        # WolframAlpha
        return 1 / 26 * (x + 1 / 10) * (np.sin(5 * np.log(x + 1 / 10)) - 5 * np.cos(5 * np.log(x + 1 / 10)))

    def evaluate_integral_squared(self, x: float) -> float:
        # WolframAlpha
        return ((10 * x + 1) * (-10 * np.sin(10 * np.log(x + 0.1)) - np.cos(10 * np.log(x + 0.1)) + 101)) / 2020


class Cube(TestFunction1D):
    name = "x^3"
    name_cli = name

    def evaluate(self, x: float) -> float:
        return x ** 3

    def evaluate_integral(self, x: float) -> float:
        return (1 / 4) * (x ** 4)

    def evaluate_integral_squared(self, x: float) -> float:
        return (1 / 7) * (x ** 7)


class Line(TestFunction1D):
    name = "x"
    name_cli = name

    def evaluate(self, x: float) -> float:
        return x

    def evaluate_integral(self, x: float) -> float:
        return (1 / 2) * (x ** 2)

    def evaluate_integral_squared(self, x: float) -> float:
        return (1 / 3) * (x ** 3)


class QuadraticShift(TestFunction1D):
    name = "2(x-1)^2"
    name_cli = name

    def evaluate(self, x: float) -> float:
        return 2 * ((x - 1) ** 2)

    def evaluate_integral(self, x: float) -> float:
        return (2 / 3) * ((x - 1) ** 3)

    def evaluate_integral_squared(self, x: float) -> float:
        return (4 / 5) * ((x - 1) ** 5)


class Quadratic(TestFunction1D):
    name = "x^2"
    name_cli = name

    # Constants for bounding coefficient magnitude
    D = 1 / 2
    r = 2

    def evaluate(self, x: float) -> float:
        # return -2*(x-0.25) ** 2 - x/4
        return x ** 2

    def evaluate_integral(self, x: float) -> float:
        # return -2*(1 / 3) * ((x-0.25) ** 3) - (1/8)*(x**2)
        return (1 / 3) * (x ** 3)

    def evaluate_integral_squared(self, x: float) -> float:
        return (1 / 5) * (x ** 5)


class Constant(TestFunction1D):
    name = "1"
    name_cli = name

    def evaluate(self, x: float | np.ndarray) -> float | list[float]:
        if isinstance(x, (float, int)):
            return 1
        else:
            return [1] * len(x)

    def evaluate_integral(self, x: float) -> float:
        return x

    def evaluate_integral_squared(self, x: float) -> float:
        return x


class ExponentialCosine(TestFunction1D):
    name = "\\cos 4\\pi x \\cdot e^x"
    name_cli = "cos(4 π x) e^(x)"

    def evaluate(self, x: float) -> float:
        return np.exp(x) * np.cos(np.pi * x * 4)

    def evaluate_integral(self, x: float) -> float:
        return np.exp(x) * (np.cos(4 * np.pi * x) + 4 * np.pi * np.sin(4 * np.pi * x)) / (1 + 16 * np.pi ** 2)

    def evaluate_integral_squared(self, x: float) -> float:
        return (1 / (4 + 64 * (np.pi ** 2))) * (
                np.exp(2 * x) * (4 * np.pi * np.sin(8 * np.pi * x) + np.cos(8 * np.pi * x) + (16 * np.pi ** 2 + 1)))
    # (e^(2 x) (4 π sin(8 π x) + cos(8 π x) + 16 π^2 + 1))/(4 + 64 π^2) + constant


class SineExponential(TestFunction1D):
    name = "\\sin ( e^{\\frac{\\pi}2x + 0.75})"
    name_cli = "sin(e^(π/2 x +0.75))"

    def evaluate(self, x: float) -> float:
        return np.sin(np.exp((np.pi / 2) * x + 0.75))

    def evaluate_integral(self, x: float) -> float:
        si, _ = spc.sici(np.exp((np.pi / 2) * x + 0.75))
        return 2 / np.pi * si

    def evaluate_integral_squared(self, x: float) -> float:
        _, ci = spc.sici(np.exp(2 * np.exp((np.pi / 2) * x + 0.75)))
        return x / 2 - (1 / np.pi) * ci


class ExponentialSine(TestFunction1D):
    name = "e^{\\sin(2 \\pi x)}"
    name_cli = "e^(sin(2 π x))"

    def evaluate(self, x: float) -> float:
        return np.exp(np.sin(2 * np.pi * x))

    def evaluate_squared(self, x: float) -> float:
        return np.exp(2 * np.sin(2 * np.pi * x))

    def evaluate_integral(self, x: float) -> float:
        return quad(self.evaluate, 0, x)[0]

    def evaluate_integral_squared(self, x: float) -> float:
        return quad(self.evaluate_squared, 0, x)[0]


class HighPolynomial(TestFunction1D):
    name = "x^{11}"
    name_cli = "x^11"

    def evaluate(self, x: float) -> float:
        return x ** 11

    def evaluate_integral(self, x: float) -> float:
        return (x ** 12) / 12

    def evaluate_integral_squared(self, x: float) -> float:
        return x ** 23 / 23


class TaylorCosine(TestFunction1D):
    name = ""
    name_cli = ""

    def evaluate(self, x: float) -> float:
        return 1 - (np.pi ** 2 * x ** 2) / 2 + (np.pi ** 4 * x ** 4) / 24 - (np.pi ** 6 * x ** 6) / 720 + (
                np.pi ** 8 * x ** 8) / 40320

    def evaluate_integral(self, x: float) -> float:
        return (np.pi ** 8 * x ** 9) / 362880 - (np.pi ** 6 * x ** 7) / 5040 + (np.pi ** 4 * x ** 5) / 120 - (
                np.pi ** 2 * x ** 3) / 6 + x

    def evaluate_integral_squared(self, x: float) -> float:
        pass


class InverseSine(TestFunction1D):
    name = ""
    name_cli = ""

    def evaluate(self, x: float) -> float:
        return 1 / (np.sin(2 * x) + 1.1)

    def evaluate_integral(self, x: float) -> float:
        k = 2
        return (20.0 / (k * np.sqrt(21.0))) * np.arctan((11.0 * np.tan((k * x) / 2.0) + 10.0) / np.sqrt(21.0))

    def evaluate_integral_squared(self, x: float) -> float:
        SQ21 = np.sqrt(21)
        x = np.asarray(x)
        t = np.tan(x)
        R = 11.0 * t**2 + 20.0 * t + 11.0
        rational = (10000.0 * t + 11000.0) / (231.0 * R)
        arctan_term = (1100.0 / (21.0 * SQ21)) * np.arctan((11.0 * t + 10.0) / SQ21)
        return rational + arctan_term  # + C
