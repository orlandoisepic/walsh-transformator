import numpy as np
import scipy.special as spc
from scipy.integrate import dblquad

from utility.templates.test_functions import TestFunction2D


class TensorCosine(TestFunction2D):
    name = "\\cos \\pi x \\cdot \\cos \\pi y"
    name_cli = "cos(pi x) * cos(pi y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.cos(x * np.pi) * np.cos(y * np.pi)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (1 / (np.pi ** 2)) * np.sin(x * np.pi) * np.sin(y * np.pi)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return ((x * (2 * np.pi * y + np.sin(2 * np.pi * y))) / (8 * np.pi) +
                (np.sin(2 * np.pi * x) * (2 * np.pi * y + np.sin(2 * np.pi * y))) / (16 * (np.pi ** 2)))


class XCosine(TestFunction2D):
    name = "x \\cdot \\cos(x+y) + y"
    name_cli = "x * cos(x + y) + y"

    def evaluate(self, x: float, y: float) -> float:
        return x * np.cos(x + y) + y

    def evaluate_integral(self, x: float, y: float) -> float:
        return (x * y ** 2) / 2 + np.sin(x + y) - x * np.cos(x + y)

    def evaluate_integral_squared(self, x: float, y: float) -> float:
        return (x ** 3 * y) / 6 + 1 / 48 * (3 - 6 * x ** 2) * np.cos(2 * (x + y)) + (x * y ** 3) / 3 + 2 * x * np.sin(
            x + y) + 1 / 8 * x * np.sin(2 * (x + y)) + 2 * y * np.sin(x + y) + (2 - 2 * x * y) * np.cos(x + y)


class Cosine(TestFunction2D):
    name = "\\cos(2x + y)"
    name_cli = "cos(2x + y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.cos(2 * x + y)

    def evaluate_integral(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        return - 1 / 2 * np.cos(2 * x + y)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        return (x * y) / 2 - 1 / 16 * np.cos(2 * (2 * x + y)) + (y ** 2) / 8


class CosineXSquare(TestFunction2D):
    name = "\\cos(x^2 + y)"
    name_cli = "cos(x^2 + y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.cos(x ** 2 + y)

    def evaluate_integral(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        z = np.sqrt(2 / np.pi) * x
        S, C = spc.fresnel(z)

        value = np.sqrt(np.pi / 2) * (C * np.sin(y) + S * np.cos(y))
        return value

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float) -> float | np.ndarray:
        z = (2 * x) / np.sqrt(np.pi)
        S, C = spc.fresnel(z)

        fresnel_term = (1 / 8) * np.sqrt(np.pi) * (C * np.sin(2 * y) + S * np.cos(2 * y))
        polynomial_term = (x ** 3) / 6 + (x * y) / 2

        return fresnel_term + polynomial_term


class QuadraticAdd(TestFunction2D):
    name = "xy + x"
    name_cli = name

    def evaluate(self, x: float, y: float) -> float:
        return x * y + x

    def evaluate_integral(self, x: float, y: float) -> float:
        return (1 / 4) * (x ** 2) * (y ** 2) + (1 / 2) * (x ** 2) * y

    def evaluate_integral_squared(self, x: float, y: float) -> float:
        return (1 / 9) * (x ** 3) * (y + 1) ** 3


class QuadraticMax(TestFunction2D):
    name = "x^2 + y^2 + xy + x + y + 1"
    name_cli = name

    def evaluate(self, x: float, y: float) -> float:
        return x ** 2 + y ** 2 + x * y + x + y + 1

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return ((1 / 3) * (x ** 3) * y + (1 / 3) * (y ** 3) * x +
                (1 / 4) * (x ** 2) * (y ** 2) + (1 / 2) * (x ** 2) * y + (1 / 2) * (y ** 2) * x + x * y)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (((x ** 5) * y) / 5 + ((x ** 4) * (y ** 2)) / 4 + ((x ** 4) * y) / 2 + ((x ** 3) * (y ** 3)) / 3
                + (2 * (x ** 3) * (y ** 2)) / 3 + ((x ** 3) * y) + ((x ** 2) * (y ** 4)) / 4
                + (2 * (x ** 2) * (y ** 3)) / 3 + ((x ** 2) * (y ** 2)) + ((x ** 2) * y) + (x * (y ** 5)) / 5
                + (x * (y ** 4)) / 2 + (x * (y ** 3)) + (x * (y ** 2)) + (x * y))


class QuadraticMid(TestFunction2D):
    name = "xy + x + y + 1"
    name_cli = name

    def evaluate(self, x: float, y: float) -> float:
        return x * y + x + y + 1

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (1 / 4) * (x ** 2) * (y ** 2) + (1 / 2) * (x ** 2) * y + (1 / 2) * (y ** 2) * x + x * y

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return 1 / 9 * (x + 1) ** 3 * (y + 1) ** 3


class QuadraticMin(TestFunction2D):
    name = "xy"
    name_cli = name

    def evaluate(self, x: float, y: float) -> float:
        return x * y

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (1 / 4) * (x ** 2) * (y ** 2)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (1 / 9) * (x ** 3) * (y ** 3)


class CubicPolynomial(TestFunction2D):
    name = "-x^3 - y^3 + x^2 + y^2 + xy"
    name_cli = name

    def evaluate(self, x: float, y: float) -> float:
        return -(x ** 3) - (y ** 3) + (x ** 2) + (y ** 2) + (x * y)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (-(1 / 4) * (x ** 4) * y - (1 / 4) * (y ** 4) * x + (1 / 3) * (x ** 3) * y + (1 / 3) * (y ** 3) * x
                + (1 / 4) * (x * y) ** 2)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return ((x ** 7) * y) / 7 - ((x ** 6) * y) / 3 - ((x ** 5) * (y ** 2)) / 5 + ((x ** 5) * y) / 5 + (
                (x ** 4) * (y ** 4)) / 8 - ((x ** 4) * (y ** 3)) / 6 + ((x ** 4) ** (y ** 2)) / 4 - (
                (x ** 3) * (y ** 4)) / 6 + ((x ** 3) * (y ** 3)) / 3 - ((x ** 2) * (y ** 5)) / 5 + (
                (x ** 2) * (y ** 4)) / 4 + (x * (y ** 7)) / 7 - (x * (y ** 6)) / 3 + (x * (y ** 5)) / 5


class ExponentialAdd(TestFunction2D):
    name = "e^{x+y}"
    name_cli = "e^(x + y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.exp(x + y)

    def evaluate_integral(self, x: float, y: float) -> float:
        return np.exp(x + y)

    def evaluate_integral_squared(self, x: float, y: float) -> float:
        return (1 / 4) * np.exp(2 * (x + y))


class ExponentialMult(TestFunction2D):
    name = "e^{xy}"
    name_cli = "e^(xy)"

    def evaluate(self, x: float, y: float) -> float:
        return np.exp(x * y)

    def evaluate_integral(self, x: float, y: float) -> float:
        # No closed form exists; ChatGPT suggestion
        return dblquad(lambda v, u: np.exp(u * v), 0, x, lambda u: 0, lambda u: y)[0]

    def evaluate_integral_squared(self, x: float, y: float) -> float:
        return dblquad(lambda v, u: np.exp(2 * u * v), 0, x, lambda u: 0, lambda u: y)[0]


class Rational(TestFunction2D):
    name = "\\frac{2x-2y}{x+6y+1}"
    name_cli = "(2x - 2y)/(x + 6y + 1)"

    def evaluate(self, x: float, y: float) -> float:
        return (2 * x - 2 * y) / (x + 6 * y + 1)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        # WolframAlpha
        return 1 / 72 * (x * (- 7 * x + 60 * y + 10) + 2 * (7 * (x ** 2) + 2 * x - 252 * (y ** 2) - 72 * y - 5)
                         * np.log(x + 6 * y + 1))

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        # WolframAlpha
        return (1 / 54) * (6 * x * ((43 * y) + 5) - (7 * (x ** 2) + (2 * x) + 1512 * (y ** 2) + (432 * y) + 31)
                           * np.log(x + (6 * y) + 1) - 21 * (x ** 2) + 2304 * (y ** 2) + (690 * y) + 51)


class Sine(TestFunction2D):
    name = "\\sin(2x + y) + \\cos(0.5x+9y)"
    name_cli = "sin(2x + y) + cos(0.5x+9y)"

    def evaluate(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return np.sin(2 * x + y) + np.cos(x / 2 + 9 * y) / 2

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return -np.sin(2 * x + y) / 2 - np.cos(x / 2 + 9 * y) / 9

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return 5 * x * y / 8 + np.cos(4 * x + 2 * y) / 16 - np.cos(x + 18 * y) / 144 - np.sin(
            5 * x / 2 + 10 * y) / 50 + np.sin(3 * x / 2 - 8 * y) / 24


class TensorSine(TestFunction2D):
    name = "\\sin(\\pi x)\\sin(\\pi y)"
    name_cli = "sin(pi x) sin(pi y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (np.cos(np.pi * x) * np.cos(np.pi * y)) / (np.pi ** 2)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return ((x * y) / 4 - (x * np.sin(2 * np.pi * y)) / (8 * np.pi) +
                (np.sin(2 * np.pi * x) * (np.sin(2 * np.pi * y) - 2 * np.pi * y)) / (16 * (np.pi ** 2)))


class Constant(TestFunction2D):
    name = "1"
    name_cli = "1"

    def evaluate(self, x: float, y: float) -> float:
        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            return 1
        else:
            return [1 * len(x)] * len(y)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return x * y

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return x * y


class ExponentialCosine(TestFunction2D):
    name = "e^x \\cdot \\cos 2 \\pi y"
    name_cli = "e^x * cos(2 pi y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.exp(x) * np.cos(2 * np.pi * y)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (np.exp(x) * np.sin(2 * np.pi * y)) / (2 * np.pi)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (np.exp(2 * x) * (4 * np.pi * y + np.sin(4 * np.pi * y))) / (16 * np.pi)


class SineExponential(TestFunction2D):
    name = "e^{x+y} \\cdot \\sin(\\pi x + y)"
    name_cli = "e^(x+y) sin(pi x + y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.exp(x + y) * np.sin(np.pi * x + y)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return np.exp(x + y) / (2 * (1 + np.pi ** 2)) * (
                (1 - np.pi) * np.sin(np.pi * x + y) - (1 + np.pi) * np.cos(np.pi * x + y))

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return np.exp(2 * (x + y)) / 8 - np.exp(2 * (x + y)) / (16 * (1 + np.pi ** 2)) * (
                (1 - np.pi) * np.cos(2 * np.pi * x + 2 * y) + (1 + np.pi) * np.sin(2 * np.pi * x + 2 * y))


class TensorSineCosine(TestFunction2D):
    name = "\\sin(2x) \\cdot \\cos(\\pi y + 1)"
    name_cli = "sin(2x) * cos(pi y + 1)"

    def evaluate(self, x: float, y: float) -> float:
        return np.sin(2 * x) * np.cos(np.pi * y + 1)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return - (np.cos(2 * x) * np.sin(np.pi * y + 1)) / (2 * np.pi)

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (x * (np.pi * y + 1)) / (4 * np.pi) - ((np.pi * y + 1) * np.sin(4 * x)) / (16 * np.pi) + (
                x * np.sin(2 * (np.pi * y + 1))) / (8 * np.pi) - (np.sin(4 * x) * np.sin(2 * (np.pi * y + 1))) / (
                32 * np.pi)


class CosineProduct(TestFunction2D):
    name = "\\cos(x+y) \\cdot \\cos(x y)"
    name_cli = "cos(x + y) * cos(x * y)"

    def evaluate(self, x: float, y: float) -> float:
        return np.cos(x + y) * np.cos(x * y)

    def evaluate_integral(self, x: float, y: float) -> float:
        integrand = lambda t, s: np.cos(s + t) * np.cos(s * t)

        val, err = dblquad(integrand, 0.0, x, lambda s: 0.0, lambda s: y)
        return val

    # 1/2 ( - sin(1) ci1 - sin(1) ci2 + cos(1) si1 + cos(1) si2)
    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        integrand = lambda t, s: (np.cos(s + t) * np.cos(s * t)) ** 2

        val, err = dblquad(integrand, 0.0, x, lambda s: 0.0, lambda s: y)
        return val


class Polynomial(TestFunction2D):
    name = "2xy\\cdot (x-2y)"
    name_cli = "2xy * (x - 2y)"

    def evaluate(self, x: float, y: float) -> float:
        return 2 * x * y * (x - 2 * y)

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (x ** 3 * y ** 2) / 3 - (2 * x ** 2 * y ** 3) / 3

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (4 * x ** 5 * y ** 3) / 15 - x ** 4 * y ** 4 + (16 * x ** 3 * y ** 5) / 15


class TaylorCosine(TestFunction2D):
    name = "1 - \\frac {x^2}{2} - y^2 - xy + \\frac{x^3}{30} + \\frac{x^2 y}{8} + \\frac{xy^3}{6} - \\frac{x^2 y^2}{4} + \\frac{y^4}{24}"
    name_cli = "1 - x^2 / 2 - y^2 - x * y + x^3 / 30 + x^2 * y / 8 + x * y^3 / 6 - x^2 * y^2 / 4 + y^4 / 24"

    def evaluate(self, x: float, y: float) -> float:
        return 1 - x ** 2 / 2 - y ** 2 - x * y + x ** 3 / 30 + x ** 2 * y / 8 + x * y ** 3 / 6 - x ** 2 * y ** 2 / 4 + y ** 4 / 24

    def evaluate_integral(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return x * y - (x ** 3) * y / 6 - x * (y ** 3) / 6 - (x ** 2) * (y ** 2) / 4 + (x ** 4) * y / 120 + (
                x ** 3) * (y ** 2) / 48 + (x ** 2) * (y ** 4) / 48 - (x ** 3) * (y ** 3) / 36 + x * (
                y ** 5) / 120 - x * (y ** 3) / 6

    def evaluate_integral_squared(self, x: float | np.ndarray, y: float | np.ndarray) -> float:
        return (1 / 6300) * x ** 7 * y + (-1 / 1080) * x ** 6 * y ** 3 + (1 / 1440) * x ** 6 * y ** 2 + (
                -1 / 180) * x ** 6 * y + (1 / 400) * x ** 5 * y ** 5 + (-37 / 14400) * x ** 5 * y ** 4 + (
                17 / 960) * x ** 5 * y ** 3 + (-23 / 1200) * x ** 5 * y ** 2 + (1 / 20) * x ** 5 * y + (
                -1 / 288) * x ** 4 * y ** 6 + (1 / 450) * x ** 4 * y ** 5 + (1 / 48) * x ** 4 * y ** 4 + (
                -19 / 720) * x ** 4 * y ** 3 + (1 / 8) * x ** 4 * y ** 2 + (1 / 60) * x ** 4 * y + (
                1 / 3024) * x ** 3 * y ** 7 + (1 / 1728) * x ** 3 * y ** 6 + (1 / 120) * x ** 3 * y ** 5 + (
                -1 / 48) * x ** 3 * y ** 4 + (1 / 6) * x ** 3 * y ** 3 + (1 / 24) * x ** 3 * y ** 2 + (
                -1 / 3) * x ** 3 * y + (1 / 1152) * x ** 2 * y ** 8 + (-5 / 144) * x ** 2 * y ** 6 + (
                7 / 24) * x ** 2 * y ** 4 + (-1 / 2) * x ** 2 * y ** 2 + (1 / 5184) * x * y ** 9 + (
                -1 / 84) * x * y ** 7 + (13 / 60) * x * y ** 5 + (-2 / 3) * x * y ** 3 + (1) * x * y
