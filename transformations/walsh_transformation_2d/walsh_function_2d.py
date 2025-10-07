import numpy as np

from transformations.walsh_transformation_1d.walsh_function_1d import WalshFunction
from utility.templates.base_functions import DiscreteBaseFunction2D

class WalshFunction2D(DiscreteBaseFunction2D):

    def __init__(self, order_x: int, order_y: int, n: int):
        super().__init__(order_x, order_y, n)

        walsh_x: WalshFunction = WalshFunction(order_x, n)
        walsh_y: WalshFunction = WalshFunction(order_y, n)

        self._values_x = walsh_x.values
        self._values_y = walsh_y.values

        self.level_x = walsh_x.level
        self.level_y = walsh_y.level
        self.shift_x = walsh_x.shift
        self.shift_y = walsh_y.shift

    @property
    def name(self) -> str:
        return "Walsh"

    @property
    def values_x(self) -> list[int | float]:
        return self._values_x

    @property
    def values_y(self) -> list[int | float]:
        return self._values_y

    @property
    def plot_title(self) -> str:
        return f"Walsh function of order {self.order_x} in $x$ and {self.order_y} in $y$ out of {2 ** self.n}."

    @property
    def max_scale(self) -> float:
        return 1

    @property
    def square_integral(self) -> float:
        """
        The integral of the squared base function on the interval [0,1]ᵈ.
        This is useful for calculating the base functions coefficient.

        IMPORTANT: This method is only implemented for Walsh functions for consistency across base functions.
        It is recommended to use get_coefficients_integration_orthonormal(), which does not need the square integrals.
        :return: The value of the integral of the squared base function on the interval [0,1]ᵈ.
        """
        return 1
