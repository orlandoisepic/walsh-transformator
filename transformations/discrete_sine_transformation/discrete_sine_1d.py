import numpy as np

from utility.templates.base_functions import DiscreteBaseFunction1D


class DiscreteSineFunction(DiscreteBaseFunction1D):
    """
    This class represents a discrete sine function.
    """

    def __init__(self, order: int, n: int):
        """
        Implements a discrete sine function with 2^n samples of the given order+1 (frequency).
        :param order: Adding 1 is the frequency of the discrete sine function.
        :param n: 2^n is the number of samples of the discrete sine function.
        """
        super().__init__(order, n)
        values: list[float] = []

        for i in range(2 ** n):
            # multiplying with √2 makes the function orthonormal
            values.append(float(np.sin(np.pi * (2 * i + 1) * (self.order + 1) / (2 ** (self.n + 1))) * np.sqrt(2)))

        if self.order == 2 ** self.n - 1:
            values /= np.sqrt(2)  # normalization of last base function

        self._values = values

    @property
    def name(self) -> str:
        return "Discrete sine"

    @property
    def plot_title(self) -> str:
        return f"{self.name} function {self.order + 1} of {2 ** self.n}"

    @property
    def values(self) -> list[float] | list[int]:
        return self._values

    @property
    def max_scale(self) -> float:
        return 2.61
