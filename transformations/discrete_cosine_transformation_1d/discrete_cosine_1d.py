import numpy as np

import utility.utils as u
from utility.templates.base_functions import DiscreteBaseFunction1D


class DiscreteCosineFunction(DiscreteBaseFunction1D):
    """
    This class represents a discrete cosine function.
    """

    def __init__(self, order: int, n: int):
        """
        Implements a discrete cosine function with 2^n samples of the given order (frequency).
        :param order: The frequency of the discrete cosine function.
        :param n: 2^n is the number of samples of the discrete cosine function.
        """
        super().__init__(order, n)

        values: list[float] = []
        if order == 0:
            values = [1 for _ in range(2 ** self.n)]
        else:
            for i in range(2 ** n):
                # multiplying with √2 makes the function orthonormal
                values.append(float(np.cos(np.pi * (2 * i + 1) * self.order / (2 ** (self.n + 1))) * np.sqrt(2)))

        self._values = values

    @property
    def name(self) -> str:
        return "Discrete cosine"

    @property
    def plot_title(self) -> str:
        return f"{self.name} function {self.order + 1} of {2 ** self.n}"

    @property
    def values(self) -> list[float] | list[int]:
        return self._values

    @property
    def max_scale(self) -> float:
        return 2.61
