import numpy as np

from transformations.discrete_cosine_transformation_1d.discrete_cosine_1d import DiscreteCosineFunction
from utility.templates.base_functions import DiscreteBaseFunction2D


class DiscreteCosineFunction2D(DiscreteBaseFunction2D):

    def __init__(self, order_x: int, order_y: int, n: int):
        super().__init__(order_x, order_y, n)

        cosine_x: DiscreteCosineFunction = DiscreteCosineFunction(order_x, n)
        cosine_y: DiscreteCosineFunction = DiscreteCosineFunction(order_y, n)

        self._values_x = cosine_x.values
        self._values_y = cosine_y.values

        self._max_scale = np.max(self.values)

        self.level_x = cosine_x.level
        self.level_y = cosine_y.level
        self.shift_x = cosine_x.shift
        self.shift_y = cosine_y.shift

    @property
    def name(self) -> str:
        return "Discrete cosine"

    @property
    def plot_title(self) -> str:
        return f"{self.name} function {self.order_x + 1} of {2 ** self.n} in $x$ and {self.order_y + 1} of {2 ** self.n} in $y$."

    @property
    def values_x(self) -> list[float]:
        return self._values_x

    @property
    def values_y(self) -> list[float]:
        return self._values_y

    @property
    def max_scale(self) -> float:
        return self._max_scale

