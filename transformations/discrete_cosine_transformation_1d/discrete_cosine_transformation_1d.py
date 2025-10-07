from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.test_functions import TestFunction
from utility.templates.base_transformations import Transformation1D

if __name__ == "__main__":
    from discrete_cosine_1d import DiscreteCosineFunction
else:
    from .discrete_cosine_1d import DiscreteCosineFunction


class DiscreteCosineTransformation(Transformation1D):
    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        super().__init__(n, function, boundary_n)
        base_functions: list[DiscreteCosineFunction] = []
        for i in range(2 ** n):
            base_functions.append(DiscreteCosineFunction(i, self.boundary_n))
        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Discrete Cosine"

    @property
    def base_functions(self) -> list[DiscreteBaseFunction1D] | list[list[DiscreteBaseFunction2D]]:
        return self._base_functions
