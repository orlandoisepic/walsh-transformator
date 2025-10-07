if __name__ == "__main__":  # "if" here so that import is correct depending on cli or file execution
    from discrete_cosine_function_2d import DiscreteCosineFunction2D
else:
    from .discrete_cosine_function_2d import DiscreteCosineFunction2D

from utility.templates.base_functions import DiscreteBaseFunction2D
from utility.templates.test_functions import TestFunction, Image
from utility.templates.base_transformations import Transformation2D


class DiscreteCosineTransformation2D(Transformation2D):

    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        super().__init__(n, function, boundary_n)

        base_functions: list[list[DiscreteCosineFunction2D]] = [
            [DiscreteCosineFunction2D(i, j, self.boundary_n)
             for j in range(2 ** n)]
            for i in range(2 ** n)
        ]
        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Discrete Cosine"

    @property
    def base_functions(self) -> list[list[DiscreteBaseFunction2D]]:
        return self._base_functions
