import numpy as np

from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.test_functions import TestFunction
from utility.templates.base_transformations import Transformation1D

if __name__ == "__main__":
    from discrete_sine_1d import DiscreteSineFunction
else:
    from .discrete_sine_1d import DiscreteSineFunction
from utility.test_functions_1d import Quadratic, Exponential, ExponentialSine


class DiscreteSineTransformation(Transformation1D):
    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        super().__init__(n, function, boundary_n)

        base_functions: list[DiscreteSineFunction] = []
        for i in range(2 ** n):
            base_functions.append(DiscreteSineFunction(i, self.n))
        self._base_functions = base_functions


    @property
    def name(self) -> str:
        return "Discrete Sine"

    @property
    def base_functions(self) -> list[DiscreteBaseFunction1D] | list[list[DiscreteBaseFunction2D]]:
        return self._base_functions


# n = 8
# f = ExponentialSine()
# sally = DiscreteSineTransformation(n, f)
# sally.plot_base_matrix()
# walcoef = sally.get_coefficients_integration_orthonormal()
#
# t_vals = sally.sample_transform(walcoef)
# f_vals = f.sample()
# sally.plot_transformation(t_vals, f_vals)
# sally.plot_coefficients(walcoef)
# sally.plot_coefficients(walcoef, sorted=True)
