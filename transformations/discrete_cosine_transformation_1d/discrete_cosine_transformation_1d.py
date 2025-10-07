import numpy as np

from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.test_functions import TestFunction
from utility.templates.base_transformations import Transformation1D
from utility.test_functions_2d import ExponentialAdd

if __name__ == "__main__":
    from discrete_cosine_1d import DiscreteCosineFunction
else:
    from .discrete_cosine_1d import DiscreteCosineFunction
from utility.test_functions_1d import Quadratic, Exponential


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

# n = 8
# f = Exponential()
# keira = DiscreteCosineTransformation(n, f)
# # keira.plot_base_matrix()
# walcoef = keira.get_coefficients_integration_orthonormal()
#
# t_vals = keira.sample_transform(walcoef)
# f_vals = f.sample()
# keira.plot_transformation(t_vals, f_vals)
# keira.plot_coefficients(walcoef)
# n = 5
# f = Quadratic()
# keira = DiscreteCosineTransformation(n, f)
# walcoef = keira.get_coefficients_integration()
# f_vals = f.sample()
# t_vals = keira.sample_transform(walcoef)
# keira.plot_transformation(t_vals, f_vals)
# keira.plot_coefficients(walcoef)
