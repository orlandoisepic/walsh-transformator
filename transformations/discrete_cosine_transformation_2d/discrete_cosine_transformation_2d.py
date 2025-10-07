import numpy as np

from utility.test_functions_2d import ExponentialAdd, QuadraticMax, Cosine, TaylorCosine

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
#
# f = Image("../../images/prime_x.png", 256)
# n= 8
# keira = DiscreteCosineTransformation2D(n, f)
#
# coef = keira.get_coefficients_integration_orthonormal()
# t_vals = keira.sample_transform(coef)
# keira.plot_transformation(t_vals, subtitle="og")
#
# coef_sparse = keira.discard_coefficients_sparse_grid(coef, 12)
# sparse_t_vals = keira.sample_transform(coef_sparse)
# keira.plot_transformation(sparse_t_vals, subtitle="sparse")
# keira.plot_coefficients(coef_sparse, subtitle="sparse")
#
# coef_opt = keira.discard_coefficients_percentage(coef, 81.25)
# opt_t_vals = keira.sample_transform(coef_opt)
# keira.plot_transformation(opt_t_vals, subtitle="opt")
# keira.plot_coefficients(coef_opt, subtitle="opt")
