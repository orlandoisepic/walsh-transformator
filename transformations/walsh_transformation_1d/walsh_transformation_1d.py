import numpy as np

import utility.utils as u
from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.base_transformations import Transformation1D
from utility.test_functions_1d import Cosine, Exponential, InverseSine
from utility.utils import sequency_to_dyadic

if __name__ == "__main__":
    from walsh_function_1d import WalshFunction
else:
    from .walsh_function_1d import WalshFunction
from utility.templates.test_functions import TestFunction
from utility.test_functions_1d import Quadratic


class WalshTransformation(Transformation1D):
    """
    This class implements a Walsh transformation for a given function.
    """

    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        super().__init__(n, function, boundary_n)

        # To use a different ordering use WalshFunction(ordering[i],n)
        dyadic_order = u.sequency_to_dyadic(n)

        # cos_boundaries = u.calculate_boundaries_cos(n)
        # cos_boundary_order: np.ndarray = u.get_order_from_bounds(cos_boundaries)
        #
        # cos2pi_boundaries = u.calculate_boundaries_cos2pi(n)
        # cos2pi_boundary_order: np.ndarray = u.get_order_from_bounds(cos2pi_boundaries)

        self.exp_boundaries = u.calculate_boundaries_exp(self.boundary_n)
        self.boundary_order_exp: np.ndarray = u.get_order_from_bounds(self.exp_boundaries)

        base_functions: list[WalshFunction] = []
        # Doing the trick
        if self.boundary_n > self.n:
            for i in range(2 ** n):
                base_functions.append(WalshFunction(self.boundary_order_exp[i], self.boundary_n))
        else:
            for i in range(2 ** n):
                base_functions.append(WalshFunction(dyadic_order[i], n))

        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Walsh"

    @property
    def base_functions(self) -> list[DiscreteBaseFunction1D]:
        return self._base_functions

# n = 8
# b = n + 4
# f = InverseSine()
# violet = WalshTransformation(n, f)
# walter = WalshTransformation(n, f, boundary_n=b)
# violet_coef = violet.get_coefficients_integration_orthonormal()
# walter_coef = walter.get_coefficients_integration_orthonormal()
# violet_f_vals = f.sample()
# walter_f_vals = f.sample() if b <= 10 else f.sample(2**b)
# violet_t_vals = violet.sample_transform(violet_coef)
# walter_t_vals = walter.sample_transform(walter_coef)
# violet.plot_transformation(violet_t_vals, violet_f_vals)
# walter.plot_transformation(walter_t_vals, walter_f_vals, subtitle="boundary n")
# violet.plot_coefficients(violet_coef, subtitle="normal")
# walter.plot_coefficients(walter_coef, subtitle="boundary n")
# print(violet.get_squared_l2_error(violet_coef))
# print("boundary n", walter.get_squared_l2_error(walter_coef))
