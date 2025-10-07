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

        dyadic_order = u.sequency_to_dyadic(n)

        self.exp_boundaries = u.calculate_boundaries_exp(self.boundary_n)
        self.boundary_order_exp: np.ndarray = u.get_order_from_bounds(self.exp_boundaries)

        base_functions: list[WalshFunction] = []
        # Doing the trick
        if self.boundary_n > self.n:
            for i in range(2 ** n):
                base_functions.append(WalshFunction(self.boundary_order_exp[i], self.boundary_n))
        else:
            for i in range(2 ** n):
                # To use a different ordering use WalshFunction(ordering[i], n)
                base_functions.append(WalshFunction(dyadic_order[i], n))

        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Walsh"

    @property
    def base_functions(self) -> list[DiscreteBaseFunction1D]:
        return self._base_functions
