import numpy as np
from numpy import ndarray

import utility.utils as u
from utility.templates.base_functions import DiscreteBaseFunction2D
from utility.templates.base_transformations import Transformation2D
from utility.test_functions_2d import QuadraticMax

if __name__ == "__main__":  # "if" here so that import is correct depending on cli or file execution
    from walsh_function_2d import WalshFunction2D
else:
    from .walsh_function_2d import WalshFunction2D
from utility.templates.test_functions import TestFunction

u.latex_font()


class WalshTransformation2D(Transformation2D):

    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1, dynamic: bool = False):
        """
        Initialize a Walsh transformation for a given function.
        The Walsh-Functions are stored in a 2D matrix, where each entry represents all values of the corresponding Walsh
        function.
        The Walsh (interpolation) matrix is built by flattening all the values of Walsh function i,j and storing them in
        line i+j*8 of the matrix.
        :param n: 2^n is the number of Walsh functions per dimension.
        :param function: The function to transform.
        """
        super().__init__(n, function, boundary_n)

        if dynamic:
            dynamic_order_x, dynamic_order_y = self.get_dynamic_order_sepdim()
            base_functions: list[list[WalshFunction2D]] = [
                [WalshFunction2D(dynamic_order_x[i], dynamic_order_y[j], n)
                 for j in range(2 ** n)]
                for i in range(2 ** n)
            ]
        elif self.boundary_n > self.n:
            # Do the trick
            exp_boundaries = u.calculate_boundaries_exp(self.boundary_n)
            exp_boundary_order: np.ndarray = u.get_order_from_bounds(exp_boundaries)
            base_functions: list[list[WalshFunction2D]] = [
                [WalshFunction2D(exp_boundary_order[i], exp_boundary_order[j], self.boundary_n)
                 for j in range(2 ** n)]
                for i in range(2 ** n)
            ]
        else:
            dyadic_order = u.sequency_to_dyadic(n)

            # exp_boundaries = u.calculate_boundaries_exp(self.boundary_n)
            # exp_boundary_order: np.ndarray = u.get_order_from_bounds(exp_boundaries)
            base_functions: list[list[WalshFunction2D]] = [
                # To use a different order, use order[i] and order[j] instead of i and j here
                #[WalshFunction2D(exp_boundary_order[i], exp_boundary_order[j], n)
                # [WalshFunction2D(cos_boundary_order[i], cos_boundary_order[j], n)
                [WalshFunction2D(dyadic_order[i], dyadic_order[j], n)
                #[WalshFunction2D(i, j, n)
                 for j in range(2 ** n)]
                for i in range(2 ** n)
            ]
        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Walsh"

    @property
    def base_functions(self) -> list[list[DiscreteBaseFunction2D]]:
        return self._base_functions

    def get_dynamic_order_sepdim(self) -> tuple[list[int], list[int]]:
        """
        Get a dynamic ordering of base functions based on the 1-D coefficients of the transformation.
        This assumes the base functions to be orthonormal.
        :return: A list with indices of base functions.
        """
        integrals: list[list[float]] = self.get_function_integral_matrix()
        epsilon = 1e-5

        x_order = range(2 ** self.n)
        index_x: int = 0

        y_order = range(2 ** self.n)
        index_y: int = 0

        while True:
            # Both directions either ran into troubles or have found a good ordering
            if index_x == -1 and index_y == -1:
                return x_order, y_order
            # If we have tried all rows, give up
            if index_x >= 2 ** self.n:
                index_x = -1
            if index_y >= 2 ** self.n:
                index_y = -1

            if index_x != -1 and index_x < 2 ** self.n:
                x_coef: list[float] = []
                x_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(i, index_x, self.n) for i in range(2 ** self.n)]
                # Get coefficients of x-base-functions
                for i, phi in enumerate(x_base):
                    coef = float(np.sum(np.array(phi.values) * np.array(integrals)))
                    x_coef.append(abs(coef))

                xmin, xmin2, xmax2, xmax = np.partition(x_coef, [0, 1, -2, -1])[[0, 1, -2, -1]]

                # Check both first and last
                if xmax - xmin < epsilon or xmax2 - xmin2 < epsilon:
                    # Too little variation
                    index_x += 1
                else:
                    # Variation ok, so keep it this way
                    x_order = list(np.argsort(np.abs(x_coef))[::-1])
                    for x in range(len(x_order)):
                        x_order[x] = int(x_order[x])
                    index_x = -1
            if index_y != -1 and index_y < 2 ** self.n:
                y_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(index_y, j, self.n) for j in range(2 ** self.n)]
                y_coef: list[float] = []
                # Get coefficients of y-base-functions
                for j, phi in enumerate(y_base):
                    coef = float(np.sum(np.array(phi.values) * np.array(integrals)))
                    # print(j, walcoef)
                    y_coef.append(abs(coef))

                ymin, ymin2, ymax2, ymax = np.partition(y_coef, [0, 1, -2, -1])[[0, 1, -2, -1]]

                if ymax - ymin < epsilon or ymax2 - ymin2 < epsilon:
                    index_y += 1
                else:
                    y_order = list(np.argsort(np.abs(y_coef))[::-1])
                    for y in range(len(y_order)):
                        y_order[y] = int(y_order[y])
                    index_y = -1

# f = QuadraticMax()
# n = 3
# walter = WalshTransformation2D(n, f)
# walter.plot_base_matrix()
# walcoef = walter.get_coefficients_integration()
# #print(min(abs(walcoef[walcoef > 0])), max(abs(walcoef)))
# t_vals = walter.sample_transform(walcoef)
# f_vals = f.sample()
# f.plot()
# #walter.plot_coefficients(walcoef)
# walter.plot_transformation(t_vals, f_vals)
