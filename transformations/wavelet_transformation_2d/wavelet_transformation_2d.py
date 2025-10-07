import numpy as np

from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.base_transformations import Transformation2D
from utility.templates.test_functions import TestFunction, Image
from utility.test_functions_2d import QuadraticMax, TaylorCosine, CosineXSquare

if __name__ == "__main__":  # "if" here so that import is correct depending on cli or file execution
    from haar_wavelet_2d import HaarWavelet2D
else:
    from .haar_wavelet_2d import HaarWavelet2D


class WaveletTransformation2D(Transformation2D):

    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        """
        Initializes a two-dimensional Wavelet transformation for the given function by initializing all 2^n Haar wavelets.
        :param n: 2^n is the number of Haar wavelets per dimension and n is the maximum level.
        :param function: The function to transform.
        """
        super().__init__(n, function, boundary_n)

        # To use a different ordering, use order[i] and order[j] instead of i and j
        base_functions: list[list[HaarWavelet2D]] = [
            [HaarWavelet2D(i, j, self.n)
             for j in range(2 ** self.n)]
            for i in range(2 ** self.n)]
        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Wavelet"

    @property
    def base_functions(self) -> list[DiscreteBaseFunction1D] | list[list[DiscreteBaseFunction2D]]:
        return self._base_functions

#
# f = Image("../../images/prime_x.png", 256)
# n = 8
# helena = WaveletTransformation2D(n, f)
#
# coef = helena.get_coefficients_integration_orthonormal()
# t_vals = helena.sample_transform(coef)
# helena.plot_transformation(t_vals, subtitle="og")
#
# coef_sparse = helena.discard_coefficients_sparse_grid(coef, 12)
# sparse_t_vals = helena.sample_transform(coef_sparse)
# helena.plot_transformation(sparse_t_vals, subtitle="sparse")
# helena.plot_coefficients(coef_sparse, subtitle="sparse")
#
# coef_opt = helena.discard_coefficients_percentage(coef, 81.25)
# opt_t_vals = helena.sample_transform(coef_opt)
# helena.plot_transformation(opt_t_vals, subtitle="opt")
# helena.plot_coefficients(coef_opt, subtitle="opt")
