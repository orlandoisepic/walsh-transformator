from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.base_transformations import Transformation2D
from utility.templates.test_functions import TestFunction

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
