from utility.templates.base_functions import DiscreteBaseFunction1D, DiscreteBaseFunction2D
from utility.templates.test_functions import TestFunction
from utility.templates.base_transformations import Transformation1D

if __name__ == "__main__":
    from haar_wavelet_1d import HaarWavelet
else:
    from .haar_wavelet_1d import HaarWavelet


class WaveletTransformation(Transformation1D):
    def __init__(self, n: int, function: TestFunction, boundary_n: int = -1):
        super().__init__(n, function, boundary_n)

        base_functions: list[HaarWavelet] = []
        for i in range(2 ** n):
            base_functions.append(HaarWavelet(i, n))
        self._base_functions = base_functions

    @property
    def name(self) -> str:
        return "Wavelet"

    @property
    def base_functions(self) -> list[DiscreteBaseFunction1D] | list[list[DiscreteBaseFunction2D]]:
        return self._base_functions
