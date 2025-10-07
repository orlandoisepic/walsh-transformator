import numpy as np

from transformations.wavelet_transformation_1d.haar_wavelet_1d import HaarWavelet
from utility.templates.base_functions import DiscreteBaseFunction2D


class HaarWavelet2D(DiscreteBaseFunction2D):
    """
    This class represents a two-dimensional Haar wavelet.
    """

    def __init__(self, order_x: int, order_y: int, n: int):
        super().__init__(order_x, order_y, n)

        wavelet_x: HaarWavelet = HaarWavelet(order_x, n)
        wavelet_y: HaarWavelet = HaarWavelet(order_y, n)

        self.level_x = wavelet_x.level
        self.level_y = wavelet_y.level
        self.shift_x = wavelet_x.shift
        self.shift_y = wavelet_y.shift

        self._values_x = wavelet_x.values
        self._values_y = wavelet_y.values
        self._max_scale = np.max(self.values)
        self.scale = wavelet_x.scale * wavelet_y.scale

    @property
    def name(self) -> str:
        return "Wavelet"

    @property
    def plot_title(self) -> str:
        f_per_level_x = 2 ** (self.level_x - 1) if self.level_x > 0 else 1
        f_per_level_y = 2 ** (self.level_y - 1) if self.level_y > 0 else 1
        return f"Level {self.level_x} Wavelet {self.shift_x + 1} of {f_per_level_x} in $x$ and level {self.level_y} Wavelet {self.shift_y + 1} of {f_per_level_y} in $y$."

    @property
    def values_x(self) -> list[float]:
        return self._values_x

    @property
    def values_y(self) -> list[float]:
        return self._values_y

    @property
    def max_scale(self) -> float:
        return self._max_scale
