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

        # outer product of 1D values
        # values = np.outer(np.array(wavelet_x.values), np.array(wavelet_y.values))
        #
        # self._values = values.T
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

    # @property
    # def values(self) -> np.ndarray:
    #     return self._values

    @property
    def max_scale(self) -> float:
        return self._max_scale

    # def __init__(self, order_x: int, order_y: int, level_x: int, level_y: int, n: int, index_x: int = -1,
    #              index_y: int = -1, intervals: list[float] = None) -> None:
    #     """
    #     Initialize a two-dimensional Haar wavelet.
    #     :param order_x:
    #     :param order_y:
    #     :param level_x:
    #     :param level_y:
    #     :param n:
    #     :param index_x:
    #     :param index_y:
    #     :param intervals:
    #     """
    #     self.n = n
    #     if index_x == -1 or index_y == -1:
    #         if level_x == 0 and level_y == 0:
    #             self.order_x = 1
    #             self.order_y = 1
    #         else:
    #             self.order_x = order_x
    #             self.order_y = order_y
    #         self.level_x = level_x
    #         self.level_y = level_y
    #
    #         if intervals is None:
    #             haar_wavelet_x = HaarWavelet(order_x, level_x, n)
    #             haar_wavelet_y = HaarWavelet(order_y, level_y, n)
    #             self.intervals = haar_wavelet_x.intervals
    #         else:
    #             haar_wavelet_x = HaarWavelet(order_x, level_x, n, intervals=intervals)
    #             haar_wavelet_y = HaarWavelet(order_y, level_y, n, intervals=intervals)
    #             self.intervals = intervals
    #     else:  # Use Wavelet ordering by index
    #         if intervals is None:
    #             haar_wavelet_x = HaarWavelet(order_x, level_x, n, index=index_x)
    #             haar_wavelet_y = HaarWavelet(order_y, level_y, n, index=index_y)
    #             self.intervals = haar_wavelet_x.intervals
    #         else:
    #             haar_wavelet_x = HaarWavelet(order_x, level_x, n, index=index_x, intervals=intervals)
    #             haar_wavelet_y = HaarWavelet(order_y, level_y, n, index=index_y, intervals=intervals)
    #         self.order_x = haar_wavelet_x.order
    #         self.order_y = haar_wavelet_y.order
    #         self.level_x = haar_wavelet_x.level
    #         self.level_y = haar_wavelet_y.level
    #
    #     values = np.zeros((len(haar_wavelet_x.values), len(haar_wavelet_y.values)))
    #
    #     for i in range(len(haar_wavelet_x.values)):
    #         for j in range(len(haar_wavelet_y.values)):
    #             values[j][i] = haar_wavelet_x.values[i] * haar_wavelet_y.values[j]
    #     self.values = values
    #
    # def evaluate(self, x: float, y: float) -> int:
    #     # if any((x < 0 or x > 1) for x in x) or any((y < 0 or y > 1) for y in y):
    #     #    raise ValueError("x and y must be between 0 and 1.")
    #
    #     for i in range(len(self.intervals) - 1):
    #         if self.intervals[i] <= x <= self.intervals[i + 1]:
    #             x_index = i
    #         if self.intervals[i] <= y <= self.intervals[i + 1]:
    #             y_index = i
    #     return int(self.values[x_index][y_index])
    #
    # def evaluate_vec(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     if x.shape != y.shape:
    #         raise ValueError("Shape of x and y must be the same.")
    #     ratio = x.shape[0] / len(self.values)
    #     if int(ratio) - ratio != 0:
    #         # If the ration is not an integer, then the samples are not correct
    #         raise ValueError("Number of samples must be a power of two.")
    #     long_values = np.repeat(np.repeat(self.values, ratio, axis=1), ratio, axis=0)
    #     return long_values
    #
    # def plot(self) -> None:
    #     """
    #     Plots the 2-dimensional Wavelet function on the unit square as a heatmap.
    #     :return:
    #     """
    #     plt.imshow(self.values, cmap="gray", interpolation="none", extent=[0, 1, 0, 1])
    #     plt.colorbar(label="Value")
    #     # Not advisable for n > 3
    #     # plt.xticks(np.linspace(0, 1, 2 ** self.n + 1), labels=self.intervals)
    #     # plt.yticks(np.linspace(0, 1, 2 ** self.n + 1), labels=self.intervals)
    #     f_per_level_x = 2 ** (self.level_x - 1) if self.level_x > 0 else 1
    #     f_per_level_y = 2 ** (self.level_y - 1) if self.level_y > 0 else 1
    #     plt.title(
    #         f"Level {self.level_x} Wavelet {self.order_x} of {f_per_level_x} in $x$ and level {self.level_y} Wavelet {self.order_y} of {f_per_level_y} in $y$.")
    #     plt.xlabel("$x$")
    #     plt.ylabel("$y$")
    #     plt.show()

# ORDER LEVEL VERSION
# n = 2
# helena = HaarWavelet2D(0, 0, 0, 0, n)
# helena.plot()
# for i in range(n + 1):  # level x
#     for j in range(n + 1):  # level y
#         if i == 0 and j == 0:
#             continue
#         i_prime = i if i >= 1 else 1
#         j_prime = j if j >= 1 else 1
#         for k in range(2 ** (i_prime - 1)):  # order x
#             for l in range(2 ** (j_prime - 1)):  # order y
#                 helena = HaarWavelet2D(k, l, i, j, n)
#                 helena.plot()

# INDEX VERSION
# n = 2
# for i in range(2 ** n):
#     for j in range(2 ** n):
#         helena = HaarWavelet2D(0, 0, 0, 0, n, index_x=i, index_y=j)
#         helena.plot()

# ============ new version =========
# n = 2
# for i in range(2 ** n):
#     for j in range(2 ** n):
#         helena = HaarWavelet2D(i, j, n)
#         helena.plot()
