import numpy as np

from utility import utils as u
from utility.templates.base_functions import DiscreteBaseFunction1D

u.latex_font()


class HaarWavelet(DiscreteBaseFunction1D):
    """
    This class implements a one-dimensional Haar wavelet.
    """

    def __init__(self, order: int, n: int):
        """
         Initialize a Haar wavelet. The wavelet has an order and a level.
         :param order: The order of the wavelet, i.e., the number if the Wavelets are enumerated starting from 0.
         :param n: A parameter controlling the maximum level and detail of the wavelet.
         It is 1 <= level <= n and 0 <= shift <= 2^level - 1
         """
        super().__init__(order, n)

        level_offset: int = 2 ** (n - self.level)
        self.j = n - self.level + 1

        # Each entry is the value in the i-th interval
        values: list[float] = [0 for _ in range(2 ** n)]

        # Scale of the Wavelets values
        scale: float = 2 ** ((self.level - 1) / 2)  # for orthonormality # 2^(j/2 -1/2)=√2^(j-1)
        self.scale: float = scale

        # If using only the necessary number of intervals, then the Wavelet is equal to its scale in interval 2(order)-1
        # and -scale in the next interval. However, when always using 2^n values, the intervals have to be scaled,
        # i.e., an interval of a wavelet of level l has width 2^(n-l).
        if self.level == 0:
            self._values = [1 for _ in range(2 ** n)]
            self.scale = 1
        else:
            for i in range(level_offset):
                values[(2 * self.shift) * level_offset + i] = scale
            for i in range(level_offset):
                values[(2 * self.shift + 1) * level_offset + i] = -scale
            self._values = values

        # if order != 0:
        #     values = u.affine_transform(self.values, lower=-1 - 3.5 * order, upper=1 - 3.5 * order)
        # else:
        #     v = -0.5
        #     values = [v for _ in range(2 ** self.n)]
        # print(f"\\addplot[blue,thick,const plot] coordinates ""{")
        # for i, val in enumerate(values):
        #     print(f"({(i * 0.0625).__round__(4)},{val.__round__(3)})")
        # print(f"({(8 * 0.0625).__round__(4)},{(values[-1]).__round__(3)})")
        # print("}; %"f" {order}")

    @property
    def name(self) -> str:
        return "Wavelet"

    @property
    def plot_title(self) -> str:
        f_per_level: int = 2 ** (self.level - 1) if self.level > 0 else 1
        return f"Level {self.level} Wavelet {self.shift + 1} of {f_per_level}"

    @property
    def values(self) -> list[float] | list[int]:
        return self._values

    @property
    def max_scale(self) -> float:
        return 2 ** (self.n / 2 - 1 / 2) + 1

# SHIFT LEVEL VERSION
# n = 2
# helena = HaarWavelet(0, 0, n)
# helena.plot()
# print(0,0)
# for i in range(1, n + 1):  # level=0,1,...,n, but 0 is separate
#     for j in range(2 ** (i - 1)):  # shift: 0,...,2^(level-1)
#         helena = HaarWavelet(0,n,level=i, shift=j)
#         print(i,j)
#         helena.plot()
#
# INDEX VERSION
# n = 3
# for i in range(2 ** n):
#     helena = HaarWavelet(i, n)
#     helena.plot()
