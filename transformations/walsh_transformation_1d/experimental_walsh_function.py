import numpy as np
import matplotlib.pyplot as plt

from utility import utils as u

from utility.templates.base_functions import DiscreteBaseFunction1D


class ExperimentalWalshFunction(DiscreteBaseFunction1D):
    """
    This class represents a Walsh function.
    """

    def __init__(self, order: int, n: int, intervals_b: list[list[int]] = None):
        """
        Implements a Walsh function of the given order, where 2^n is the maximum number of intervals and Walsh functions.
        :param order: (Sequency-)order of the Walsh function It is 0 ≤ order < 2^n.
        :param n: 2^n is the maximum number of intervals.
        :param intervals_b: The number of each interval in binary.
        By default, the binary-interval number will be calculated upon initialization.
        """
        super().__init__(order, n)

        if intervals_b is None:
            intervals_b: list[list[int]] = []
        order_b: list[int] = u.int_to_binary(order, n + 1)
        for i in range(2 ** n):
            intervals_b.append(u.int_to_binary(i, n))
        values: list[int] = []
        order_graycode: list[int] = []
        # XOR the bits of order, they are always the same. This yields the graycode of order
        for i in range(len(order_b) - 1):
            order_graycode.append(order_b[i] ^ order_b[i + 1])
        # Reverse xor'd order, instead of all intervals
        order_graycode.reverse()

        for i in range(2 ** n):
            # value at i-th interval
            exponent = np.dot(np.array(intervals_b[i]), np.array(order_graycode))
            values.append((-1) ** int(exponent))
        self._values = values

        self.nu, self.mu = u.calc_nu_mu(self.order)

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
        return "Walsh"

    @property
    def plot_title(self) -> str:
        return f"Walsh Function {self.order + 1} of {2 ** self.n}"

    @property
    def values(self) -> list[float] | list[int]:
        return self._values

    @property
    def max_scale(self) -> float:
        return 1.61803398875


# n = 3
# dyadic_order = u.sequency_to_dyadic(n)
#
# for i in range(2 ** n):
#     violet = ExperimentalWalshFunction(dyadic_order[i], n)
#     violet.plot()
