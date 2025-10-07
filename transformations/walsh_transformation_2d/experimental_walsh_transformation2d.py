import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utility.utils as u
from utility.templates.base_functions import DiscreteBaseFunction2D
from utility.templates.base_transformations import Transformation2D
from utility.test_functions_2d import CosineProduct, TaylorCosine
from walsh_function_2d import WalshFunction2D
from utility.templates.test_functions import TestFunction, TestFunction2D, TestFunctionType, Image
import utility.test_functions_2d as tf

u.latex_font()


class ExperimentalWalshTransformation2D(Transformation2D):

    def __init__(self, n: int, function: TestFunction, dynamic: bool = False, boundary_n: int = -1):
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

        exp_bounds: list[float] = u.calculate_boundaries_exp(self.boundary_n)
        self.exp_order: list[int] = u.get_order_from_bounds(exp_bounds)

        dyad_order: list[int] = u.sequency_to_dyadic(n)

        if dynamic:
            if boundary_n == -1:
                dynamic_order_x, dynamic_order_y = self.get_dynamic_order_sepdim()
                base_functions: list[list[WalshFunction2D]] = [
                    [WalshFunction2D(dynamic_order_x[i], dynamic_order_y[j], n)
                     # [WalshFunction2D(dynamic_order_x[i], dynamic_order_y[j], n)
                     # [WalshFunction2D(i, j, n)
                     for j in range(2 ** n)]
                    for i in range(2 ** n)
                ]
            else:
                dynamic_order_x, dynamic_order_y = self.get_dynamic_order_sepdim(trick=True)
                base_functions: list[list[WalshFunction2D]] = [
                    [WalshFunction2D(dynamic_order_x[i], dynamic_order_y[j], self.boundary_n)
                     # [WalshFunction2D(dynamic_order_x[i], dynamic_order_y[j], n)
                     # [WalshFunction2D(i, j, n)
                     for j in range(2 ** n)]
                    for i in range(2 ** n)
                ]


        elif boundary_n != -1:
            base_functions: list[list[WalshFunction2D]] = [
                [WalshFunction2D(self.exp_order[i], self.exp_order[j], boundary_n)
                 # [WalshFunction2D(dynamic_order_x[i], dynamic_order_y[j], n)
                 # [WalshFunction2D(i, j, n)
                 for j in range(2 ** n)]
                for i in range(2 ** n)
            ]

        else:

            base_functions: list[list[WalshFunction2D]] = [
                # To use a different order, use order[i] and order[j] instead of i and j here
                [WalshFunction2D(dyad_order[i], dyad_order[j], n)
                 for j in range(2 ** n)]
                for i in range(2 ** n)
            ]
        self._base_functions = base_functions

    def plot_coefficients(self, coefficients: np.ndarray, subtitle: str = "", sorted: bool = False, first_n: int = 0,
                          cli: bool = False, fig: Figure = None, index: int = -1,
                          vmin: int = -1, vmax: int = -1) -> None:
        subtitle = "\n" + subtitle if subtitle else ""

        coefficients_to_plot = coefficients.copy()  # do not modify the original coefficients[]
        fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=False)
        if sorted:
            coefficients_to_plot[::-1] = np.sort(abs(np.array(coefficients_to_plot)))
            if first_n > 0:
                coefficients_to_plot = coefficients_to_plot[:first_n]
                subtitle += f"\nshowing {first_n} coefficients per dimension"
            plt.plot(abs(coefficients_to_plot), color="blue")
            plt.xscale("log")
            plt.yscale("log")
            subtitle += "\nsorted in descending order"
        else:
            coefficients_to_plot = np.reshape(coefficients_to_plot, (2 ** self.n, 2 ** self.n))
            if first_n > 0:
                coefficients_to_plot = coefficients_to_plot[:first_n, :first_n]
                subtitle += f"\nshowing {first_n} coefficients per dimension"
            original_coefficients = abs(
                self.get_coefficients_integration_orthonormal().reshape(2 ** self.n, 2 ** self.n)).T
            # vmin = original_coefficients.min()
            # vmin = original_coefficients[original_coefficients != 0].min()
            # vmax = original_coefficients.max()
            # print(vmin, vmax)
            im = ax.imshow(abs(coefficients_to_plot.T), cmap="inferno", norm=LogNorm(
                # vmin=1.920890529971686e-12, vmax=0.4396774653226998  # cos(2x+y)
                # vmin=1.196959198423997e-16 - 1.1e-16, vmax=1.6371253500460854 + 0.2  # ExpCos
                # vmax=0.37931285324984976, vmin=2.1378105461008698e-09,
                # vmin=8.131516293641283e-19, vmax=1.0938921864969489
                # vmin=3.2526065174565133e-16 - 3e-16, vmax=0.9993055555555554 + 0.1  # TaylorCosine
                vmin=1.8778990229281062e-06, vmax=123.49873352050781
            ))
            sparse_matrix = u.create_levelsum_matrix(self.n)
            mask = sparse_matrix > 12
            segments = u.boundary_segments_from_mask(mask)
            lc = LineCollection(segments, colors="black", linewidths=1.5, zorder=5)
            ax.add_collection(lc)

            # This shows the coefficients again in the zero area, but with lower alpha.
            if "sparse" in subtitle:
                original_coefficients[~mask] = np.nan
                cmap = plt.get_cmap("inferno").copy()
                cmap.set_bad(alpha=0.0)
                ax.imshow(original_coefficients, cmap=cmap, alpha=0.65, norm=LogNorm(
                    # vmin=3.2526065174565133e-16 - 3e-16, vmax=0.9993055555555554 + 0.1  # TaylorCos
                    # vmin=1.920890529971686e-12, vmax=0.4396774653226998 # Cos
                    # vmin=1.196959198423997e-16, vmax=1.6371253500460854  # ExpCos
                    vmin=1.8778990229281062e-06, vmax=123.49873352050781
                ))
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7.5%", pad=0.13)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            fig.subplots_adjust(right=0.89)
        title: str = (f"Coefficients of {self.name} transformation of\n"
                      f"$\\displaystyle {self.function.name}$\n"
                      f"with {2 ** self.n} {self.name} functions per dimension" + subtitle)
        # fig.suptitle(title)
        plt.savefig(f"{subtitle}.pdf", dpi=400, pad_inches=0.01)
        plt.show()

    @property
    def name(self) -> str:
        return "Walsh"

    @property
    def base_functions(self) -> list[list[DiscreteBaseFunction2D]]:
        return self._base_functions

    def get_dynamic_order_sepdim(self, trick: bool = False) -> tuple[list[int], list[int]]:
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
                # This forces 1D base functions to stay at their positions.
                # This could be helpful for non-tensor functions, but is not so for tensor functions,
                # as for tensors, perfect sorting would be destroyed.
                # For non-tensors, however, perfect sorting cannot be achieved anyway, and generally, 1D base functions are important.
                # if x_order[0] != 0:
                #     x_order.remove(0)
                #     x_order = [0] + x_order
                # if y_order[0] != 0:
                #     y_order.remove(0)
                #     y_order = [0] + y_order
                return x_order, y_order

            #print("current dynamic index", index_x, index_y)
            # If we have tried all rows, give up
            if index_x >= 2 ** self.n:
                index_x = -1
            if index_y >= 2 ** self.n:
                index_y = -1

            if index_x != -1 and index_x < 2 ** self.n:
                x_coef: list[float] = []
                if not trick:
                    x_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(i, index_x, self.n) for i in
                                                            range(2 ** self.n)]
                else:
                    # TODO
                    x_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(self.exp_order[i], index_x, self.boundary_n)
                                                            for i in
                                                            range(2 ** self.n)]
                # Get coefficients of x-base-functions
                for i, phi in enumerate(x_base):
                    coef = float(np.sum(np.array(phi.values) * np.array(integrals)))
                    x_coef.append(abs(coef))

                xmin: float = min(x_coef)
                xmax: float = max(x_coef)
                xmax2 = sorted(x_coef)[-2]
                xmin2 = sorted(x_coef)[2]

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
                if not trick:
                    y_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(index_y, j, self.n) for j in
                                                            range(2 ** self.n)]
                else:
                    # TODO
                    y_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(index_y, self.exp_order[j], self.boundary_n)
                                                            for j in
                                                            range(2 ** self.n)]
                y_coef: list[float] = []
                # Get coefficients of y-base-functions
                for j, phi in enumerate(y_base):
                    coef = float(np.sum(np.array(phi.values) * np.array(integrals)))
                    # print(j, walcoef)
                    y_coef.append(abs(coef))
                ymin: float = min(y_coef)
                ymax: float = max(y_coef)
                ymax2 = sorted(y_coef)[-2]
                ymin2 = sorted(y_coef)[2]

                if ymax - ymin < epsilon or ymax2 - ymin2 < epsilon:
                    index_y += 1
                else:
                    y_order = list(np.argsort(np.abs(y_coef))[::-1])
                    for y in range(len(y_order)):
                        y_order[y] = int(y_order[y])
                    index_y = -1

    def get_dynamic_order_sumdim(self) -> tuple[list[int], list[int]]:
        """
        Get a dynamic ordering of base functions based on the 1-D coefficients of the transformation.
        This assumes the base functions to be orthonormal.
        :return: A list with indices of base functions.
        """
        integrals: list[list[float]] = self.get_function_integral_matrix()
        epsilon = 1e-5

        order = range(2 ** self.n)
        index: int = 0

        while True:
            # Both directions either ran into troubles or have found a good ordering
            if index == -1:
                return order
            # If we have tried all rows, give up
            if index >= 2 ** self.n:
                index = -1

            if index != -1 and index < 2 ** self.n:
                x_coef: list[float] = []
                y_coef: list[float] = []
                x_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(i, index, self.n) for i in range(2 ** self.n)]
                y_base: list[DiscreteBaseFunction2D] = [WalshFunction2D(index, j, self.n) for j in range(2 ** self.n)]
                # Get coefficients of x-base-functions
                for i, phi in enumerate(x_base):
                    coef = float(np.sum(np.array(phi.values) * np.array(integrals)))
                    x_coef.append(abs(coef))
                for j, phi in enumerate(y_base):
                    coef = float(np.sum(np.array(phi.values) * np.array(integrals)))
                    # print(j, walcoef)
                    y_coef.append(abs(coef))

                coef_sum = np.array(x_coef) + np.array(y_coef)
                max_coef = coef_sum.max()
                min_coef = coef_sum.min()
                if max_coef - min_coef < epsilon:
                    index += 1
                else:
                    order = list(np.argsort(coef_sum)[::-1])
                    return order

    def get_function_integral_matrix(self) -> np.ndarray:
        integrals: list[list[float]] = []

        # When using boundary_n, a base function can have more values
        length: int = 2 ** max(self.n, self.boundary_n)
        if self.type == TestFunctionType.FUNCTION:

            # Calculate all integrals first
            for i in range(length):
                integrals_i: list[float] = []
                for j in range(length):
                    integrals_i.append((
                            self.function.evaluate_integral((j + 1) / length, (i + 1) / length)
                            - self.function.evaluate_integral(j / length, (i + 1) / length)
                            - self.function.evaluate_integral((j + 1) / length, i / length)
                            + self.function.evaluate_integral(j / length, i / length)))
                integrals.append(integrals_i)

        elif self.type == TestFunctionType.IMAGE:
            # Images are not affected by boundary_n.
            # For an image, each subintegral is equal to the function value, times the interval area
            # Since the integrals are always from c * 1/2^n to (c+1) * 1/2^n, the interval size (area) is (1/2^n)^2 = 1/4^n
            # This means that scaling all image values by 1/4^n is equal to their integral
            function_values = self.function.get_equidistant_data_points(length)
            integrals = function_values * (1 / (4 ** self.n))
        # else-case cannot happen, since initializing such a function will throw an error
        return integrals


# f = tf.QuadraticAdd()  # exp > dyn > paley   === viele 0
# f= tf.QuadraticMid() # exp > dyn > paley     === viele 0
# f = tf.QuadraticMax() # exp > dyn > paley    === viele 0
# f= tf.CubicPolynomial() # exp > dyn > paley  === viele 0
# f = tf.XCosine() # exp > paley > dyn
# f = tf.Sine() # exp > paley > dyn (exp 1.05 besser als paley, aber 6.4 besser als dyn)
# f= tf.TensorCosine() # dyn > paley > exp         === tensor
# f = tf.SineExponential() # exp > paley > dyn
# f = tf.ExponentialCosine()  # dyn > paley > exp    === tensor
# f= tf.ExponentialAdd() # exp > dyn > paley       === tensor
# f = tf.Rational() # exp > paley > dynamic
# f = tf.TensorSineCosine() # dyn > paley > exp    === tensor
# f = tf.Cosine()  # exp > paley > dyn
# f = tf.CosineXSquare()  # exp > dyn > paley


# f = tf.Polynomial()
# f = CosineProduct()
# f = tf.TaylorCosine()
# f = Image("../../images/prime_x.png", 256)
# f = Image("../../images/couple.pgm", 256)


def compare_pal_exp_dyn(n: int, boundary_n: int, f: TestFunction):
    print(f.name_cli)
    violet = ExperimentalWalshTransformation2D(n, f)
    walter = ExperimentalWalshTransformation2D(n, f, boundary_n=boundary_n)
    vivian = ExperimentalWalshTransformation2D(n, f, dynamic=True)
    # verena = ExperimentalWalshTransformation2D(n, f, dynamic=True, boundary_n=7)
    # walter.plot_base_matrix()
    walter_coef = walter.get_coefficients_integration_orthonormal()
    violet_coef = violet.get_coefficients_integration_orthonormal()
    vivian_coef = vivian.get_coefficients_integration_orthonormal()
    # verena_coef = verena.get_coefficients_integration_orthonormal()

    walter_t_vals = walter.sample_transform(walter_coef)
    violet_t_vals = violet.sample_transform(violet_coef)
    vivian_t_vals = vivian.sample_transform(vivian_coef)
    # verena_t_vals = verena.sample_transform(verena_coef)

    f_vals = f.sample()

    f.plot()
    violet.plot_transformation(violet_t_vals, f_vals, subtitle="paley")
    walter.plot_transformation(walter_t_vals, f_vals, subtitle="Exptrick")

    violet.plot_coefficients(violet_coef, subtitle="paley")
    walter.plot_coefficients(walter_coef, subtitle="exp trick")
    vivian.plot_coefficients(vivian_coef, subtitle="dynamic")
    # verena.plot_coefficients(verena_coef, subtitle="dynamic trick")

    # print(min(abs(violet_coef) + abs(walter_coef) + abs(vivian_coef)))
    # print(max(abs(violet_coef) + abs(walter_coef) + abs(vivian_coef)))

    violet_l2_square = violet.get_squared_l2_error(violet_coef)
    walter_l2_square = walter.get_squared_l2_error(walter_coef)
    vivian_l2_square = vivian.get_squared_l2_error(vivian_coef)
    # verena_e21 = verena.get_squared_l2_error(verena_coef)
    violet_l1 = violet.get_l1_error(violet_coef)
    walter_l1 = walter.get_l1_error(walter_coef)
    vivian_l1 = vivian.get_l1_error(vivian_coef)

    violet_linf = violet.get_linf_error(violet_coef)
    walter_linf = walter.get_linf_error(walter_coef)
    vivian_linf = vivian.get_linf_error(vivian_coef)

    violet_sparse_coef = violet.discard_coefficients_sparse_grid(violet_coef)
    walter_sparse_coef = walter.discard_coefficients_sparse_grid(walter_coef)
    vivian_sparse_coef = vivian.discard_coefficients_sparse_grid(vivian_coef)
    # verena_sparse_coef = verena.discard_coefficients_sparse_grid(verena_coef)

    violet_l2_square_sparse = violet.get_squared_l2_error(violet_sparse_coef)
    walter_l2_square_sparse = walter.get_squared_l2_error(walter_sparse_coef)
    vivian_l2_square_sparse = vivian.get_squared_l2_error(vivian_sparse_coef)
    # verena_e22 = verena.get_squared_l2_error(verena_sparse_coef)

    violet_l1_sparse = violet.get_l1_error(violet_sparse_coef)
    walter_l1_sparse = walter.get_l1_error(walter_sparse_coef)
    vivian_l1_sparse = vivian.get_l1_error(vivian_sparse_coef)

    violet_linf_sparse = violet.get_linf_error(violet_sparse_coef)
    walter_linf_sparse = walter.get_linf_error(walter_sparse_coef)
    vivian_linf_sparse = vivian.get_linf_error(vivian_sparse_coef)

    violet.plot_coefficients(violet_sparse_coef, subtitle="paley sparse")
    walter.plot_coefficients(walter_sparse_coef, subtitle="exp trick sparse")
    vivian.plot_coefficients(vivian_sparse_coef, subtitle="dynamic sparse")
    # verena.plot_coefficients(verena_sparse_coef, subtitle="dynamic trick")

    print("full grid:")
    print(f" paley:, {np.sqrt(violet_l2_square):.3e}, exp trick:, {np.sqrt(walter_l2_square):.3e}, "
          f"ratio: {(np.sqrt(violet_l2_square) / np.sqrt(walter_l2_square)):.3e}")
    print(f" paley:, {np.sqrt(violet_l2_square):.3e}, dynamic: {np.sqrt(vivian_l2_square):.3e}, "
          f"ratio: {(np.sqrt(violet_l2_square) / np.sqrt(vivian_l2_square)):.3e}")
    print(f" dynamic: {np.sqrt(vivian_l2_square):.3e}, exp trick:, {np.sqrt(walter_l2_square):.3e}, "
          f"ratio:,  {(np.sqrt(vivian_l2_square) / np.sqrt(walter_l2_square)):.3e}")
    min_err_str = f"exp trick: {np.sqrt(walter_l2_square):.3e}" if min(walter_l2_square, vivian_l2_square,
                                                                       violet_l2_square) == walter_l2_square else ""
    min_err_str = f"paley: {np.sqrt(violet_l2_square):.3e}" if min(walter_l2_square, vivian_l2_square,
                                                                   violet_l2_square) == violet_l2_square else min_err_str
    min_err_str = f"dynamic: {np.sqrt(vivian_l2_square):.3e}" if min(walter_l2_square, vivian_l2_square,
                                                                     violet_l2_square) == vivian_l2_square else min_err_str
    print(" MINIMUM", min_err_str)

    print("sparse grid:")
    print(f" paley: {np.sqrt(violet_l2_square_sparse):.3e}, exp trick: {np.sqrt(walter_l2_square_sparse):.3e}, "
          f"ratio:, {(np.sqrt(violet_l2_square_sparse) / np.sqrt(walter_l2_square_sparse)):.3e}")
    print(f" paley: {np.sqrt(violet_l2_square_sparse):.3e}, dynamic: {np.sqrt(vivian_l2_square_sparse):.3e}, "
          f"ratio:, {(np.sqrt(violet_l2_square_sparse) / np.sqrt(vivian_l2_square_sparse)):.3e}")
    print(f" dynamic: {np.sqrt(vivian_l2_square_sparse):.3e}, exp trick: {np.sqrt(walter_l2_square_sparse):.3e}, "
          f"ratio:, {(np.sqrt(vivian_l2_square_sparse) / np.sqrt(walter_l2_square_sparse)):.3e}")
    min_err_str = f"exp trick: {np.sqrt(walter_l2_square_sparse):.3e}" if min(walter_l2_square_sparse,
                                                                              vivian_l2_square_sparse,
                                                                              violet_l2_square_sparse) == walter_l2_square_sparse else ""
    min_err_str = f"paley: {np.sqrt(violet_l2_square_sparse):.3e}" if min(walter_l2_square_sparse,
                                                                          vivian_l2_square_sparse,
                                                                          violet_l2_square_sparse) == violet_l2_square_sparse else min_err_str
    min_err_str = f"dynamic: {np.sqrt(vivian_l2_square_sparse):.3e}" if min(walter_l2_square_sparse,
                                                                            vivian_l2_square_sparse,
                                                                            violet_l2_square_sparse) == vivian_l2_square_sparse else min_err_str
    print(" MINIMUM", min_err_str)

    print("")
    print(f"full grid:")
    print(f"Paley & {violet_l1.__round__(3)} & {np.sqrt(violet_l2_square).__round__(3)} & {violet_linf.__round__(3)}")
    print(f"Trick & {walter_l1.__round__(3)} & {np.sqrt(walter_l2_square).__round__(3)} & {walter_linf.__round__(3)}")
    print(f"Dynam & {vivian_l1.__round__(3)} & {np.sqrt(vivian_l2_square).__round__(3)} & {vivian_linf.__round__(3)}")
    print("")

    print(f"sparse grid:")
    print(f"            L1         L2         Linf")
    print(f"Paley & {violet_l1_sparse:.3e} & {np.sqrt(violet_l2_square_sparse):.3e} & {violet_linf_sparse:.3e}")
    print(f"Trick & {walter_l1_sparse:.3e} & {np.sqrt(walter_l2_square_sparse):.3e} & {walter_linf_sparse:.3e}")
    print(f"Dynam & {vivian_l1_sparse:.3e} & {np.sqrt(vivian_l2_square_sparse):.3e} & {vivian_linf_sparse:.3e}")
    print(f"Paley & {violet_l1_sparse} & {np.sqrt(violet_l2_square_sparse)} & {violet_linf_sparse}")
    print(f"Trick & {walter_l1_sparse} & {np.sqrt(walter_l2_square_sparse)} & {walter_linf_sparse}")
    print(f"Dynam & {vivian_l1_sparse} & {np.sqrt(vivian_l2_square_sparse)} & {vivian_linf_sparse}")
    print(
        f"Paley & {violet_l1_sparse.__round__(3)} & {np.sqrt(violet_l2_square_sparse).__round__(3)} & {violet_linf_sparse.__round__(3)}")
    print(
        f"Trick & {walter_l1_sparse.__round__(3)} & {np.sqrt(walter_l2_square_sparse).__round__(3)} & {walter_linf_sparse.__round__(3)}")
    print(
        f"Dynam & {vivian_l1_sparse.__round__(3)} & {np.sqrt(vivian_l2_square_sparse).__round__(3)} & {vivian_linf_sparse.__round__(3)}")


# n = 8
# walter = ExperimentalWalshTransformation2D(n, f)
# coef = walter.get_coefficients_integration_orthonormal()
# t_vals = walter.sample_transform(coef)
# walter.plot_transformation(t_vals, subtitle="og")
# walter.plot_coefficients(coef, subtitle="og")
# print(abs(coef).min(), abs(coef).max())
#
# coef_sparse = walter.discard_coefficients_sparse_grid(coef, 12)
# sparse_t_vals = walter.sample_transform(coef_sparse)
# walter.plot_transformation(sparse_t_vals, subtitle="sparse")
# walter.plot_coefficients(coef_sparse, subtitle="sparse")
#
# coef_opt = walter.discard_coefficients_percentage(coef, 81.25)
# opt_t_vals = walter.sample_transform(coef_opt)
# walter.plot_transformation(opt_t_vals, subtitle="opt")
# walter.plot_coefficients(coef_opt, subtitle="opt")