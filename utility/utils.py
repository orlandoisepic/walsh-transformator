"""
This file includes helper functions for discrete transformations and other purposes.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def latex_font():
    """
    Set the font used in plots to a LaTeX similar font
    """
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14


def padded_binary(x: int, length: int) -> str:
    """
    Transforms the given integer to binary representation, padded to length n.
    :param x: The integer to transform.
    :param length: The length of the binary representation.
    """
    return bin(x)[2:].zfill(length)


def int_to_binary(n: int, length: int) -> list[int]:
    """
    Convert the given integer to binary, where each bit is stored separately and padded to the given length.
    :param n: The integer to convert.
    :param length: The length of the binary representation. If log₂n < length, then n will be padded to the given length.
    :return: A list of 0-1-integers.
    """
    bits = []
    while n:
        bits.append(n & 1)
        n >>= 1
    # Pad to length
    padding = [0] * (length - len(bits))
    bits += padding
    return bits[::-1]


def sample_image(path: str, size: int) -> np.ndarray:
    """
    Samples an image from the given path with the given size and converts it to greyscale.
    :param path: The path to the image.
    :param size: The size of the image per dimension.
    :return: The sampled image as a numpy array.
    """
    # Load image and convert to greyscale ("L")
    image = Image.open(path).convert("L")
    # Downsample to size × size
    image_resized = image.resize((size, size))
    # Convert to numpy-array to transpose and change datatype to 64-bit (original is 8-bit, which can lead to problems)
    return np.array(image_resized, dtype=np.int64)


def coinflip() -> bool:
    """
    Simulates a coin flip.
    :return: True with probability 0.5, False otherwise.
    """
    return random.choice([True, False])


def get_coinflips(n: int) -> list[bool]:
    """
    Return a list containing n random bools.
    :param n: The number of coin flips.
    :return: A list of bools.
    """
    return [coinflip() for _ in range(n)]


def is_float(string: str) -> bool:
    """
    Test if a string is a floating point number.
    :param string: The string to test.
    :return: True, if the string is a floating point number.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_power_of_two(number: int) -> bool:
    """
    Test if a number is a power of two.
    :param number: The number to test.
    :return: True, if the number is a power of two.
    """
    return number & (number - 1) == 0


def increase_to_next_power_of_two(number: int) -> int:
    """
    Increases a given number to the next power of two.
    If the number already is a power of two, i.e., ∃n: number=2ⁿ, then it is increased to 2ⁿ⁺¹
    :param number: The number to increase.
    :return: The increased number.
    """
    if number < 0:
        raise ValueError("Number must be positive.")
    elif number == 0:
        return 1
    elif is_power_of_two(number):
        return number * 2
    else:
        modified = np.ceil(np.log2(number))
        return int(2 ** modified)


def count_ones_position(number: int) -> int:
    """
    Count the number of 1s in the binary representation of the given number.
    """
    count = 0
    # This loop stops once number is equal to zero
    while number:
        number &= number - 1
        count += 1
    return count


def calc_nu_mu(number: int) -> tuple[int, int]:
    """
    Calculates nu (the number of 1-bits in the binary representation of the given number)
    and mu (the exponents+1 of the 1-bits) of the given number.
    The definition of nu and mu comes from the paper
        'Formulas for the Walsh coefficients of smooth functions and their application to bounds on the Walsh coefficients'
    :param number: The number to calculate nu and mu for.
    :return: A tuple with nu and mu.
    """
    number = int(number)
    ones_count = 0
    exponent_sum = 0
    while number:
        dropped_bit = number & -number
        pos = dropped_bit.bit_length() - 1
        exponent_sum += pos
        ones_count += 1
        number &= number - 1
    return ones_count, exponent_sum + ones_count


def dyadic_to_sequency(n: int) -> list[int]:
    """
    Create a sequency ordering of Walsh functions from a dyadic-ordering of them.
    This means that at index k of the resulting array will be the dyadic order of sequency-ordered Walsh function k.
    :param n: 2^n is the number of Walsh functions, i.e., the ordering is from 0 to 2^n-1.
    :return: A list of indices indicating the dyadic ordering of Walsh functions, based on a sequency-ordering.
    """
    dyadic_ordering = [0 for _ in range(2 ** n)]
    if n == 0:
        return dyadic_ordering
    for i in range(0, n):  # 0,...,n-1
        a: int = 2 ** i
        b: int = 2 * a - 1
        for j in range(0, a):  # 0,...,a-1
            dyadic_ordering[j + a] = b - dyadic_ordering[j]
    return dyadic_ordering


def sequency_to_dyadic(n: int) -> list[int]:
    """
    Change ordering of Walsh functions from sequency to dyadic ordering of Walsh functions.
    :param n: 2^n is the number of Walsh functions.
    :return: A list with the new ordering of Walsh functions.
    This means that sequency ordered Walsh function i's dyadic ordering is now result[i].
    TL;DR: Use this to obtain dyadic ordering of Walsh functions.
    """
    dyadic_ordering = [0 for _ in range(2 ** n)]
    if n == 0:
        return dyadic_ordering
    for i in range(1, n + 1):
        a: int = 2 ** (i - 1)
        b: float = a - 0.5
        b_prime = a
        # Use b' and j' to be able to use them in the loop
        # => number of iterations is preserved, for calculation the original value is used
        for j_prime in range(1, b_prime + 1):
            j = j_prime - 0.5
            # Casting to integer will not harm, as both b and j are of the form .5
            dyadic_ordering[int(b + j)] = dyadic_ordering[int(b - j)] + a
    return dyadic_ordering


def sequency_to_natural(n: int) -> list[int]:
    """
    Create a natural (Hadamard) ordering of Walsh functions from a sequency-ordering of them.
    This means that at index k of the resulting array will be the natural order of sequency-ordered Walsh function k.
    :param n: 2^n is the number of Walsh functions, i.e., the ordering is from 0 to 2^n-1.
    :return: A list of indices indicating the natural ordering of Walsh functions, based on a sequency-ordering.
    """
    natural_ordering = [0 for _ in range(2 ** n)]
    if n == 0:
        return natural_ordering
    for i in range(1, n + 1):
        a: int = 2 ** (n - i)
        b: float = 2 ** (i - 1) - 0.5
        b_prime = 2 ** (i - 1)
        for j_prime in range(1, b_prime + 1):
            j = j_prime - 0.5
            natural_ordering[int(b + j)] = natural_ordering[int(b - j)] + a
    return natural_ordering


def walsh_function_setup_helper(n: int) -> tuple[list[float], list[list[int]]]:
    """
    Calculates the intervals for all Walsh functions, as well as their binary representation.
    The intervals of each Walsh function are the same, regardless of their order.
    This way, the Walsh functions do not have to compute the intervals themselves.
    :param n: 2^n is the number of Walsh functions
    :return: A list of floats containing the interval boundaries, as well as a list of binary strings representing the interval number.
    """
    # Create 2^n intervals => 2^n +1 data points (including 0 and 1)
    h: float = 1 / (2 ** n)
    intervals: list[float] = []
    intervals_b: list[list[int]] = []
    for i in range(2 ** n + 1):
        intervals.append(i * h)
        intervals_b.append(int_to_binary(i, n))
    # remove the last element, as there are only 2^n intervals (i.e., from 0 to 2^n -1)
    intervals_b.pop(len(intervals_b) - 1)
    return intervals, intervals_b


def base_function_setup_helper(n: int) -> list[float]:
    """
    Calculates the intervals for all base functions.
    The intervals of each base function are the same, regardless of their order.
    This way, the base functions do not have to compute the intervals themselves.
    :param n: 2^n is the number of base functions
    :return: A list of floats containing the interval boundaries.
    """
    # Create 2^n intervals => 2^n +1 data points (including 0 and 1)
    h: float = 1 / (2 ** n)
    intervals: list[float] = []
    for i in range(2 ** n + 1):
        intervals.append(i * h)
    # remove the last element, as there are only 2^n intervals (i.e., from 0 to 2^n -1)
    return intervals


def calculate_boundaries_exp(n: int) -> list[float]:
    """
    Calculate boundaries for the coefficients of all Walsh functions for approximating f(x) = eˣ.
    :param n: 2^n is the number of Walsh functions up to which the boundaries are computed.
    :return: An array containing the boundaries of all Walsh functions,
    where entry i is the boundary of (sequency ordered) Walsh function number i.
    """
    m_2: int = 2  # constant
    C_2: int = 2  # constant
    b: int = 2  # base
    # Calculate nu and mu for dyadic ordering, to get the correct bound
    dyadic_ordering: list[int] = sequency_to_dyadic(n)
    # Constants, calculated for f=e^x, as this yields good bounds, close to the actual values
    D: float = np.e - 1
    r: float = 1

    boundaries: list[float] = []
    for i in range(2 ** n):
        dyadic_order: int = dyadic_ordering[i]
        nu, mu = calc_nu_mu(dyadic_order)
        bound: float = D * (b ** -mu) * ((r / m_2) ** nu) * (C_2 ** min(1, nu))
        boundaries.append(bound)
    return boundaries


def calculate_boundaries_cos(n: int) -> list[float]:
    """
    Calculate boundaries for the coefficients of all Walsh functions for approximating f(x) = cos(x).
    :param n: 2^n is the number of Walsh functions up to which the boundaries are computed.
    :return: An array containing the boundaries of all Walsh functions,
    where entry i is the boundary of (sequency ordered) Walsh function number i.
    """
    m_2: int = 2
    C_2: int = 2
    b: int = 2
    dyadic_ordering: list[int] = sequency_to_dyadic(n)

    # D: float = np.sin(1)
    # r: float = (1 - np.cos(1)) / np.sin(1)

    D: float = 2 / np.pi
    r: float = np.pi

    boundaries: list[float] = []
    for i in range(2 ** n):
        dyadic_order: int = dyadic_ordering[i]
        nu, mu = calc_nu_mu(dyadic_order)
        bound: float = D * (b ** -mu) * ((r / m_2) ** (nu - 1) * (C_2 ** min(1, nu)))
        boundaries.append(bound)
    return boundaries


def calculate_boundaries_cos2pi(n: int) -> list[float]:
    """
    Calculate boundaries for the coefficients of all Walsh functions for approximating f(x) = cos(x).
    :param n: 2^n is the number of Walsh functions up to which the boundaries are computed.
    :return: An array containing the boundaries of all Walsh functions,
    where entry i is the boundary of (sequency ordered) Walsh function number i.
    """
    m_2: int = 2
    C_2: int = 2
    b: int = 2
    dyadic_ordering: list[int] = sequency_to_dyadic(n)

    # D: float = np.sin(1)
    # r: float = (1 - np.cos(1)) / np.sin(1)

    D: float = 2 / np.pi
    r: float = 2 * np.pi

    boundaries: list[float] = []
    for i in range(2 ** n):
        dyadic_order: int = dyadic_ordering[i]
        nu, mu = calc_nu_mu(dyadic_order)
        bound: float = D * (b ** -mu) * ((r / m_2) ** (nu - 1) * (C_2 ** min(1, nu)))
        boundaries.append(bound)
    return boundaries


def get_order_from_bounds(boundaries: list[float]) -> np.ndarray:
    """
    Obtain the ordering of the Walsh functions through sorting their boundaries.
    :param boundaries: The boundaries of the Walsh functions, which use Kaczmarz ordering. This means that at entry i the boundary of (sequency ordered) Walsh function number i is stored.
    :return: The sorted indices of the Walsh functions. This means, at index i is the new order of (old sequency ordered) Walsh function number i.
    """
    indices = np.argsort(boundaries)[::-1]
    return indices


def get_level_shift(number: int):
    """
    For a given number representing the enumeration index of a Wavelet, return the level j of the Wavelet and its shift.
    It is 2^j - 1 ≤ number < 2^(j+1) - 1
    :param number: The index of the Wavelet, when enumerating them.
    :return: The level j and order k of the Wavelet.
    """
    if number < 0:
        raise ValueError("Wavelet index must be >= 0.")
    if number == 0:
        return 0, 0
    j = int(number).bit_length()
    k = int(number) - (1 << (j - 1))
    return j, k


def determine_maximum_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Determine the maximum distance between two vectors.
    If necessary, the vector v_2 will be up-/down-sampled to v_1's dimensions.
    This assumes both v_1 and v_2 to have dimensions 2^n x 2^n, for some n's.
    :param vec1: The first vector.
    :param vec2: The second vector.
    :return: The maximum distance between two elements of the vectors.
    """
    if vec1.shape != vec2.shape:
        # Ratio of the shape difference, e.g., ratio = 2 = 32 / 16
        ratio = vec1.shape[0] / vec2.shape[0]
        print("ratio", ratio)
        if ratio > 1:
            if vec2.ndim == 1:
                vec2 = np.repeat(vec2, int(ratio), axis=0)
            elif vec2.ndim == 2:
                vec2 = np.repeat(np.repeat(vec2, int(ratio), axis=0), int(ratio), axis=1)
            else:
                raise ValueError("Vector has too many dimensions.")
        else:
            # Take only every 1 / ratio element, e.g., ratio = 0.5 = 16 / 32 => only every second value
            if vec2.ndim == 1:
                vec2 = vec2[::int(1 / ratio)]
            elif vec2.ndim == 2:
                vec2 = vec2[::int(1 / ratio), ::int(1 / ratio)]
            else:
                raise ValueError("Vector has too many dimensions.")
    difference = abs(vec1 - vec2)
    return np.max(difference)


def affine_transform(vector: np.ndarray, lower: float = -1, upper: float = 1) -> np.ndarray:
    """
    Return the affine transform of an input vector to the interval given by lower and upper bounds, [-1,1] by default.
    :param vector: The vector to transform.
    :param lower: The lower bound of the interval.
    :param upper: The upper bound of the interval.
    :return: A vector containing the scaled original vector.
    """
    if lower >= upper:
        raise ValueError("Lower bound must be smaller than upper bound.")
    np_vector = np.asarray(vector, dtype=float)
    row_min = np_vector.min()
    row_max = np_vector.max()
    distance = row_max - row_min
    if distance > 0:
        scaled_vector = lower + (np_vector - row_min) * (upper - lower) / distance
    else:
        scaled_vector = np_vector * upper
    return scaled_vector


def affine_transform_per_vector(matrix: np.ndarray, lower: float = -1, upper: float = 1) -> np.ndarray:
    """
    Return an affine transformation of each vector of the matrix to the interval given by lower and upper bounds.
    :param matrix: The matrix that contains the vectors to transform.
    :param lower: The lower bound of the interval.
    :param upper: The upper bound of the interval.
    :return: A matrix containing the scaled vectors of the original matrix.
    """
    if lower >= upper:
        raise ValueError("Lower bound must be smaller than upper bound.")
    np_matrix = np.asarray(matrix, dtype=float)

    if np.ndim(np_matrix) >= 2:
        row_min = np_matrix.min(axis=1, keepdims=True)
        row_max = np_matrix.max(axis=1, keepdims=True)
    else:
        return affine_transform_per_vector(np_matrix, lower=lower, upper=upper)
    distance = row_max - row_min

    scaled_matrix = lower + (np_matrix - row_min) * (upper - lower) / distance

    return scaled_matrix


def determine_samples(n: int) -> int:
    """
    Determine the number of samples given some n, where the number of samples decreases for larger n,
    with a minimum of 10 samples and a maximum of 100 samples.
    This function follows a quadratic which is 100 at n=3 and 10 at n=6.
    :param n:
    :return:
    """
    if n <= 3:
        return 100
    elif n == 4:
        return 60
    elif n == 5:
        return 30
    else:  # >= 6
        return 10


def get_scale(n: int) -> int:
    """
    Determine a scale parameter based on a given number.
    The scale parameter will be a power of 2 and decreasing, for increasing n.
    The maximum is 2^{7*2} for n ≤ 3, and its minimum is 2^{2*2} for n ≥ 8.
    :param n: The number to determine the scale.
    :return: The scale.
    """
    if n <= 3:
        return 16384  # (2⁷)² such that log₄(scale)=7
    elif n == 4:
        return 4096  # (2⁶)² log₄(scale)=6
    elif n == 5:
        return 1024  # (2⁵)² log₄(scale)=5
    elif n == 6:
        return 256  # (2⁴)² log₄(scale)=4
    elif n == 7:
        return 64  # (2³)² log₄(scale)=3
    else:  # n >= 8
        return 16  # 16 log₄(scale)=2


def is_sorted_descending(array: list[int | float]) -> bool:
    """
    Check if a given array is sorted in descending order.
    :param array: The array to check.
    :return: True, if the array is sorted in descending order.
    """
    return all(array[i] >= array[i + 1] for i in range(len(array) - 1))


def plot_nu_mu(n: int) -> None:
    """
    Plot the mu and nu functions for 0 ≤ k ≤ 2ⁿ-1.
    :param n: The number up to which nu(k) and mu(k) are plotted.
    """
    nus: list[int] = []
    mus: list[int] = []
    for i in range(2 ** n):
        nu, mu = calc_nu_mu(i)
        nus.append(nu)
        mus.append(mu)

    plt.figure(figsize=(4.5, 4.5), constrained_layout=True)
    x = range(2 ** n)

    plt.plot(x, nus, "--", color="lime")
    plt.plot(x, mus, "--", color="green")
    plt.scatter(x, nus, label="$\\nu(k)$", color="lime")
    plt.scatter(x, mus, label="$\\mu(k)$", color="green")
    plt.xlabel("$k$")
    plt.legend()
    # plt.savefig("nu_mu.pdf", dpi=400, pad_inches=0.01)
    plt.show()


def create_levelsum_matrix(n: int) -> np.ndarray:
    """
    Create a matrix which represents the level-sum of its entries, where the level is ⌊log(i)⌋ and i the index
    :param n: The size of the matrix is 2ⁿ × 2ⁿ.
    :return: A matrix representing the level-sum of its entries, as defined above.
    """
    g = np.fromiter((k.bit_length() for k in range(2 ** n)), dtype=np.int64)

    return g[:, None] + g[None, :]

def boundary_segments_from_mask(mask):
    """
    Given a 2D boolean mask (shape Ny x Nx), return a list of line segments
    (each segment is [(x0,y0),(x1,y1)]) along the pixel edges where mask
    changes value. Coordinates are in imshow data coords: pixel centers at
    integers, pixel edges at half-integers (-0.5 .. N-0.5).
    This can be used to insert a "border" along the mask values in a heatmap plot.
    """
    Ny, Nx = mask.shape

    # Vertical boundaries: between columns j-1 and j for j in 0..Nx
    # diff_v_full[r, j] = True when mask[r, j-1] != mask[r, j]
    # define mask[..., -1] and mask[..., Nx] as False (outside)
    diff_v_full = np.zeros((Ny, Nx + 1), dtype=bool)
    # first boundary (left edge) compares outside(False) with mask[:,0]
    diff_v_full[:, 0] = mask[:, 0] != False
    # interior boundaries
    diff_v_full[:, 1:Nx] = mask[:, 1:Nx] != mask[:, :Nx - 1]
    # last boundary (right edge) compares mask[:,Nx-1] with outside(False)
    diff_v_full[:, Nx] = mask[:, Nx - 1] != False

    segs = []

    # For each vertical boundary at x = j - 0.5, find contiguous rows that are True
    for j in range(Nx + 1):
        col = diff_v_full[:, j]  # length Ny
        if not col.any():
            continue
        # find contiguous runs of True in col
        true_idx = np.flatnonzero(col)
        # group contiguous indices into runs
        starts = [true_idx[0]]
        ends = []
        for k in range(1, true_idx.size):
            if true_idx[k] != true_idx[k - 1] + 1:
                ends.append(true_idx[k - 1])
                starts.append(true_idx[k])
        ends.append(true_idx[-1])
        # convert runs to y spans (edge coords)
        x_coord = j - 0.5
        for s, e in zip(starts, ends):
            y0 = s - 0.5
            y1 = e + 0.5
            segs.append([(x_coord, y0), (x_coord, y1)])

    # Horizontal boundaries: between rows i-1 and i for i in 0..Ny
    diff_h_full = np.zeros((Ny + 1, Nx), dtype=bool)
    diff_h_full[0, :] = mask[0, :] != False
    diff_h_full[1:Ny, :] = mask[1:Ny, :] != mask[:Ny - 1, :]
    diff_h_full[Ny, :] = mask[Ny - 1, :] != False

    # For each horizontal boundary at y = i - 0.5, find contiguous cols that are True
    for i in range(Ny + 1):
        row = diff_h_full[i, :]
        if not row.any():
            continue
        true_idx = np.flatnonzero(row)
        starts = [true_idx[0]]
        ends = []
        for k in range(1, true_idx.size):
            if true_idx[k] != true_idx[k - 1] + 1:
                ends.append(true_idx[k - 1])
                starts.append(true_idx[k])
        ends.append(true_idx[-1])
        y_coord = i - 0.5
        for s, e in zip(starts, ends):
            x0 = s - 0.5
            x1 = e + 0.5
            segs.append([(x0, y_coord), (x1, y_coord)])

    return segs