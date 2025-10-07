import numpy as np

from utility.test_functions_1d import Quadratic
from transformations.wavelet_transformation_1d.wavelet_transformation_1d import WaveletTransformation
from transformations.wavelet_transformation_1d.haar_wavelet_1d import HaarWavelet

from utility.test_functions_2d import QuadraticAdd
from transformations.wavelet_transformation_2d.wavelet_transformation_2d import WaveletTransformation2D
from transformations.wavelet_transformation_2d.haar_wavelet_2d import HaarWavelet2D


def test_haar_wavelet_factor_1d():
    """
    Test that Haar Wavelets have the correct factor, i.e., the factor that makes them orthonormal.
    """
    n = 3
    wavelets = [HaarWavelet(i, n) for i in range(2 ** n)]

    actual_factors: list[float] = [w.scale for w in wavelets]
    expected_factors: list[float] = [1, 1, np.sqrt(2), np.sqrt(2), 2, 2, 2, 2]

    for i in range(len(wavelets)):
        f_e: float = expected_factors[i]
        f_a: float = actual_factors[i]
        assert np.isclose(f_e, f_a, atol=1e-10, rtol=0), (f"Wavelet factors {i} do not match.\n"
                                                          f"Expected: {f_e}, actual: {f_a}")


def test_wavelet_coefficients_1d():
    """
    Test that the coefficients of the Wavelet transformation match for a small example.
    """
    f = Quadratic()
    n: int = 3
    helena = WaveletTransformation(n, f)

    actual_coefficients: list[float] = helena.get_coefficients_integration_orthonormal()
    expected_coefficients: list[float] = [1 / 3, -1 / 4, -np.sqrt(2) / 32, -3 * np.sqrt(2) / 32, -1 / 128, -3 / 128,
                                          -5 / 128, -7 / 128]

    assert len(actual_coefficients) == len(expected_coefficients), "Length of Wavelet coefficients does not match."

    for i in range(len(expected_coefficients)):
        c_e: float = expected_coefficients[i]
        c_a: float = actual_coefficients[i]
        assert np.isclose(c_e, c_a), (f"Coefficient {i} of Wavelet transformation does not match.\n"
                                      f"Expected: {c_e}, actual: {c_a}.")


def test_haar_wavelet_factor_2d():
    """
    Test that Haar Wavelets have the correct factor, i.e., the factor that makes them orthonormal.
    """
    n = 2
    wavelets = [[HaarWavelet2D(i, j, n) for i in range(2 ** n)]
                for j in range(2 ** n)]

    actual_factors: list[list[float]] = []
    for vec in wavelets:
        scale_vec: list[float] = []
        for w in vec:
            scale_vec.append(w.scale)
        actual_factors.append(scale_vec)

    expected_factors: list[list[float]] = [
        [1, 1, np.sqrt(2), np.sqrt(2)],
        [1, 1, np.sqrt(2), np.sqrt(2)],
        [np.sqrt(2), np.sqrt(2), 2, 2],
        [np.sqrt(2), np.sqrt(2), 2, 2]
    ]

    for i in range(2 ** n):
        for j in range(2 ** n):
            f_e: float = expected_factors[i][j]
            f_a: float = actual_factors[i][j]
            assert np.isclose(f_e, f_a, atol=1e-10, rtol=0), (f"Wavelet factors {i},{j} do not match.\n"
                                                              f"Expected: {f_e}, actual: {f_a}")


def test_wavelet_coefficients_2d():
    """
    Test that the coefficients of the Wavelet transformation match for a small example.
    """
    f = QuadraticAdd()
    n: int = 1
    helena = WaveletTransformation2D(n, f)

    actual_coefficients: list[float] = helena.get_coefficients_integration_orthonormal()
    expected_coefficients: list[float] = [3 / 4, -1 / 8, -3 / 8, 1 / 16]

    assert len(actual_coefficients) == len(expected_coefficients), "Length of Wavelet coefficients does not match."

    for i in range(len(expected_coefficients)):
        c_e: float = expected_coefficients[i]
        c_a: float = actual_coefficients[i]
        assert np.isclose(c_e, c_a), (f"Coefficient {i} of Wavelet transformation does not match.\n"
                                      f"Expected: {c_e}, actual: {c_a}.")


def test_wavelet_orthonormality_1d():
    """
    Test that Wavelet functions are orthonormal.
    """
    n: int = 3

    for i in range(2 ** n):
        for j in range(2 ** n):
            helena = HaarWavelet(i, n)
            herbert = HaarWavelet(j, n)
            hel_values = helena.values
            her_values = herbert.values

            scalar_product: float = np.dot(hel_values, her_values) / (2 ** n)
            if i == j:
                assert np.isclose(scalar_product, 1), "Wavelet function norm is not 1."
            else:
                assert np.isclose(scalar_product, 0), "Wavelet function scalar product is not 0."


def test_wavelet_orthonormality_2d():
    """
    Test that Wavelet functions are orthonormal.
    """
    n: int = 3

    for i in range(2 ** n):
        for j in range(2 ** n):
            for k in range(2 ** n):
                for l in range(2 ** n):
                    helena = HaarWavelet2D(i, j, n)
                    herbert = HaarWavelet2D(k, l, n)
                    hel_values = helena.values
                    her_values = herbert.values
                    scalar_product: float = (np.array(hel_values) * np.array(her_values)).sum() / (4 ** n)

                    if i == k and j == l:
                        assert np.isclose(scalar_product, 1), "Wavelet function norm is not 1."
                    else:
                        assert np.isclose(scalar_product, 0), "Wavelet function scalar product is not 0."


def test_wavelet_function_square_integral():
    """
    Test that the square integral attribute of Wavelet functions is correct.
    Because they are orthonormal, the integral of their square is equal to their squared norm, and thus should be 1.
    """
    n: int = 3

    for i in range(2 ** n):
        helena = HaarWavelet(i, n)
        assert np.isclose(helena.square_integral, 1), "Norm of Wavelet function is not 1."
