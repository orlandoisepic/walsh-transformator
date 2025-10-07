import numpy as np

from transformations.discrete_cosine_transformation_1d.discrete_cosine_transformation_1d import \
    DiscreteCosineTransformation
from transformations.discrete_cosine_transformation_2d.discrete_cosine_transformation_2d import \
    DiscreteCosineTransformation2D
from transformations.discrete_cosine_transformation_1d.discrete_cosine_1d import DiscreteCosineFunction
from transformations.discrete_cosine_transformation_2d.discrete_cosine_function_2d import DiscreteCosineFunction2D

from utility.test_functions_1d import Quadratic
from utility.test_functions_2d import QuadraticAdd


def test_cosine_coefficients_1d():
    """
    Test that the coefficients of the discrete Cosine transformation match for a small example.
    I rounded up to three decimals.
    """
    f = Quadratic()
    n: int = 3
    keira = DiscreteCosineTransformation(n, f)

    actual_coefficients: list[float] = keira.get_coefficients_integration_orthonormal()
    expected_coefficients: list[float] = [1 / 3, -0.285, 0.07, -0.03, 1 / 64, -0.009, 0.005, -0.002]

    assert len(actual_coefficients) == len(
        expected_coefficients), "Length of discrete Cosine coefficients does not match."

    for i in range(len(expected_coefficients)):
        c_e: float = expected_coefficients[i]
        c_a: float = actual_coefficients[i]
        assert np.isclose(c_e, c_a, atol=1e-3), (f"Coefficient {i} of discrete Cosine transformation does not match.\n"
                                                 f"Expected: {c_e}, actual: {c_a}.")


def test_cosine_coefficients_2d():
    """
    Test that the coefficients of the discrete Cosine transformation match for a small example.
    """
    f = QuadraticAdd()
    n: int = 1
    keira = DiscreteCosineTransformation2D(n, f)

    actual_coefficients: list[float] = keira.get_coefficients_integration_orthonormal()
    expected_coefficients: list[float] = [3 / 4, -1 / 8, -3 / 8, 1 / 16]

    assert len(actual_coefficients) == len(
        expected_coefficients), "Length of discrete Cosine coefficients does not match."

    for i in range(len(expected_coefficients)):
        c_e: float = expected_coefficients[i]
        c_a: float = actual_coefficients[i]
        assert np.isclose(c_e, c_a), (f"Coefficient {i} of discrete Cosine transformation does not match.\n"
                                      f"Expected: {c_e}, actual: {c_a}.")


def test_cosine_orthonormality_1d():
    """
    Test that discrete Cosine functions are orthonormal.
    """
    n: int = 3

    for i in range(2 ** n):
        for j in range(2 ** n):
            keira = DiscreteCosineFunction(i, n)
            carl = DiscreteCosineFunction(j, n)
            k_values = keira.values
            c_values = carl.values

            scalar_product: float = np.dot(k_values, c_values) / (2 ** n)
            if i == j:
                assert np.isclose(scalar_product, 1), "Discrete Cosine function norm is not 1."
            else:
                assert np.isclose(scalar_product, 0), "Discrete Cosine scalar product is not 0."


def test_cosine_orthonormality_2d():
    """
    Test that discrete Cosine functions are orthonormal.
    """
    n: int = 3

    for i in range(2 ** n):
        for j in range(2 ** n):
            for k in range(2 ** n):
                for l in range(2 ** n):
                    keira = DiscreteCosineFunction2D(i, j, n)
                    carl = DiscreteCosineFunction2D(k, l, n)
                    k_values = keira.values
                    c_values = carl.values
                    scalar_product: float = (np.array(k_values) * np.array(c_values)).sum() / (4 ** n)

                    if i == k and j == l:
                        assert np.isclose(scalar_product, 1), "Discrete Cosine norm is not 1."
                    else:
                        assert np.isclose(scalar_product, 0), "Discrete Cosine scalar product is not 0."


def test_wavelet_function_square_integral():
    """
    Test that the square integral attribute of discrete Cosine functions is correct.
    Because they are orthonormal, the integral of their square is equal to their squared norm, and thus should be 1.
    """
    n: int = 3

    for i in range(2 ** n):
        keira = DiscreteCosineFunction(i, n)
        assert np.isclose(keira.square_integral, 1), "Norm of discrete Cosine function is not 1."
