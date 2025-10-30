import numpy as np

from transformations.walsh_transformation_1d.walsh_function_1d import WalshFunction
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation
from transformations.walsh_transformation_2d.walsh_function_2d import WalshFunction2D
from transformations.walsh_transformation_2d.walsh_transformation_2d import WalshTransformation2D
from utility.test_functions_1d import Quadratic
from utility.test_functions_2d import QuadraticAdd


def test_walsh_values_1d():
    """
    Test that all values of Walsh functions are +1 or -1.
    :return:
    """
    f = Quadratic()
    n: int = 3
    violet = WalshTransformation(n, f)
    values: list[list[int]] = []
    for phi in violet.base_functions:
        values.append(phi.values)

    assert all((v in [1, -1] for v in vec) for vec in values), "Walsh transformation has other values than +1 and -1."


def test_walsh_coefficients_1d():
    """
    Test that the coefficients of the Wavelet transformation match for a small example.
    """
    f = Quadratic()
    n: int = 3
    violet = WalshTransformation(n, f)

    actual_coefficients: list[float] = violet.get_coefficients_integration_orthonormal()
    expected_coefficients: list[float] = [1 / 3, -1 / 4, -1 / 8, 1 / 16, -1 / 16, 1 / 32, 1 / 64, 0]
    # [1 / 3, -1 / 4, 3 / 48, -3 / 24, 1 / 64, 0, 1 / 32, -1 / 16]

    assert len(actual_coefficients) == len(expected_coefficients), "Length of Walsh coefficients does not match."

    for i in range(len(expected_coefficients)):
        c_e: float = expected_coefficients[i]
        c_a: float = actual_coefficients[i]
        assert np.isclose(c_e, c_a), (f"Coefficient {i} of Walsh transformation does not match.\n"
                                      f"Expected: {c_e}, actual: {c_a}.")


def test_walsh_values_2d():
    """
    Test that all values of Walsh functions are +1 or -1.
    """
    f = QuadraticAdd()
    n: int = 3
    violet = WalshTransformation2D(n, f)

    values: list[list[list[int]]] = violet.base_values

    assert all(((v in [1, -1] for v in row) for row in vec) for vec in
               values), "Walsh transformation has other values than +1 and -1."


def test_walsh_coefficients_2d():
    f = QuadraticAdd()
    n: int = 1
    violet = WalshTransformation2D(n, f)

    actual_coefficients: list[float] = violet.get_coefficients_integration_orthonormal()
    expected_coefficients: list[float] = [3 / 4, -1 / 8, -3 / 8, 1 / 16]

    assert len(actual_coefficients) == len(expected_coefficients), "Length of Walsh coefficients does not match."

    for i in range(len(expected_coefficients)):
        c_e: float = expected_coefficients[i]
        c_a: float = actual_coefficients[i]
        assert np.isclose(c_e, c_a), (f"Coefficient {i} of Walsh transformation does not match.\n"
                                      f"Expected: {c_e}, actual: {c_a}.")


def test_walsh_orthonormality_1d():
    """
    Test that Walsh functions are orthonormal.
    """
    n: int = 3

    for i in range(2 ** n):
        for j in range(2 ** n):
            violet = WalshFunction(i, n)
            walter = WalshFunction(j, n)
            v_values = violet.values
            w_values = walter.values

            scalar_product: float = np.dot(v_values, w_values) / (2 ** n)
            if i == j:
                assert np.isclose(scalar_product, 1), "Walsh function norm is not 1."
            else:
                assert np.isclose(scalar_product, 0), "Walsh function scalar product is not 0."


def test_walsh_orthonormality_2d():
    """
    Test that Walsh functions are orthonormal.
    """
    n: int = 3

    for i in range(2 ** n):
        for j in range(2 ** n):
            for k in range(2 ** n):
                for l in range(2 ** n):
                    violet = WalshFunction2D(i, j, n)
                    walter = WalshFunction2D(k, l, n)
                    v_values = violet.values
                    w_values = walter.values
                    scalar_product: float = (np.array(v_values) * np.array(w_values)).sum() / (4 ** n)

                    if i == k and j == l:
                        assert np.isclose(scalar_product, 1), "Walsh function norm is not 1."
                    else:
                        assert np.isclose(scalar_product, 0), "Walsh function scalar product is not 0."


def test_walsh_function_square_integral():
    """
    Test that the square integral attribute of Walsh functions is correct.
    Because they are orthonormal, the integral of their square is equal to their squared norm, and thus should be 1.
    """
    n: int = 3

    for i in range(2 ** n):
        violet = WalshFunction(i, n)
        assert np.isclose(violet.square_integral, 1), "Norm of Walsh function is not 1."
