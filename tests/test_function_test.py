import numpy as np

import utility.test_functions_1d as tf1d
import utility.test_functions_2d as tf2d


def test_test_function_sample_1d():
    """
    Test that the sampling method works as expected.
    """
    f = tf1d.Exponential()
    actual_values: list[float] = f.sample(4)
    expected_values: list[float] = [np.exp(0), np.exp(1 / 3), np.exp(2 / 3), np.exp(1)]

    assert len(actual_values) == len(expected_values), "Length of sampled values do not match."
    for actual_value, expected_value in zip(actual_values, expected_values):
        assert np.isclose(actual_value, expected_value), (f"Test function sampling values do not match.\n"
                                                          f"Expected: {expected_value}, actual: {actual_value}")


def test_test_function_sample_2d():
    """
    Test that the sampling method works as expected.
    """
    f = tf2d.QuadraticAdd()  # xy + x

    actual_values: list[list[float]] = f.sample(4)  # 4 samples per dimension (equidistant) => 0, 1/3, 2/3, 1
    expected_values: list[list[float]] = [
        [0, 1 / 3, 2 / 3, 1],  # y = 0
        [0, 4 / 9, 8 / 9, 4 / 3],  # y = 1/3
        [0, 5 / 9, 10 / 9, 5 / 3],  # y = 2/3
        [0, 2 / 3, 4 / 3, 2]  # y = 1
    ]

    assert len(actual_values) == len(expected_values), "Length of sampled values do not match."

    for i in range(len(expected_values)):
        for j in range(len(expected_values)):
            v_e = expected_values[i][j]
            v_a = actual_values[i][j]
            assert np.isclose(v_e, v_a), (f"Test function sampling values do not match.\n"
                                          f"Expected: {v_e}, actual: {v_a}")


def test_evaluate_integral_1d():
    """
    Test that the integral evaluation works as expected.
    """
    # TODO try it out for all test functions: calculate F(x) for some x and compare


def test_evaluate_integral_2d():
    """
    Test that the integral evaluation works as expected.
    """
    # TODO try it out for all test functions


def test_evaluate_square_integral_1d():
    """
    Test that the evaluation of the integral of the squared function works as expected.
    """
    # TODO


def test_evaluate_square_integral_2d():
    """
    Test that the evaluation of the integral of the squared function works as expected.
    """
    # TODO


def test_l2_norm_1d():
    """
    Test that the calculation of the L2 norm of the function works as expected.
    """
    # TODO: if we know that the square integral is correct, then we just need to test this for one function


def test_l2_norm_2d():
    """
    Test that the calculation of the L2 norm of the function works as expected.
    """
    # TODO
