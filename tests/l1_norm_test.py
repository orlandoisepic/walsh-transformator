import numpy as np

import utility.utils as u
import tests.test_utils as tu
from utility.templates.base_transformations import Transformation
import utility.test_functions_1d as tf1d
import utility.test_functions_2d as tf2d


def test_constant_l1_norm_1d():
    """
    Test implementation against constant.
    It should be close to zero, as all three transformations can approximate the constant exactly.
    """
    expected_error_norm: float = 0

    f = tf1d.Constant()
    transformations: dict[str, Transformation] = tu.set_up(3, f)
    coefficients = tu.get_coefficients(transformations)

    # Test calculated error norm
    # The error norm should be zero, as all three transforms can approximate the constant function correctly
    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        l1_error = transformation.get_l1_error(coef)
        assert np.isclose(l1_error, expected_error_norm,
                          atol=1e-14), f"Expected: {expected_error_norm}, got: {l1_error}"


def test_constant_l1_norm_2d():
    """
    Test implementation against constant.
    It should be close to zero, as all three transformations can approximate the constant exactly.
    """
    expected_error_norm: float = 0

    f = tf2d.Constant()
    transformations: dict[str, Transformation] = tu.set_up(3, f)
    coefficients = tu.get_coefficients(transformations)
    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        l1_error = transformation.get_l1_error(coef)
        assert np.isclose(l1_error, expected_error_norm,
                          atol=1e-14), f"Expected: {expected_error_norm}, got: {l1_error}"


def test_brute_force_close_l1_norm_1d():
    """
    Test implementation against brute force.
    """
    no_samples: int = 2 ** 24  # A lot of samples are needed here actually to make it close to my implementation
    f = tf1d.Sine()
    x = np.linspace(0, 1, no_samples)
    y = f.evaluate(x)

    transformations: dict[str, Transformation] = tu.set_up(3, f)
    coefficients = tu.get_coefficients(transformations)
    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        values = transformation.sample_transform(coef, samples=no_samples)
        f_minus_approx = np.abs(y - values)
        total_brute_force = np.sum(f_minus_approx) / no_samples
        actual = transformation.get_l1_error(coef)

        assert np.isclose(total_brute_force, actual, atol=1e-10, rtol=1e-7), (
            f"Expected: {total_brute_force}, got: {actual}.\n"
            f"Difference: {abs(total_brute_force - actual):.3e}\n"
            f"Ratio: {abs((total_brute_force - actual) / actual):.3e}\n")


def test_brute_force_convergence_l1_norm_1d():
    """
    Test if brute force converges against implementation.
    We know that brute force converges on the true solution for infinite samples.
    """
    no_samples: list[int] = [2 ** i for i in range(6, 16, 2)]
    differences: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    f = tf1d.Sine()
    transformations: dict[str, Transformation] = tu.set_up(3, f)
    coefficients = tu.get_coefficients(transformations)

    for n in no_samples:
        x = np.linspace(0, 1, n)
        y = f.evaluate(x)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefficients[transformation_str]
            values = transformation.sample_transform(coef, samples=n)
            function_minus_approx = np.abs(y - values)
            total_brute_force = np.sum(function_minus_approx) / n
            actual = transformation.get_l1_error(coef)

            differences[transformation_str].append(abs(actual - total_brute_force))

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        assert u.is_sorted_descending(differences[transformation_str]), \
            f"Brute force does not converge against {transformation.name.capitalize()}"


def test_brute_force_close_l1_norm_2d():
    """
    Test implementation against brute force.
    """
    no_samples: int = 2 ** 10
    f = tf2d.Rational()
    x = np.linspace(0, 1, no_samples)
    y = np.linspace(0, 1, no_samples)
    X, Y = np.meshgrid(x, y)
    Z = f.evaluate(X, Y)

    transformations: dict[str, Transformation] = tu.set_up(5, f)
    coefficients = tu.get_coefficients(transformations)

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        t_values = transformation.sample_transform(coef, samples=no_samples)
        difference = np.abs(Z - t_values)

        total_brute_force = np.sum(difference) / (no_samples ** 2)
        actual = transformation.get_l1_error(coef)

        assert np.isclose(total_brute_force, actual, atol=1e-5, rtol=1e-3), (
            f"Expected: {total_brute_force}, got: {actual}.\n"
            f"Difference: {abs(total_brute_force - actual):.3e}\n"
            f"Ratio: {abs((total_brute_force - actual) / actual):.3e}\n")


def test_brute_force_convergence_l1_norm_2d():
    """
    Test if brute force converges against implementation.
    We know that brute force converges on the true solution for infinite samples.
    """
    no_samples: list[int] = [2 ** i for i in range(4, 14, 2)]
    differences: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    f = tf2d.Sine()
    transformations: dict[str, Transformation] = tu.set_up(3, f)
    coefficients = tu.get_coefficients(transformations)

    for n in no_samples:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = f.evaluate(X, Y)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefficients[transformation_str]
            values = transformation.sample_transform(coef, samples=n)
            function_minus_approx = np.abs(Z - values)
            total_brute_force = np.sum(function_minus_approx) / (n ** 2)
            actual = transformation.get_l1_error(coef)

            differences[transformation_str].append(abs(actual - total_brute_force))

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        assert u.is_sorted_descending(differences[transformation_str]), \
            f"Brute force does not converge against {transformation.name.capitalize()}."


def test_decreasing_l1_norm_1d():
    """
    Test that the L1 norm of the error decreases with increasing base functions.
    """
    ns: list[int] = [i for i in range(2, 12, 2)]
    f = tf1d.Rational()
    l1_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for n in ns:
        transformations: dict[str, Transformation] = tu.set_up(n, f)
        coefs = tu.get_coefficients(transformations)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefs[transformation_str]
            l1: float = transformation.get_l1_error(coef)
            l1_history[transformation_str].append(l1)

    for transformation_str in transformations:
        l1_errors: list[float] = l1_history[transformation_str]
        assert u.is_sorted_descending(l1_errors), \
            f"{transformation_str.capitalize()}'s L1 error does not decrease with increasing base functions."


def test_decreasing_l1_norm_2d():
    """
    Test that the L1 norm of the error decreases with increasing base functions.
    """
    ns: list[int] = [i for i in range(2, 7)]
    f = tf2d.Rational()
    l1_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for n in ns:
        transformations: dict[str, Transformation] = tu.set_up(n, f)
        coefs: dict[str, np.ndarray] = tu.get_coefficients(transformations)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefs[transformation_str]
            l1: float = transformation.get_l1_error(coef)
            l1_history[transformation_str].append(l1)

    for transformation_str in transformations:
        l1_errors: list[float] = l1_history[transformation_str]
        assert u.is_sorted_descending(l1_errors), \
            f"{transformation_str.capitalize()}'s L1 error does not decrease with increasing base functions."


def test_increasing_l1_norm_1d():
    """
    Test that the L1 norm of the error increases with decreasing base functions used.
    """
    f = tf1d.CubicPolynomial()
    transformations: dict[str, Transformation] = tu.set_up(8, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    l1_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    percentages: list[float] = [p * 10 for p in range(0, 11)]

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            l1: float = transformation.get_l1_error(coef_discarded).__round__(14)  # to avoid/mitigate numerical errors
            l1_error_history[transformation_str].append(l1)

    for transformation_str in transformations:
        l1_errors: list[float] = l1_error_history[transformation_str]
        assert u.is_sorted_descending(l1_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L1 error does not increase with decreasing coefficients."


def test_increasing_l1_norm_2d():
    """
    Test that the L1 norm of the error increases with decreasing coefficients used.
    """
    f = tf2d.CubicPolynomial()
    transformations: dict[str, Transformation] = tu.set_up(6, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    l1_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    percentages: list[float] = [p * 10 for p in range(0, 11)]

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            # Round to avoid/mitigate numerical errors.
            # It can happen that many coefficients are close to zero due to noise, so they need to be "rounded away".
            l1: float = transformation.get_l1_error(coef_discarded).__round__(10)
            l1_error_history[transformation_str].append(l1)

    for transformation_str in transformations:
        l1_errors: list[float] = l1_error_history[transformation_str]
        assert u.is_sorted_descending(l1_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L1 error does not increase with decreasing coefficients."
