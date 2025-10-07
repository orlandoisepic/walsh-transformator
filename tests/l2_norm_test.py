import numpy as np

import utility.utils as u
import tests.test_utils as tu
from utility.templates.base_transformations import Transformation
import utility.test_functions_1d as tf1d
import utility.test_functions_2d as tf2d
from utility.templates.test_functions import Image


def test_decreasing_l2_norm_1d():
    """"
    Test that the L2 norm of the error decreases with increasing base functions.
    """
    ns: list[int] = [i for i in range(2, 9)]
    f = tf1d.Rational()
    l2_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for n in ns:
        transformations: dict[str, Transformation] = tu.set_up(n, f)
        coefs: dict[str, np.ndarray] = tu.get_coefficients(transformations)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefs[transformation_str]
            l2: float = transformation.get_squared_l2_error(coef)
            l2_history[transformation_str].append(l2)

    for transformation_str in transformations:
        l2_errors: list[float] = l2_history[transformation_str]
        assert u.is_sorted_descending(l2_errors), \
            f"{transformation_str.capitalize()}'s L2 error does not decrease with increasing base functions."


def test_decreasing_l2_norm_2d():
    """
    Test that the L2 norm of the error decreases with increasing base functions.
    """
    ns: list[int] = [i for i in range(2, 7)]
    f = tf2d.Rational()
    l2_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for n in ns:
        transformations: dict[str, Transformation] = tu.set_up(n, f)
        coefs: dict[str, np.ndarray] = tu.get_coefficients(transformations)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefs[transformation_str]
            l2: float = transformation.get_squared_l2_error(coef)
            l2_history[transformation_str].append(l2)

    for transformation_str in transformations:
        l2_errors: list[float] = l2_history[transformation_str]
        assert u.is_sorted_descending(l2_errors), \
            f"{transformation_str.capitalize()}'s L2 error does not decrease with increasing base functions."


def test_increasing_l2_norm_1d():
    """
    Test that the L2 norm of the error increases with decreasing coefficients used.
    """
    percentages: list[float] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    f = tf1d.CubicPolynomial()
    transformations: dict[str, Transformation] = tu.set_up(8, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    l2_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            l2: float = transformation.get_squared_l2_error(coef_discarded)
            l2_error_history[transformation_str].append(l2)

    for transformation_str in transformations:
        l2_errors: list[float] = l2_error_history[transformation_str]
        assert u.is_sorted_descending(l2_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L2 error does not increase with decreasing coefficients."


def test_increasing_l2_norm_2d():
    """
    Test that the L2 norm of the error increases with decreasing coefficients used.
    """
    percentages: list[float] = [p * 10 for p in range(0, 11)]
    f = tf2d.CubicPolynomial()
    transformations: dict[str, Transformation] = tu.set_up(6, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    l2_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            l2: float = transformation.get_squared_l2_error(coef_discarded)
            l2_error_history[transformation_str].append(l2)

    for transformation_str in transformations:
        l2_errors: list[float] = l2_error_history[transformation_str]
        assert u.is_sorted_descending(l2_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L2 error does not increase with decreasing coefficients."


def test_edge_cases_l2_norm_image():
    """
    Test that the L2 norm of the error is 0 when using all base functions
    and equal to the image's L2 norm when using 0 base functions.
    """
    f = Image("images/obunga64.jpeg", 64)
    transformations: dict[str, Transformation] = tu.set_up(6, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)

    image_l2: float = f.l2_norm_square()
    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        l2: float = transformation.get_squared_l2_error(coef)
        assert np.isclose(l2, 0, atol=1e-10, rtol=1e-8), \
            (f"{transformation_str.capitalize()}'s L2 error is not zero for all coefficients used."
             f"Expected: {0}. Actual: {l2}.")

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        new_coef = transformation.discard_coefficients_percentage(coef, 100)
        l2: float = transformation.get_squared_l2_error(new_coef)
        assert np.isclose(l2, image_l2, atol=1e-10, rtol=1e-8), \
            (f"{transformation_str.capitalize()}'s L2 error is not equal to the image's L2 error for "
             f"no coefficients used. Expected: {0}. Actual: {l2}.")


def test_increasing_l2_norm_image():
    """
    Test that the L2 norm of the error increases with decreasing coefficients used.
    """
    percentages: list[float] = [p * 10 for p in range(0, 11)]
    f = Image("images/obunga64.jpeg", 64)
    transformations: dict[str, Transformation] = tu.set_up(6, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    l2_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            l2: float = transformation.get_squared_l2_error(coef_discarded)
            l2_error_history[transformation_str].append(l2)

    for transformation_str in transformations:
        l2_errors: list[float] = l2_error_history[transformation_str]
        assert u.is_sorted_descending(l2_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L2 error does not increase with decreasing coefficients."
