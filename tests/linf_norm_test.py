import numpy as np

import utility.utils as u
import tests.test_utils as tu
from utility.templates.base_transformations import Transformation
import utility.test_functions_1d as tf1d
import utility.test_functions_2d as tf2d
from utility.templates.test_functions import Image


def test_increasing_linf_norm_1d():
    """
    Test that the L∞ norm of the error increases with decreasing coefficients used.
    """
    percentages: list[float] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    f = tf1d.Cosine()
    transformations: dict[str, Transformation] = tu.set_up(8, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    linf_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            linf: float = transformation.get_linf_error(coef_discarded).__round__(10)
            linf_error_history[transformation_str].append(linf)

    for transformation_str in transformations:
        linf_errors: list[float] = linf_error_history[transformation_str]
        assert u.is_sorted_descending(linf_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L∞ error does not increase with decreasing coefficients."


def test_increasing_linf_norm_2d():
    """
    Test that the L∞ norm of the error increases with decreasing coefficients used.
    :return:
    """
    percentages: list[float] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    f = tf2d.Cosine()
    transformations: dict[str, Transformation] = tu.set_up(6, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    linf_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            linf: float = transformation.get_linf_error(coef_discarded).__round__(10)
            linf_error_history[transformation_str].append(linf)

    for transformation_str in transformations:
        linf_errors: list[float] = linf_error_history[transformation_str]
        assert u.is_sorted_descending(linf_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L∞ error does not increase with decreasing coefficients."


def test_increasing_linf_norm_image():
    """
    Test that the L∞ norm of the error increases with decreasing coefficients used.
    """
    percentages: list[float] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    f = Image("images/pyramid.jpg", 32)
    transformations: dict[str, Transformation] = tu.set_up(5, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)
    linf_error_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        for p in percentages:
            coef_discarded = transformation.discard_coefficients_percentage(coef, p)
            linf: float = transformation.get_linf_error(coef_discarded).__round__(10)
            linf_error_history[transformation_str].append(linf)

    for transformation_str in transformations:
        linf_errors: list[float] = linf_error_history[transformation_str]
        assert u.is_sorted_descending(linf_errors[::-1]), \
            f"{transformation_str.capitalize()}'s L∞ error does not increase with decreasing coefficients."


def test_edge_cases_linf_norm_image():
    """
    Test that the L∞ norm of the error is 0 for all coefficients used and
    equal to the images maximum pixel value for no coefficients used.
    """
    f = Image("images/obunga64.jpeg", 64)
    transformations: dict[str, Transformation] = tu.set_up(6, f)
    coefficients: dict[str, np.ndarray] = tu.get_coefficients(transformations)

    image_linf: float = np.max(f.sample())

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        linf: float = transformation.get_linf_error(coef)
        assert np.isclose(linf, 0, atol=1e-10, rtol=1e-8), \
            (f"{transformation_str.capitalize()}'s L∞ error is not zero for all coefficients used."
             f"Expected: {0}. Actual: {linf}.")

    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = coefficients[transformation_str]
        new_coef = transformation.discard_coefficients_percentage(coef, 100)
        linf: float = transformation.get_linf_error(new_coef)
        assert np.isclose(linf, image_linf, atol=1e-10, rtol=1e-8), \
            (f"{transformation_str.capitalize()}'s L∞ error is not equal to the image's L∞ error for "
             f"no coefficients used. Expected: {0}. Actual: {linf}.")


def test_decreasing_linf_norm_1d():
    """"
    Test that the L∞ norm of the error decreases with increasing base functions.
    """
    ns: list[int] = [i for i in range(2, 9)]
    f = tf1d.Exponential()
    linf_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for n in ns:
        transformations: dict[str, Transformation] = tu.set_up(n, f)
        coefs: dict[str, np.ndarray] = tu.get_coefficients(transformations)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefs[transformation_str]
            linf: float = transformation.get_linf_error(coef)
            linf_history[transformation_str].append(linf)

    for transformation_str in transformations:
        linf_errors: list[float] = linf_history[transformation_str]
        assert u.is_sorted_descending(linf_errors), \
            f"{transformation_str.capitalize()}'s L∞ error does not decrease with increasing base functions."


def test_decreasing_linf_norm_2d():
    """"
    Test that the L∞ norm of the error decreases with increasing base functions.
    """
    ns: list[int] = [i for i in range(2, 7)]
    f = tf2d.ExponentialAdd()
    linf_history: dict[str, list[float]] = {"walsh": [], "wavelet": [], "cosine": []}

    for n in ns:
        transformations: dict[str, Transformation] = tu.set_up(n, f)
        coefs: dict[str, np.ndarray] = tu.get_coefficients(transformations)
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            coef = coefs[transformation_str]
            linf: float = transformation.get_linf_error(coef)
            linf_history[transformation_str].append(linf)

    for transformation_str in transformations:
        linf_errors: list[float] = linf_history[transformation_str]
        assert u.is_sorted_descending(linf_errors), \
            f"{transformation_str.capitalize()}'s L∞ error does not decrease with increasing base functions."
