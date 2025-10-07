import numpy as np

from utility.templates.test_functions import Image


def test_sample_image():
    """
    Test that the sampling of images works as expected.
    """
    resolution: int = 16
    test_image = Image("images/greyscale.png", resolution)  # I created it myself :)
    actual_values: list[list[int]] = test_image.sample()

    expected_values: list[list[int]] = [
        [((i + 4) % 16 + 1) * ((j + 8) % 16 + 1) - 1 for i in range(resolution)]
        for j in range(resolution)
    ]

    for i in range(resolution):
        for j in range(resolution):
            v_e = expected_values[i][j]
            v_a = actual_values[i][j]
            assert v_e == v_a, (f"Image values {i, j} do not match.\n"
                                f"Expected: {v_e}, actual: {v_a}.")


def test_l2_norm_image():
    """
    Test that the calculation of the L2 norm of the image works as expected.
    """
    resolution: int = 16
    test_image = Image("images/greyscale.png", resolution)

    actual_value: float = test_image.l2_norm_square()
    expected_value: float = 8598.75

    assert np.isclose(actual_value, expected_value), (f"Image L2 norm does not match.\n"
                                                      f"Expected: {expected_value}, actual: {actual_value}")
