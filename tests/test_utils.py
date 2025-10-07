import numpy as np

from utility import utils as u
from utility import cli_helper as helper
from utility.color import Painter
from utility.timer import Timer
from utility.templates.test_functions import TestFunction, TestFunctionType, TestFunction1D, TestFunction2D, Image
from utility.templates.base_transformations import Transformation
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation
from transformations.walsh_transformation_2d.walsh_transformation_2d import WalshTransformation2D
from transformations.wavelet_transformation_1d.wavelet_transformation_1d import WaveletTransformation
from transformations.wavelet_transformation_2d.wavelet_transformation_2d import WaveletTransformation2D
from transformations.discrete_cosine_transformation_1d.discrete_cosine_transformation_1d import \
    DiscreteCosineTransformation
from transformations.discrete_cosine_transformation_2d.discrete_cosine_transformation_2d import \
    DiscreteCosineTransformation2D


def set_up(n: int, f: TestFunction) -> dict[str, Transformation]:
    if isinstance(f, TestFunction1D):
        violet = WalshTransformation(n, f)
        helena = WaveletTransformation(n, f)
        keira = DiscreteCosineTransformation(n, f)
    elif isinstance(f, TestFunction2D) or isinstance(f, Image):
        violet = WalshTransformation2D(n, f)
        helena = WaveletTransformation2D(n, f)
        keira = DiscreteCosineTransformation2D(n, f)
    else:
        raise TypeError("Invalid test function type.")

    return {"walsh": violet, "wavelet": helena, "cosine": keira}

def get_coefficients(transformations:dict[str,Transformation]) -> dict[str,np.ndarray]:
    coefficients: dict[str,np.ndarray] = {}
    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        coef = transformation.get_coefficients_integration_orthonormal()
        coefficients[transformation_str] = coef
    return coefficients