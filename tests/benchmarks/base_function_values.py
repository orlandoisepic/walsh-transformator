"""
Extract base function values for LaTeX plots.
"""
import numpy as np

import utility.utils
from transformations.discrete_cosine_transformation_1d.discrete_cosine_transformation_1d import \
    DiscreteCosineTransformation
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation
from transformations.wavelet_transformation_1d.wavelet_transformation_1d import WaveletTransformation
from utility.test_functions_1d import Exponential
from utility.templates.base_transformations import Transformation

transformation_objects: dict[str, Transformation] = {}


def set_up(n: int):
    f = Exponential()
    violet = WalshTransformation(n, f)
    helena = WaveletTransformation(n, f)
    keira = DiscreteCosineTransformation(n, f)

    transformation_objects["walsh"] = violet
    transformation_objects["wavelet"] = helena
    transformation_objects["cosine"] = keira


def get_function_values(transformation: str, n: int, first_n: int, yoffset: float = 3.25) -> str:
    """
    get function values for given transformation
    :param transformation: the transformation object whose values we want
    :param n: 2^n is no. of base functions
    :param yoffset: the space between functions.
    :return: latex string
    """
    if transformation == "wavelet":
        top_center = 0
        gap = 1
        half_ranges = [f.scale for f in transformation_objects[transformation].base_functions]
        centers = [top_center]
        for i in range(len(half_ranges) - 1):
            next_center = centers[i] - half_ranges[i] - half_ranges[i + 1] - gap
            centers.append(next_center)
        print(centers)

    output: str = ""
    h: float = 1 / (2 ** n)
    transformer = transformation_objects[transformation]
    for j, phi in enumerate(transformer.base_functions):
        yoffset_new = yoffset
        if j >= first_n:
            break
        output += f"\\addplot[blue,thick,const plot] coordinates {{"
        if transformation == "wavelet":
            for i, v in enumerate(phi.values):
                output += f"({h * i},{v + centers[j]})"
            output += f"({h * len(phi.values)},{phi.values[-1] + centers[j]})"

        else:
            for i, v in enumerate(phi.values):
                output += f"({h * i},{v - yoffset_new * j})"
            output += f"({h * len(phi.values)},{phi.values[-1] - yoffset_new * j})"

        output += f"}};"
    return output


def get_x_axes(transformation: str, n: int, first_n: int, yoffset: float = 3.25) -> str:
    """
    x axes for plots.
    :param transformation: the transformation object whose values we want
    :param n: 2^n is no of funcitons
    :param first_n: only the first n base functions get an x axis
    :param yoffset: the space between each function.
    3.25 seems to work well, provided that the values of the functions are not too large (<√2)
    :return: latex string
    """
    if transformation == "wavelet":
        top_center = 0
        gap = 1
        half_ranges = [f.scale for f in transformation_objects[transformation].base_functions]
        centers = [top_center]
        for i in range(len(half_ranges) - 1):
            next_center = centers[i] - half_ranges[i] - half_ranges[i + 1] - gap
            centers.append(next_center)
        print(centers)
    output: str = ""
    for i in range(2 ** n):
        yoffset_new = yoffset
        if i >= first_n:
            break
        output += f"\\addplot[gray,thin,dashed] coordinates {{"
        if transformation == "wavelet":
            output += (f"({0},{0 + centers[i]})"
                       f"({1},{0 + centers[i]})"
                       f"}};")
        else:
            output += (f"({0},{0 - yoffset_new * i})"
                       f"({1},{0 - yoffset_new * i})"
                       f"}};")
    return output


def get_interval_borders(n: int, ymax: float = 1, ymin: float = -25) -> str:
    """
    interval borders for discrete function
    :param n: 2^n is number of functions
    :param ymax: the upper (north) end of the axis
    :param ymin: the lower (south) end of the axis
    :return: latex string
    """
    output: str = ""
    h: float = 1 / (2 ** n)
    for i in range(1, 2 ** n):
        output += (f"\\addplot[lightgray,dashed] coordinates {{"
                   f"({i * h}, {ymin}) ({i * h}, {ymax})"
                   f"}};")
    return output


def input_to_file(string: str, file_name: str = "cool_file") -> None:
    """
    Write the given string to a .tex file. Automatically appends .tex to the given filename.
    :param string: The string to write to the file.
    :return: None
    """
    file_name = file_name + ".tex"
    f = open(file_name, 'w')  # open file in write mode (clears before writing)
    f.write(string)
    f.close()


def create_latex_plot(trafo: str, n: int, first_n: int = -1, yoffset: float = 3.25, ymin: float = -25, ymax: float = 1,
                      filename: str = "", x_axes: bool = False, interval_borders: bool = False) -> None:
    """
    if n>3: ymin/ymax have to be adjusted.
    :param trafo: transformation whose values we want
    :param n: 2^n is no of functions
    :param first_n: the first n base functions whose values we want
    :param yoffset: the yoffset between functions
    :param ymin: the smallest y value (i.e. this should be smaller than the smallest value a fucntion takes plus its y-offset)
    :param ymax: the largest y value, analogously to ymin
    :param filename: the filename to write to
    :param x_axes: whether to plot x axis or not
    :param interval_borders: whether to plot interval borders or not
    """
    set_up(n)
    if first_n == -1:
        first_n = 2 ** n

    complete_string: str = ""
    complete_string += get_x_axes(trafo, n, first_n, yoffset=yoffset) if x_axes else ""
    if trafo == "wavelet":
        ymin = -32.5 # this is just intermediary.
    complete_string += get_interval_borders(n, ymin=ymin, ymax=ymax) if interval_borders else ""
    complete_string += get_function_values(trafo, n, first_n, yoffset=yoffset)

    filename = filename if filename else trafo

    input_to_file(complete_string, file_name=filename)


create_latex_plot("walsh", 3, first_n=8, filename="wal3", x_axes=True, interval_borders=True)
