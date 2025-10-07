import numpy as np
from concurrent.futures import ProcessPoolExecutor

from utility.timer import Timer
from utility.templates.test_functions import TestFunction, TestFunctionType, TestFunction1D, TestFunction2D, Image
import utility.test_functions_1d as tf1d, utility.test_functions_2d as tf2d
from utility.templates.base_transformations import Transformation
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation
from transformations.walsh_transformation_2d.walsh_transformation_2d import WalshTransformation2D
from transformations.wavelet_transformation_1d.wavelet_transformation_1d import WaveletTransformation
from transformations.wavelet_transformation_2d.wavelet_transformation_2d import WaveletTransformation2D
from transformations.discrete_cosine_transformation_1d.discrete_cosine_transformation_1d import \
    DiscreteCosineTransformation
from transformations.discrete_cosine_transformation_2d.discrete_cosine_transformation_2d import \
    DiscreteCosineTransformation2D
import utility.cli_helper as clihelper

"""
Use with: python -m tests.benchmarks.error_measuring
This measures L1, L2 and L∞ error for Walsh, Wavelet and Cosine transformation
"""

transformation_objects: dict[str, Transformation] = {}
coefficient_arrays: dict[str, np.ndarray] = {}

l1_error_history: dict[str, list[float]] = {}
l2_error_history: dict[str, list[float]] = {}
linf_error_history: dict[str, list[float]] = {}

plot_options: dict[str, str] = {
    "walsh": "thick, blue, mark options={blue}",  # mark={square}*"
    "wavelet": "thick, red, mark options={red}",
    "cosine": "thick, green, mark options={green}"
}

theresa = Timer()


def set_up(n: int, f: TestFunction, boundary_n: int = -1) -> None:
    if isinstance(f, TestFunction1D):
        transformations_to_initialize = [
            (WalshTransformation, n, f, boundary_n),
            (WaveletTransformation, n, f, boundary_n),
            (DiscreteCosineTransformation, n, f, boundary_n),
        ]

    elif isinstance(f, TestFunction2D) or isinstance(f, Image):
        transformations_to_initialize = [
            (WalshTransformation2D, n, f, boundary_n),
            (WaveletTransformation2D, n, f, boundary_n),
            (DiscreteCosineTransformation2D, n, f),
        ]

    with ProcessPoolExecutor(max_workers=3) as executor:
        transformation_object_list = list(
            executor.map(clihelper.initialize_transformation, *zip(*transformations_to_initialize))
        )
    transformation_objects["walsh"] = transformation_object_list[0]
    transformation_objects["wavelet"] = transformation_object_list[1]
    transformation_objects["cosine"] = transformation_object_list[2]

    for transformation_str in transformation_objects:
        l1_error_history[transformation_str] = []
        l2_error_history[transformation_str] = []
        linf_error_history[transformation_str] = []

    theresa.lap()


def calculate_coefficients() -> None:
    with ProcessPoolExecutor(max_workers=3) as executor:
        coefficient_array_list = list(executor.map(clihelper.calculate_coefficients, transformation_objects.values()))

    coefficient_arrays["walsh"] = coefficient_array_list[0]
    coefficient_arrays["wavelet"] = coefficient_array_list[1]
    coefficient_arrays["cosine"] = coefficient_array_list[2]


def discard_coefficients_percentage(p: float) -> None:
    coefficients_to_discard = [
        (transformation_objects["walsh"], coefficient_arrays["walsh"], p),
        (transformation_objects["wavelet"], coefficient_arrays["wavelet"], p),
        (transformation_objects["cosine"], coefficient_arrays["cosine"], p),
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        new_coefficient_array_list = list(
            executor.map(clihelper.discard_percentage_coefficients, *zip(*coefficients_to_discard)))

    coefficient_arrays["walsh"] = new_coefficient_array_list[0][0]
    coefficient_arrays["wavelet"] = new_coefficient_array_list[1][0]
    coefficient_arrays["cosine"] = new_coefficient_array_list[2][0]


def calculate_errors() -> None:
    errors_to_calculate = [
        (transformation_objects["walsh"], coefficient_arrays["walsh"]),
        (transformation_objects["wavelet"], coefficient_arrays["wavelet"]),
        (transformation_objects["cosine"], coefficient_arrays["cosine"]),
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        error_list = list(executor.map(clihelper.calculate_errors, *zip(*errors_to_calculate)))

    for i, transformation_str in enumerate(transformation_objects):
        l1_error_history[transformation_str].append(error_list[i][0])
        l2_error_history[transformation_str].append(np.sqrt(error_list[i][1]))
        linf_error_history[transformation_str].append(error_list[i][2])


def get_l1_results(latex: bool = True, text: bool = False, reverse: bool = False) -> str:
    output: str = ""
    for transformation_str in transformation_objects:
        if text:
            output += f">>> {transformation_str.capitalize()}-transformation L1 <<<"
        if latex:
            options = plot_options[transformation_str]
            output += f"\\addplot [{options}] coordinates {{"
        l1_history = l1_error_history[transformation_str]
        if not reverse:
            for i, entry in enumerate(l1_history):
                output += f"({i}, {entry})"
        else:
            for i, entry in enumerate(l1_history):
                output += f"({100 - i}, {entry})"
        if latex:
            output += "};"
    return output


def get_l2_results(latex: bool = True, text: bool = False, reverse: bool = False) -> str:
    output: str = ""
    for transformation_str in transformation_objects:
        if text:
            output += f">>> {transformation_str.capitalize()}-transformation L2 <<<"
        if latex:
            options = plot_options[transformation_str]
            output += f"\\addplot [{options}] coordinates {{"
        l2_history = l2_error_history[transformation_str]
        if not reverse:
            for i, entry in enumerate(l2_history):
                output += f"({i}, {entry})"
        else:
            for i, entry in enumerate(l2_history):
                output += f"({100 - i}, {entry})"
        if latex:
            output += "};"
    return output


def get_linf_results(latex: bool = True, text: bool = False, reverse: bool = False) -> str:
    output: str = ""
    for transformation_str in transformation_objects:
        if text:
            output += f">>> {transformation_str.capitalize()}-transformation L∞ <<<"
        if latex:
            options = plot_options[transformation_str]
            output += f"\\addplot [{options}] coordinates {{"
        linf_history = linf_error_history[transformation_str]
        if not reverse:
            for i, entry in enumerate(linf_history):
                output += f"({i}, {entry})"
        else:
            for i, entry in enumerate(linf_history):
                output += f"({100 - i}, {entry})"
        if latex:
            output += "};"
    return output


def begin_tikzicture(xlabel: str = "", ylabel: str = ""):
    print("\\begin{figure}[h]\n"
          "\\begin{center}\n"
          "\\begin{tikzpicture}\n"
          "\\begin{loglogaxis}\n"
          f"[xlabel = {xlabel},\n"
          "xmin=0,\n"
          "xmax=150,\n"
          "x dir=reverse,\n"
          f"ylabel = {ylabel},\n"
          "legend pos= north west]")


def end_tikzicture(caption: str = "") -> None:
    print("\\legend{Walsh, Wavelet, Kosinus}\n"
          "\\end{loglogaxis}\n"
          "\\end{tikzpicture}\n"
          "\\end{center}\n"
          f"\\caption{{{caption}}}\n"
          "\\end{figure}\n")


def input_to_file(string: str, file_name: str = "cool_file") -> None:
    """
    Write the given string to a .tex file. Automatically appends .tex to the given filename.
    :param string: The string to write to the file.
    :return: None
    """
    file_name = file_name + ".tex"
    f = open(file_name, 'w')
    f.write(string)
    f.close()
    print(f"\\input{{inputs/{file_name}}}\n")


def perform_test(n: int, f: TestFunction, text: bool = False, reverse: bool = False, boundary_n: int = -1) -> None:
    """
    Calculate the L1, L2 and L∞ error norm of the given function for transformations with 2ⁿ base functions.
    The transformations are initialized, their coefficients are discarded,
    and the error norms are calculated for the smallest x% of coefficients discarded, x ∈ {0,...,100}.
    Finally, the results are printed in LaTeX suitable style.
    :param n: The number of base functions.
    :param f: The function to be transformed.
    :return: None
    """
    theresa.start()
    if text:
        print("Starting test...")
    set_up(n, f, boundary_n=boundary_n)
    if text:
        print(f"Setup complete. Elapsed time: {theresa.get_last_interval_string()}")
    calculate_coefficients()
    if text:
        print(f"Coefficients calculated. Elapsed time: {theresa.get_last_interval_string()}")
    for i in range(100 + 1):
        discard_coefficients_percentage(i)
        calculate_errors()  # First iteration is discard 0%, so original error is captured as well
    theresa.lap()
    if text:
        print(f"Finished sampling. Elapsed time: {theresa.get_last_interval_string()}")

    xlabel: str = "Prozent an Koeffizienten verwendet"
    ylabel: str = "Norm des Fehlers"

    file_name: str = ""
    file_name += "1D-" if isinstance(f, TestFunction1D) else "2D-"
    file_name += f.name_cli + "-"
    file_name += f"n={n}-"
    file_name += f"bn={boundary_n}-" if boundary_n != -1 else ""

    caption: str = f" Norm des Fehlers für ${f.name}$ mit $n=2^{{{n}}}$ Basisfunktionen"
    caption2: str = f" (die besten aus $2^{{{boundary_n}}}$)" if boundary_n != -1 else ""

    begin_tikzicture(xlabel=xlabel, ylabel="$L^1$ " + ylabel)
    input_to_file(get_l1_results(reverse=reverse), file_name=file_name + "L1")
    end_tikzicture(caption="$L^1$" + caption + caption2)
    print("% =================================")
    begin_tikzicture(xlabel=xlabel, ylabel="$L^2$ " + ylabel)
    input_to_file(get_l2_results(reverse=reverse), file_name=file_name + "L2")
    end_tikzicture(caption="$L^2$" + caption + caption2)
    print("% =================================")
    begin_tikzicture(xlabel=xlabel, ylabel="$L^\\infty$ " + ylabel)
    input_to_file(get_linf_results(reverse=reverse), file_name=file_name + "Linf")
    end_tikzicture(caption="$L^\\infty$" + caption + caption2)

    theresa.stop()
    if text:
        print(f"Test completed successfully. Total elapsed time: {theresa.get_total_time_string()}")


def perform_walsh_test(n: int, f: TestFunction, reverse: bool = False,
                       boundary_n: int = -1) -> None:
    """
    same as perform_test but only walsh transformation; and no text
    """
    if isinstance(f, TestFunction1D):
        walter = WalshTransformation(n, f, boundary_n=boundary_n)
    elif isinstance(f, TestFunction2D) or isinstance(f, Image):
        walter = WalshTransformation2D(n, f, boundary_n=boundary_n)

    coef = walter.get_coefficients_integration_orthonormal()

    l1_history: list[float] = []
    l2_history: list[float] = []
    linf_history: list[float] = []

    for p in range(100 + 1):
        mod_coef = walter.discard_coefficients_percentage(coef, p)
        l1: float = walter.get_l1_error(mod_coef)
        l1_history.append(l1)
        l2: float = np.sqrt(walter.get_squared_l2_error(mod_coef))
        l2_history.append(l2)
        linf: float = walter.get_linf_error(mod_coef)
        linf_history.append(linf)

    options = plot_options["walsh"]
    options_str: str = f"\\addplot [{options}] coordinates {{"

    l1_str: str = "" + options_str
    if not reverse:
        for i, entry in enumerate(l1_history):
            l1_str += f"({i}, {entry})"
    else:
        for i, entry in enumerate(l1_history):
            l1_str += f"({100 - i}, {entry})"
    l1_str += "};"
    file_name: str = f"walsh_n={n}"
    file_name += f"_m={boundary_n}" if boundary_n != -1 else ""
    input_to_file(l1_str, file_name=file_name + "_L1")

    l2_str: str = "" + options_str
    if not reverse:
        for i, entry in enumerate(l2_history):
            l2_str += f"({i}, {entry})"
    else:
        for i, entry in enumerate(l2_history):
            l2_str += f"({100 - i}, {entry})"
    l2_str += "};"
    file_name: str = f"walsh_n={n}"
    file_name += f"_m={boundary_n}" if boundary_n != -1 else ""
    input_to_file(l2_str, file_name=file_name + "_L2")

    linf_str: str = "" + options_str
    if not reverse:
        for i, entry in enumerate(linf_history):
            linf_str += f"({i}, {entry})"
    else:
        for i, entry in enumerate(linf_history):
            linf_str += f"({100 - i}, {entry})"
    linf_str += "};"
    file_name: str = f"walsh_n={n}"
    file_name += f"_m={boundary_n}" if boundary_n != -1 else ""
    input_to_file(linf_str, file_name=file_name + "_Linf")


if __name__ == "__main__":
    function = tf1d.Exponential()
    # function = tf2d.QuadraticMax()
    # function = tf2d.ExponentialAdd()
    n = 8
    # function = Image("images/pyramid.jpg", 2**n)
    #function = tf1d.ExponentialSine()
    perform_test(n, function, reverse=True, boundary_n=n+8)
    #perform_walsh_test(n, function, reverse=True, boundary_n=n+4)
