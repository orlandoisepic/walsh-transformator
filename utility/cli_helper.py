import random

import numpy as np

import utility.utils as u
import utility.templates.test_functions as tf
import utility.test_functions_1d as tf1d
import utility.test_functions_2d as tf2d
from utility.color import Painter
from utility.templates.base_transformations import Transformation
from utility.templates.test_functions import TestFunction

"""
A file to store all the helper methods and constructs for the cli.py file.
"""
pearl: Painter = Painter()

# Strings to use at the start and end of lines to make them stand out more.
sol_string: str = ">>> "
eol_string: str = " <<<"
sol_string2: str = " <<< "
eol_string2: str = " >>> "
sol_string3: str = "<<< "
eol_string3: str = " >>>"


def gather_information(transformation: Transformation, transform_coefficients: np.ndarray,
                       modified_transform_coefficients: np.ndarray, epsilon: float, errors: bool,
                       history_information: dict[str, str]) -> str:
    """
    Gather information about the transformation, and its coefficients and the errors it thus produces.
    :param transformation: The transformation object about which to gather information.
    :param transform_coefficients: The (original) coefficients of the transformation.
    :param modified_transform_coefficients: The modified coefficients of the transformation.
    :param epsilon: The epsilon used in discarding coefficients.
    :param errors: Whether to calculate and display errors or not.
    :param history_information: Information about the history of the current run, e.g., which method was used to discard coefficients.
    :return: A string containing the information.
    """
    output: str = ""
    total: int = transform_coefficients.shape[0]
    original_zeros: int = (transform_coefficients == 0).sum()
    modified_zeros: int = (modified_transform_coefficients == 0).sum()
    discarded: int = modified_zeros
    ratio: float = discarded / total
    last_discarding_method_used: str = history_information["last_discarding_method_used"]
    if last_discarding_method_used in ["s", "sparse"]:
        eps_str: str = f"Coefficients with sum of levels greater than {int(epsilon)} were discarded."
    else:
        eps_str: str = f"Coefficients smaller than epsilon = {epsilon:.4e} were discarded."

    output += (f"{pearl.blue(sol_string3)}{transformation.name.capitalize()}-transformation{pearl.blue(eol_string3)}\n"
               f"{eol_string2}{eps_str}\n"
               f"{eol_string2}Original zero-coefficients: {original_zeros}\n"
               f"{eol_string2}Modified zero-coefficients: {modified_zeros}\n"
               f"{eol_string2}Ratio: {(ratio * 100).__round__(2)} percent\n")

    if errors:
        l1: float = transformation.get_l1_error(modified_transform_coefficients)
        l2_square: float = transformation.get_squared_l2_error(modified_transform_coefficients)
        l2: float = np.sqrt(l2_square) if l2_square > 0 else 0
        linf: float = transformation.get_linf_error(modified_transform_coefficients)

        output += (f"{eol_string2}L1 norm of the approximation error: {l1:.3e}\n"
                   f"{eol_string2}L2 norm of the approximation error: {l2:.3e}\n"
                   f"{eol_string2}Squared L2 norm of the approximation error: {abs(l2_square):.3e}\n"
                   f"{eol_string2}L∞ norm of the approximation error: {linf:.3e}\n")
        if l2_square < 0:
            if l2_square.__round__(8) >= 0:
                output += (f"{pearl.yellow("WARNING")}: L2 error norm of {transformation.name.capitalize()} "
                           f"is negative: {l2_square:.5e}.\n{whitespaces(9)}This seems like a numerical instability.\n")
            else:
                raise ValueError(f"{pearl.red("ERROR")}: L2 error norm of {transformation.name.capitalize()} "
                                 f"is negative: {l2_square:.5e}.\nThis does not seem like numerical instability.")

    return output


def discard_coefficients_absolute(transformation: Transformation, transform_coefficients: np.ndarray,
                                  epsilon: float) -> np.ndarray:
    """
    Discards coefficients of the given transform by absolute value.
    :param transformation: The transformation whose coefficients to discard.
    :param transform_coefficients: The coefficients of the transformation.
    :param epsilon: The epsilon of discarding the coefficients.
    :return: The modified coefficients.
    """
    return transformation.discard_coefficients_absolute(transform_coefficients, epsilon)


def discard_coefficients_relative(transformation: Transformation, transform_coefficients: np.ndarray,
                                  epsilon: float) -> np.ndarray:
    """
    Discards coefficients of the given transform by relative value.
    :param transformation: The transformation whose coefficients to discard.
    :param transform_coefficients: The coefficients of the transformation.
    :param epsilon: The epsilon of discarding the coefficients.
    :return: The modified coefficients.
    """
    return transformation.discard_coefficients_relative(transform_coefficients, epsilon)


def discard_coefficients_percentage(transformation: Transformation, transform_coefficients: np.ndarray,
                                    epsilon: float) -> tuple[np.ndarray, float]:
    """
    Discards coefficients of the given transform by relative value.
    :param transformation: The transformation whose coefficients to discard.
    :param transform_coefficients: The coefficients of the transformation.
    :param epsilon: The epsilon of discarding the coefficients.
    :return: The modified coefficients.
    """
    return transformation.discard_coefficients_percentage(transform_coefficients, epsilon, cli=True)


def discard_coefficients_sparse(transformation: Transformation, transform_coefficients: np.ndarray,
                                epsilon: float) -> np.ndarray:
    """
    Discards coefficients of the given transform by the sum of their levels.
    This corresponds to the sparse grid selection of the coefficients.
    :param transformation: The transformation whose coefficients to discard.
    :param transform_coefficients: The coefficients of the transformation.
    :param epsilon: The epsilon of discarding the coefficients.
    :return: The modified coefficients.
    """
    return transformation.discard_coefficients_sparse_grid(transform_coefficients, epsilon)


def sample_transformation(transformation: Transformation, transform_coefficients: np.ndarray,
                          samples: int) -> np.ndarray:
    """
    Samples the given transformation object with the given coefficients.
    The transformation object defines a set of base functions, to which all coefficients belong.
    :param transformation: The transformation object to sample.
    :param transform_coefficients: The coefficients of the transformation object.
    :param samples: The number of samples to draw from the transformation object.
    :return: The values of the sampled transformation.
    """
    return transformation.sample_transform(transform_coefficients, samples=samples)


def calculate_coefficients(transformation: Transformation) -> np.ndarray:
    """
    Calculates the transformation coefficients for the given transformation.
    :param transformation: The transformation to calculate the coefficients for.
    :return: The coefficients for the given transformation.
    """
    return transformation.get_coefficients_integration_orthonormal()


def initialize_transformation(class_name: type[Transformation], n: int, f: TestFunction,
                              boundary_n: int, dynamic: bool = False) -> Transformation:
    """
    Initializes an object of class "class_name" with the parameters "n" and "f";
    i.e., a transformation object with n base functions per dimension for function f.
    :param class_name: The name of the class of the transformation to initialize.
    :param n: The number of base functions per dimension.
    :param f: The function to be transformed.
    :return: An initialized transformation object.
    """
    if dynamic:
        # Only Walsh transform will be initialized with dynamic=True
        return class_name(n, f, boundary_n, dynamic=True)
    else:
        return class_name(n, f, boundary_n=boundary_n)


def discard_percentage_coefficients(transformation: Transformation, coefficients: np.ndarray, p: float) -> (
        tuple[np.ndarray, float] | np.ndarray):
    """
    Discards coefficients percentage for the given transformation with the given percentage.
    :param transformation: The transformation object whose coefficients will be discarded.
    :param p: The percentage of the coefficients to discard.
    :return: A tuple containing the new coefficients and the threshold of coefficients discarded.
    """
    return transformation.discard_coefficients_percentage(coefficients, p, cli=True)


def calculate_errors(transformation: Transformation, coefficients: np.ndarray) -> tuple[float, float, float]:
    """
    Calculates the errors for the given transformation based on the given coefficients.
    The errors are the L1, squared L2 and L∞ norm of the errors of the transformation.
    :param transformation: The transformation to calculate the errors for.
    :param coefficients: The coefficients of the transformation.
    :return: A tuple containing the L1, squared L2 and L∞ norm of the errors.
    """
    l1 = transformation.get_l1_error(coefficients)
    l2square = transformation.get_squared_l2_error(coefficients)
    linf = transformation.get_linf_error(coefficients)
    return l1, l2square, linf


def dict_to_string(dictionary: dict[str, tf1d.TestFunction1D] | dict[str, tf2d.TestFunction2D]) -> str:
    """
    Transform the dictionary into a single string.
    :param dictionary: The dictionary to transform
    :return: A string representation of the dictionary, of the form '>>> key: value\\n'
    """
    keys: list[str] = list(dictionary.keys())
    values: list[tf1d.TestFunction1D] | list[tf2d.TestFunction2D] = list(dictionary.values())
    output: str = ""
    for key, value in zip(keys, values):
        output += f">>> {pearl.blue(key)}: {value.name_cli}\n"
    return output


def get_sols(n: int) -> list[str]:
    """
    Returns a list containing n randomly binary-colored sol-strings.
    The start of line string is '>>> ' and will either be colored cyan or magenta.
    :param n: The number of elements.
    :return: A list of sol-strings.
    """
    sols: list[str] = []
    bools: list[bool] = u.get_coinflips(n)
    for i in range(n):
        sol: str = pearl.cyan(sol_string) if bools[i] else pearl.magenta(sol_string)
        sols.append(sol)
    return sols


def unknown_command(cmd: str, function: str) -> bool:
    """
    Returns False and prints a helpful error message.
    :param cmd: The unknown command.
    :param function: The name of the function that was called.
    :return: False
    """
    # Unknown command
    print(f"Unknown argument \"{cmd}\" for function {function}.\n"
          f"Call \"help\" for more information.")
    return False


# Maps input strings to TestFunction1D objects (callables) with the corresponding name
function_map_1d: dict[str: tf1d.TestFunction1D] = {
    "sine": tf1d.Sine,
    "cosine": tf1d.Cosine,
    "quadratic": tf1d.Quadratic,
    "quadratic-shift": tf1d.QuadraticShift,
    "cube": tf1d.CubicPolynomial,
    "cube-symmetric": tf1d.CubicPolynomialSymmetric,
    "exponential": tf1d.Exponential,
    "rational": tf1d.Rational,
    "exponential-cosine": tf1d.ExponentialCosine,
    "exponential-sine": tf1d.ExponentialSine,
    "sine-exponential": tf1d.SineExponential,
    "sine-log": tf1d.SineLog,
}

# Maps input strings to TestFunction2D objects (callables) with the corresponding name
function_map_2d: dict[str: tf2d.TestFunction2D] = {
    "sine": tf2d.Sine,
    "cosine": tf2d.Cosine,
    "quadratic-min": tf2d.QuadraticMin,
    "quadratic-mid": tf2d.QuadraticMid,
    "quadratic-max": tf2d.QuadraticMax,
    "cube": tf2d.CubicPolynomial,
    "exponential-add": tf2d.ExponentialAdd,
    "exponential-mult": tf2d.ExponentialMult,
    "rational": tf2d.Rational,
    "sine-exponential": tf2d.SineExponential,
    "taylor-cosine": tf2d.TaylorCosine,
    "cosine-x-square": tf2d.CosineXSquare,
}

# Maps input strings to Image objects with the corresponding name
image_map: dict[str: tf.Image] = {
    "alphafly": tf.Image("images/alphafly.JPG", 256),
    "primex": tf.Image("images/prime_x.png", 256),
    "elite2": tf.Image("images/elite2.png", 256),
    "primex3": tf.Image("images/prime_x3.png", 256),
    "obunga64": tf.Image("images/obunga64.jpeg", 64),
    "obunga512": tf.Image("images/obunga512.JPG", 512),
    "couple": tf.Image("images/couple.pgm", 256),
}


def available_functions(dimensionality: int) -> str:
    """
    Returns a list of available function names, for the given dimensionality.
    :param dimensionality:
    :return:
    """
    if dimensionality == 1:
        return (f"Available {pearl.blue("functions")} in 1D are:\n"
                f"{dict_to_string(function_map_1d)}")
    elif dimensionality == 2:
        return (f"Available {pearl.blue("functions")} in 2D are:\n"
                f"{dict_to_string(function_map_2d)}")
    else:
        return available_functions(1) + available_functions(2)


def cli_help() -> None:
    """
    Print help information for the command-line interface.
    This contains all methods that can be called, as well as their required and optional parameters.
    :return:
    """
    print(f"This command-line interface uses the following syntax:\n"
          f"{pearl.blue("<<")}function{pearl.blue(">>")} {pearl.red("<<")}required arguments{pearl.red(">>")} "
          f"{pearl.green("<<")}optional arguments{pearl.green(">>")}\n"
          f"Most arguments can either be given as <argument>=<value> or just <value> or <argument>, if this identifies them.\n"
          f"The following functions are available:\n"
          # plot
          f"{pearl.blue(sol_string)}plot{pearl.blue(eol_string)}\n"
          f"{pearl.red(sol_string2)}transformation | coefficients | error | base{pearl.red(eol_string2)} [what is to be plotted]\n"
          f" {pearl.green(sol_string2)}modified = True {pearl.green(eol_string2)} [uses the modified coefficients and transformation values for plots]\n"
          f" {pearl.green(sol_string2)}sorted = True   {pearl.green(eol_string2)} [sorts the coefficients before plotting]\n"
          f" {pearl.green(sol_string2)}first-n = ...   {pearl.green(eol_string2)} [shows only the coefficients of the first n base functions]\n"
          f" {pearl.green(sol_string2)}subtitle = \"...\"{pearl.green(eol_string2)} [adds a subtitle to the plot]\n"
          f" {pearl.green(sol_string2)}original = True {pearl.green(eol_string2)} [plots the original function]\n"
          # discard coefficients
          f"{pearl.blue(sol_string)}discard-coefficients{pearl.blue(eol_string)}\n"
          f"{pearl.red(sol_string2)}absolute | relative | percentage | level-sum | level-square{pearl.red(eol_string2)} [the method of discarding the coefficients]\n"
          f"{pearl.red(sol_string2)}epsilon = ...{pearl.red(eol_string2)} [discards coefficients based on this threshold]\n"
          # print
          f"{pearl.blue(sol_string)}print{pearl.blue(eol_string)}\n"
          f"{pearl.red(sol_string2)}info{pearl.red(eol_string2)} [what is to be printed]\n"
          f" {pearl.green(sol_string2)}errors = True {pearl.green(eol_string2)} [calculates errors]\n"
          f"\n"
          f"To exit this interface, use \"exit\".")

    # TODO: plot base matrix, i.e., walsh matrix, wavelet matrix, ...
    # TODO: update print bounds
    # TODO: new compare function: two or three transforms?


def maximum_error(transformations: dict[str, Transformation], function_values: dict[str:np.ndarray],
                  values: dict[str, np.ndarray], modified: bool, modified_values: dict[str, np.ndarray]) -> float:
    """
    Determine the maximum error for a number of transformations,
    i.e., the largest difference between their values and the true functions values.
    :param transformations: A dictionary of transformations.
    :param function_values: The values of the true function.
    :param values: A dictionary containing the values of the transformations.
    :param modified: A boolean indicating whether to use the modified transformation values or not.
    :param modified_values: A dictionary containing the values of the modified transformations.
    :return: The maximum error, i.e., the largest difference between a transformation's values and the true functions values.
    """
    maxes: list[float] = []
    for transformation in transformations:
        if modified:
            transform_values = modified_values[transformation]
        else:
            transform_values = values[transformation]
        f_vals = function_values[transformation]
        maxes.append(u.determine_maximum_distance(transform_values, f_vals))
    return max(maxes)


def minimum_absolute_value(transformations: dict[str, Transformation], values: dict[str, np.ndarray],
                           modified: bool, modified_values: dict[str, np.ndarray], first_n: int) -> float:
    """
    Determine the minimum absolute value > 0 of arrays given in dictionaries.
    :param transformations: A dictionary of transformations.
    :param values: A dictionary containing the values per transformation.
    :param modified: A boolean indicating whether to use the modified values or not.
    :param modified_values: A dictionary containing the modified values per transformation.
    :param first_n: An integer indicating how many values to show.
    :return: The minimum absolute value > 0 of the given arrays, if present, else 1e-17.
    """
    mins: list[float] = []
    for transformation in transformations:
        if modified:
            if first_n > 0:
                values_to_check = modified_values[transformation][:first_n]
            else:
                values_to_check = modified_values[transformation]
        else:
            if first_n > 0:
                values_to_check = values[transformation][:first_n]
            else:
                values_to_check = values[transformation]
        absolute_values_to_check = abs(values_to_check)
        try:
            mins.append(np.min(absolute_values_to_check[absolute_values_to_check > 0]))
        except ValueError:
            mins.append(1e-17)
    return min(mins)


def maximum_absolute_value(transformations: dict[str, Transformation], values: dict[str, np.ndarray],
                           modified: bool, modified_values: dict[str, np.ndarray], first_n: int) -> float:
    """
    Determine the maximum absolute value of arrays given in dictionaries.
    :param transformations: A dictionary of transformations.
    :param values: A dictionary containing the values per transformation.
    :param modified: A boolean indicating whether to use the modified values or not.
    :param modified_values: A dictionary containing the modified values per transformation.
    :first_n: An integer indicating how many values to show.
    :return: The maximum absolute value of the given arrays.
    """
    maxs: list[float] = []
    for transformation in transformations:
        if modified:
            if first_n > 0:
                values_to_check = modified_values[transformation][:first_n]
            else:
                values_to_check = modified_values[transformation]
        else:
            if first_n > 0:
                values_to_check = values[transformation][:first_n]
            else:
                values_to_check = values[transformation]
        maxs.append(np.max(abs(values_to_check)))
    if max(maxs) == 0:
        return 1
    return max(maxs)


def maximum_value(transformations: dict[str, Transformation], values: dict[str, np.ndarray],
                  modified: bool, modified_values: dict[str, np.ndarray]) -> float:
    """
    Determine the maximum value of arrays given in dictionaries.
    :param transformations: A dictionary of transformations.
    :param values: A dictionary containing the values per transformation.
    :param modified: A boolean indicating whether to use the modified values or not.
    :param modified_values: A dictionary containing the modified values per transformation.
    :return: The maximum value of the given arrays.
    """
    maxs: list[float] = []
    for transformation in transformations:
        if modified:
            values_to_check = modified_values[transformation]
        else:
            values_to_check = values[transformation]
        maxs.append(np.max(values_to_check))

    return max(maxs)


def minimum_value(transformations: dict[str, Transformation], values: dict[str, np.ndarray],
                  modified: bool, modified_values: dict[str, np.ndarray]) -> float:
    """
    Determine the minimum value of arrays given in dictionaries.
    :param transformations: A dictionary of transformations.
    :param values: A dictionary containing the values per transformation.
    :param modified: A boolean indicating whether to use the modified values or not.
    :param modified_values: A dictionary containing the modified values per transformation.
    :return: The minimum value of the given arrays.
    """
    mins: list[float] = []
    for transformation in transformations:
        if modified:
            transform_coefficients = modified_values[transformation]
        else:
            transform_coefficients = values[transformation]
        mins.append(np.min(transform_coefficients))

    return min(mins)


def remove_trailing_newline(string: str) -> str:
    """
    Remove trailing newline characters from a string.
    :param string: The string to remove trailing newline characters from.
    :return: The string with trailing newline characters removed.
    """
    return string.rstrip("\n")


def whitespaces(n: int) -> str:
    """
    Creates a string with n whitespace characters.
    :param n: The number of whitespace characters.
    :return: A string with n whitespace characters.
    """
    return " " * n


mysterious_commands: list[str] = [
    "yooo!", "yo", "yoo", "yooo",
]


def do_something_mysterious(argument: str):
    """
    Does something mysterious.
    :param argument: A mysterious argument.
    """
    if argument in ["yooo!", "yo", "yoo", "yooo"]:  # https://www.asciiart.eu/text-to-ascii-art
        print(":::   :::  ::::::::   ::::::::   ::::::::  :::\n"
              ":+:   :+: :+:    :+: :+:    :+: :+:    :+: :+:\n"
              " +:+ +:+  +:+    +:+ +:+    +:+ +:+    +:+ +:+\n"
              "  +#++:   +#+    +:+ +#+    +:+ +#+    +:+ +#+\n"
              "   +#+    +#+    +#+ +#+    +#+ +#+    +#+ +#+\n"
              "   #+#    #+#    #+# #+#    #+# #+#    #+#    \n"
              "   ###     ########   ########   ########  ###")


def do_something_suspicious():
    """
    Does something suspicious if the stars align correctly.
    """
    suspicious_things: list[str] = []
    suspicious_things.append(  # https://www.twitchquotes.com/copypastas/3822
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣤⣤⣤⣤⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠛⠉⠙⠛⠛⠛⠛⠻⢿⣿⣷⣤⡀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⣼⣿⠋⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠈⢻⣿⣿⡄⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠀⠀⣠⣶⣾⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣄⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⢰⣿⣿⣯⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣷⡄⠀\n"
        "⠀⣀⣤⣴⣶⣶⣿⡟⠀⠀⠀⢸⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⠀\n"
        "⢰⣿⡟⠋⠉⣹⣿⡇⠀⠀⠀⠘⣿⣿⣿⣿⣷⣦⣤⣤⣤⣶⣶⣶⣶⣿⣿⣿⠀\n"
        "⢸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀\n"
        "⣸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠉⠻⠿⣿⣿⣿⣿⡿⠿⠿⠛⢻⣿⡇⠀⠀\n"
        "⣿⣿⠁⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣧⠀⠀\n"
        "⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀\n"
        "⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀\n"
        "⢿⣿⡆⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀\n"
        "⠸⣿⣧⡀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⠃⠀⠀\n"
        "⠀⠛⢿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⣰⣿⣿⣷⣶⣶⣶⣶⠶⠀⢠⣿⣿⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⣽⣿⡏⠁⠀⠀⢸⣿⡇⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⢹⣿⡆⠀⠀⠀⣸⣿⠇⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⢿⣿⣦⣄⣀⣠⣴⣿⣿⠁⠀⠈⠻⣿⣿⣿⣿⡿⠏⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠈⠛⠻⠿⠿⠿⠿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "   amogus")
    suspicious_things.append(  # https://copypastatext.com/obunga/
        "⣿⡿⣿⢿⡿⣟⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠿⣿⢿⡿⣿⣿\n"
        "⣿⣿⣻⡿⠏⠁⠀⠀⠀⡀⡀⡀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣻⣿\n"
        "⣿⣾⣟⠏⠀⠀⠀⠄⢕⠐⠔⠐⠌⠌⡪⢐⠐⡀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣽⣿\n"
        "⣿⡷⡟⠀⠀⠀⠠⢁⠁⣀⣤⣀⠀⢁⠂⡑⠐⠀⠀⢀⡤⠤⡀⠀⠀⠀⠀⠀⠀⠈⣿⣾\n"
        "⣿⡟⠁⠀⠀⠀⢌⠀⡼⠁⠀⠀⠑⠀⣧⡦⢁⠀⢠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣽\n"
        "⣿⡇⠀⠀⠀⠠⡑⡀⡇⠀⠀⠀⠀⠀⡝⡎⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿\n"
        "⣿⣿⠀⠀⠀⢌⠢⡂⠌⠂⠀⠀⡠⢸⠸⡐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿\n"
        "⣿⣽⣆⡠⡘⡌⣪⢸⡸⣘⠜⡌⡪⡪⣎⠆⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣽\n"
        "⣿⡿⣽⢰⢱⢱⢕⡳⣝⡎⡎⡪⠊⠊⠪⠐⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⣺⡷⣿\n"
        "⣿⡿⡇⡇⡇⡗⡇⡏⡇⡓⡕⡐⢅⠍⡌⠄⠀⠀⠀⠀⠀⠀⠄⠀⠀⠀⠀⠀⠀⢸⣿⢿\n"
        "⣿⡿⡇⡇⡺⣪⢏⣗⢕⠅⢂⢌⢮⡪⡂⠡⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⢈⣿⢿\n"
        "⣿⡿⣷⡱⣙⢮⣗⢗⡕⢅⢸⢼⡳⡝⡄⢑⠀⠀⠀⠀⢀⠀⠀⢂⠐⠀⠀⠀⠀⢠⡿⣿\n"
        "⣿⣟⣿⢷⡪⡪⡎⡧⡣⡱⡹⣽⢯⡫⡢⠠⡁⠂⠀⠈⠀⠀⠂⢐⠀⠂⠀⠀⣠⡾⣿⣻\n"
        "⣿⣯⣿⢿⣟⡎⡎⡎⡆⠢⡫⣯⣗⢽⡀⠂⢕⠈⠀⠀⠀⠁⠄⠐⢈⠀⠀⢰⣯⣿⣟⣿\n"
        "⣿⣻⣾⡿⣟⣯⢪⠪⡂⠡⡹⣳⡻⣕⢅⢱⠱⢀⠁⠀⠀⠁⠂⢈⠀⠀⠀⣿⣻⣾⢿⣽\n"
        "⣿⣟⣷⣿⡿⣿⣎⢎⠂⢌⢎⡗⡝⡜⠔⢈⢊⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⢿⣽⣿⣻\n"
        "⣿⡿⣽⣷⡿⣟⣷⢱⠡⠡⠨⢪⠨⠨⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⢟⣿⡾⣿\n"
        "⣿⣟⣿⢷⣿⢿⢫⠢⡣⠈⠈⠀⠀⠁⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⢄⠉⢻⣿\n"
        "⣿⣯⣿⣟⣿⡝⡜⡰⠨⠨⢪⢪⠢⡂⡂⠁⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣳⠀⠀⠹\n"
        "⣿⣻⣾⢿⠝⡧⡣⡪⡨⢊⠢⠣⠣⠣⡐⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⣕⢿⠀⠀⠀\n"
        "⣿⠟⠙⠁⢠⡗⡅⡇⡪⡠⢑⢍⠎⡅⡢⢁⠂⠀⠄⠀⠀⠀⠀⠀⠀⡰⣕⢽⢕⠀⠀⠀\n"
        "⠁⠀⠀⠀⢸⣷⠨⡪⡊⢆⠆⡂⢇⢕⠨⢐⠈⡀⠀⠀⠀⠀⠀⡠⡣⡳⣕⢽⠁⠀⠀⠀\n"
        "⠀⠀⠀⠀⣟⣿⡌⢪⢊⢎⢪⠢⡂⠄⠁⠀⠀⠀⠀⠀⢀⢠⢪⡪⡮⡳⣕⠇⠁⠀⠀⠀\n"
        "   Obunga"
    )

    suspicious_things.append(  # https://copypastatext.com/big-chungus-ascii/
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣧⠀⠀⠀⢰⡿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡟⡆⠀⠀⣿⡇⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⣿⠀⢰⣿⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡄⢸⠀⢸⣿⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡇⢸⡄⠸⣿⡇⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⢸⡅⠀⣿⢠⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣥⣾⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⡿⡿⣿⣿⡿⡅⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠉⠀⠉⡙⢔⠛⣟⢋⠦⢵⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣄⠀⠀⠁⣿⣯⡥⠃⠀⢳⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡇⠀⠀⠀⠐⠠⠊⢀⠀⢸⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⡿⠀⠀⠀⠀⠀⠈⠁⠀⠀⠘⣿⣄⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⣠⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣷⡀⠀⠀⠀\n"
        "⠀⠀⠀⣾⣿⣿⣿⣿⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣧⠀⠀\n"
        "⠀⠀⡜⣭⠤⢍⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢛⢭⣗⠀\n"
        "⠀⠀⠁⠈⠀⠀⣀⠝⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠠⠀⠀⠰⡅\n"
        "⠀⠀⢀⠀⠀⡀⠡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠔⠠⡕⠀\n"
        "⠀⠀⠀⣿⣷⣶⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠀⠀⠀⠀\n"
        "⠀⠀⠀⠘⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⠀⠀⠀⠀⠀\n"
        "⠀⠀⠀⠀⠈⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠉⢆⠀⠀⠀⠀\n"
        "⢀⠤⠀⠀⢤⣤⣽⣿⣿⣦⣀⢀⡠⢤⡤⠄⠀⠒⠀⠁⠀⠀⠀⢘⠔⠀⠀⠀⠀\n"
        "⠀⠀⡐⠈⠁⠈⠛⣛⠿⠟⠑⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
        "⠀⠉⠑⠒⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ \n"
        "   Big Chungus"
    )
    not_quite_so_suspicious_things: list[str] = []
    not_quite_so_suspicious_things.append(pearl.rainbow("This is not getting you anywhere."))
    not_quite_so_suspicious_things.append("Stop it!")
    fate = random.random()
    if fate < 0.01:
        print(random.choice(suspicious_things))
    elif fate < 0.04:
        print(random.choice(not_quite_so_suspicious_things))
