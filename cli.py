import sys
import argparse

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from prompt_toolkit.formatted_text import ANSI

from utility import utils as u
from utility import cli_helper as helper
from utility import cli_input
from utility.color import Painter
from utility.timer import Timer
from utility.templates.test_functions import TestFunction, TestFunctionType
from utility.templates.base_transformations import Transformation
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation
from transformations.walsh_transformation_2d.walsh_transformation_2d import WalshTransformation2D
from transformations.wavelet_transformation_1d.wavelet_transformation_1d import WaveletTransformation
from transformations.wavelet_transformation_2d.wavelet_transformation_2d import WaveletTransformation2D
from transformations.discrete_cosine_transformation_1d.discrete_cosine_transformation_1d import \
    DiscreteCosineTransformation
from transformations.discrete_cosine_transformation_2d.discrete_cosine_transformation_2d import \
    DiscreteCosineTransformation2D


def handle_input(command: str) -> bool:
    """
    Handles the input from the command line, by calling the appropriate helper functions
    :param command: The input from the command line
    :return: A boolean indicating if the command was successfully executed.
    """
    parts = command.split(" ")
    if not parts:
        return False

    cmd = parts[0]
    args = parts[1:]
    if len(args) == 0:
        print(f"No argument was given for function {cmd}.\n"
              f"Call \"help\" for more information.")
        return False

    if cmd == "plot":
        return plot_helper(args)
    elif cmd == "discard-coefficients" or cmd == "discard":
        return coefficient_helper(args)
    elif cmd == "print":
        return print_helper(args)
    else:
        print(f"Unknown function {cmd}.\n"
              f"Call \"help\" for more information.")
        return False


def print_helper(input: list[str]) -> bool:
    """
    This helper function is responsible for printing information about the transformations,
    e.g., the number of zero-coefficients, et cetera.
    :param input: The arguments from the command-line
    :return: A boolean indicating if the command was successfully executed.
    """
    cmd = input[0] if len(input) >= 1 else ""
    args = input[1:]  # No args are expected currently
    # Create a dict of the commands with the style key=value, to extract them safely later
    args_dict: dict[str, str | bool | float] = {}

    n: int = cli_arguments["n"]
    dimensionality: int = cli_arguments["dimensionality"]
    method: str = cli_arguments["method"]
    total: int = 2 ** (n * dimensionality)  # The total number of base functions is (2ⁿ)ᵈ = 2ⁿᵈ
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args_dict[key] = value
        elif arg in ["errors", "e"]:
            args_dict["errors"] = True

    errors: bool = bool(args_dict.get("errors", False))

    if cmd == "info":
        output: str = ("Information about the transformations:\n"
                       f"{helper.eol_string2}Method of calculating coefficients: {pearl.green(method)}\n"
                       f"{helper.eol_string2}Total number of coefficients: {pearl.green(str(total))}\n")

        information_to_gather: list[tuple[Transformation, np.ndarray, np.ndarray, float, bool, dict]] = []
        for transformation_str in transformations:
            transformation = transformations[transformation_str]
            transform_coefficients = coefficients[transformation_str]
            modified_transform_coefficients = modified_coefficients[transformation_str]
            epsilon = epsilons[transformation_str]

            # Create tuples for executor
            information_to_gather.append((transformation, transform_coefficients, modified_transform_coefficients,
                                          epsilon, errors, history_information))

        with ProcessPoolExecutor(max_workers=3) as executor:
            information: list[str] = list(executor.map(helper.gather_information, *zip(*information_to_gather)))
        for info in information:
            output += info
        print(helper.remove_trailing_newline(output))
    else:
        return helper.unknown_command(cmd, "print")
    return True


def plot_helper(input: list[str]) -> bool:
    """
    This helper function calls the appropriate functions with the correct parameters
    if the input command is plot.
    :param input: The arguments of the original input,
    anything after <command> from an input <command> <required arguments> <optional arguments>
    :return: A boolean indicating if the command was successfully executed.
    """
    # It is possible to plot coefficients, the transformation and the test function
    # cmd differentiates what to plot, args are the arguments of the plot
    cmd = input[0] if len(input) >= 1 else ""
    args = input[1:]
    # Create a dict of the commands with the style key=value, to extract them safely later
    args_dict: dict[str, str | bool | float] = {}

    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args_dict[key] = value
        # Manual checks to allow shorter commands without key=value
        elif arg == "sorted" or arg == "s":
            args_dict["sorted"] = True
        elif arg.isdigit():
            args_dict["first-n"] = int(arg)
        elif arg == "modified" or arg == "m":  # Use modified coefficients for plots
            args_dict["modified"] = True
        elif arg == "original" or arg == "o":
            args_dict["original"] = True

    # Get all possible arguments, with a default value
    sorted: bool = bool(args_dict.get("sorted", False))
    subtitle: str = args_dict.get("subtitle", "")
    first_n: int = int(args_dict.get("first-n", 0))
    modified: bool = bool(args_dict.get("modified", False))
    original: bool = bool(args_dict.get("original", False))

    fig = plt.figure(figsize=(15, 5.5), constrained_layout=True)

    ims: list[AxesImage] = []
    axs: list[Axes] = []
    if cmd == "coefficients" or cmd == "c":
        vmin = helper.minimum_absolute_value(transformations, coefficients, modified, modified_coefficients, first_n)
        vmax = helper.maximum_absolute_value(transformations, coefficients, modified, modified_coefficients,
                                             first_n) * 1.5  # so that the largest coefficient is not exactly at the border
        for i, transformation in enumerate(transformations):
            transformer = transformations[transformation]
            # Get user-defined subtitle again, to avoid using the one from the previous loop iteration
            subtitle: str = args_dict.get("subtitle", "")
            if modified:
                transform_coefficients = modified_coefficients[transformation]
                modified_zeros: int = (transform_coefficients == 0).sum()
                subtitle = f"discarding {modified_zeros} coefficients" + subtitle
            else:
                transform_coefficients = coefficients[transformation]
            im, ax = transformer.plot_coefficients(transform_coefficients,
                                                   sorted=sorted, first_n=first_n, subtitle=subtitle,
                                                   cli=True, fig=fig, index=i + 1, vmin=vmin, vmax=vmax)
            ims.append(im)
            axs.append(ax)
    elif cmd == "transformation" or cmd == "t":
        vmax = helper.maximum_value(transformations, values, modified, modified_values)
        vmin = helper.minimum_value(transformations, values, modified, modified_values)
        if vmax == vmin:  # Add a value to create a gradient, in order for the colorbar to work correctly
            vmax += 1
            vmin -= 1
        for i, transformation in enumerate(transformations):
            transformer = transformations[transformation]
            if modified:
                transform_values = modified_values[transformation]
            else:
                transform_values = values[transformation]
            f_vals = function_values[transformation]
            im, ax = transformer.plot_transformation(transform_values, f_vals,
                                                     subtitle=subtitle, cli=True, fig=fig,
                                                     index=i + 1, vmin=vmin, vmax=vmax)
            ims.append(im)
            axs.append(ax)
    elif cmd == "error" or cmd == "e":
        # Determine the maximum
        vmax = helper.maximum_error(transformations, function_values, values, modified, modified_values)
        for i, transformation in enumerate(transformations):
            transformer = transformations[transformation]
            if modified:
                transform_values = modified_values[transformation]
            else:
                transform_values = values[transformation]
            f_vals = function_values[transformation]
            im, ax = transformer.plot_error_absolute(transform_values, f_vals,
                                                     subtitle=subtitle, cli=True, fig=fig, index=i + 1, vmin=0,
                                                     vmax=vmax)
            ims.append(im)
            axs.append(ax)
            # TODO is this helpful? would need to be moved to a different function / elif, as plots are shown only after ifs
            # transformer.plot_error_relative(transform_values, function_values,
            #                                 subtitle=subtitle, cli=True, fig=fig, index=i + 1)
    elif cmd == "base" or cmd == "b":
        outer_grid = fig.add_gridspec(1, 3)
        for i, transformation in enumerate(transformations):
            transformer = transformations[transformation]
            transformer.plot_base_matrix(cli=True, fig=fig, index=i + 1, outer_grid=outer_grid)
    else:
        return helper.unknown_command(cmd, "plot")
    dimensionality = cli_arguments["dimensionality"]
    # If the coefficients are sorted, then the plot is 1D and does not need a colorbar; base matrices also don't need a colorbar
    if not sorted and dimensionality == 2 and cmd not in ["base", "b"]:
        fig.colorbar(ims[0], ax=np.array(axs).ravel())
    plt.show()
    # Plot the original function, if requested and 2D.
    # In 1D, the original function is already displayed in the transformation's plot
    if original and dimensionality == 2:
        f.plot()
    return True


def coefficient_helper(input: list[str]) -> bool:
    """
    This helper function calls the appropriate functions with the correct parameters
    if the input command is discard-coefficients.
    :param input: The arguments of the original input,
    anything after <command> from an input <command> <required arguments> <optional arguments>
    :return: A boolean indicating if the command was successfully executed.
    """
    # Coefficients can be discarded by absolute value, by relative value or by percentage
    args = input if len(input) >= 1 else None
    args_dict: dict[str, str | bool | float] = {}

    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            args_dict[key] = value
        # Manual checks to allow shorter commands
        elif arg in ["absolute", "a", "relative", "r", "percentage", "p", "s", "sparse"]:
            args_dict["method"] = arg
        elif u.is_float(arg):
            args_dict["epsilon"] = float(arg)

    eps: float = float(args_dict.get("epsilon", 0.0))
    method: str = args_dict.get("method", None)

    # Check that necessary arguments were given
    if method is None:
        print(f"Unknown method to discard coefficients.\n"
              f"Possible methods are: absolute, relative, percentage, level-sum, level-square")
        return False
    if method in ["s", "sparse"]:
        if eps == 0:
            eps = cli_arguments["n"]
    if eps == 0:
        print("No epsilon given for discarding coefficients.")
        return False

    coefficients_to_discard: list[tuple] = []
    for transformation_str in transformations:
        transformation = transformations[transformation_str]
        transform_coefficients = coefficients[transformation_str]
        coefficients_to_discard.append((transformation, transform_coefficients, eps))

    if method == "absolute" or method == "a":
        with ProcessPoolExecutor(max_workers=3) as executor:
            new_coefficients = list(executor.map(helper.discard_coefficients_absolute, *zip(*coefficients_to_discard)))

        for i, transformation_str in enumerate(transformations):
            modified_coefficients[transformation_str] = new_coefficients[i]
            epsilons[transformation_str] = eps

    elif method == "relative" or method == "r":
        with ProcessPoolExecutor(max_workers=3) as executor:
            new_coefficients = list(executor.map(helper.discard_coefficients_relative, *zip(*coefficients_to_discard)))

        for i, transformation_str in enumerate(transformations):
            transform_coefficients = coefficients[transformation_str]
            modified_coefficients[transformation_str] = new_coefficients[i]
            epsilons[transformation_str] = eps * np.abs(transform_coefficients).max()

    elif method == "percentage" or method == "p":
        with ProcessPoolExecutor(max_workers=3) as executor:
            new_coefficients, epsilae = zip(
                *list(executor.map(helper.discard_coefficients_percentage, *zip(*coefficients_to_discard))))

        for i, transformation_str in enumerate(transformations):
            modified_coefficients[transformation_str] = new_coefficients[i]
            epsilons[transformation_str] = epsilae[i]

    elif method == "sparse" or method == "s":
        with ProcessPoolExecutor(max_workers=3) as executor:
            new_coefficients = list(executor.map(helper.discard_coefficients_sparse, *zip(*coefficients_to_discard)))

        for i, transformation_str in enumerate(transformations):
            modified_coefficients[transformation_str] = new_coefficients[i]
            epsilons[transformation_str] = eps

        for i, transformation_str in enumerate(transformations):
            modified_coefficients[transformation_str] = new_coefficients[i]
            epsilons[transformation_str] = eps

    history_information["last_discarding_method_used"] = method
    samples: int = history_information["samples"]

    transformations_to_sample = [
        (violet, modified_coefficients["walsh"], samples),
        (helena, modified_coefficients["wavelet"], samples),
        (keira, modified_coefficients["cosine"], samples),
    ]

    with ProcessPoolExecutor(max_workers=3) as executor:
        sampled_transformation_values = list(
            executor.map(helper.sample_transformation, *zip(*transformations_to_sample)))

    modified_values["walsh"] = sampled_transformation_values[0]
    modified_values["wavelet"] = sampled_transformation_values[1]
    modified_values["cosine"] = sampled_transformation_values[2]

    return True


if __name__ == "__main__":
    session = cli_input.setup_cli()

    theresa: Timer = Timer()
    theresa.start()
    start_time: float = theresa.start_time
    print(f"Starting program...")

    pearl: Painter = Painter()

    # Transformations used 🙂
    helena: WaveletTransformation | WaveletTransformation2D = None
    keira: DiscreteCosineTransformation | DiscreteCosineTransformation2D = None
    violet: WalshTransformation | WalshTransformation2D = None
    # List of all transformations used
    transformations: dict[str, Transformation] = {}
    # List of all coefficients
    coefficients: dict[str, np.ndarray] = {}
    # List of all modified coefficients, e.g., after discarding some
    modified_coefficients: dict[str, np.ndarray] = {}
    # List of all transformation values
    values: dict[str, np.ndarray] = {}
    # List of all transformation values after modifying them (e.g., discarding coefficients)
    modified_values: dict[str, np.ndarray] = {}
    # List of function values per transformation. This is needed because Walsh may have a higher resolution
    function_values: dict[str, np.ndarray] = {}

    # List of all currently used epsilons
    epsilons: dict[str, float] = {"walsh": 0, "wavelet": 0, "cosine": 0}
    # Dict of information for print
    history_information: dict[str, str | int] = {"last_discarding_method_used": "", "samples": 0}

    madeleine = argparse.ArgumentParser()

    madeleine.add_argument("--n", "-n", type=int, default=0, nargs="?", const=0,
                           help="Number of base functions used per dimension is 2^n.", )
    madeleine.add_argument("--dimensionality", "-d", type=int, default=0, nargs="?", const=0,
                           help="Dimensionality of input test-function and Walsh-functions. Possible values are 1 and 2.")
    madeleine.add_argument("--function", "-f", type=str, default="", nargs="?", const="",
                           help="Function to be transformed.")
    madeleine.add_argument("--method", "-m", type=str, default="integration", nargs="?", const="integration",
                           help="Method of computing the coefficients of the Walsh-transformation.")
    madeleine.add_argument("--boundary-n", "-b", type=int, default=0, nargs="?", const=0,
                           help="Advanced feature. 2^boundary_n is the number of base functions for which the boundary will be computed."
                                "Then, the best 2^n base functions are selected for the transformation.")
    madeleine.add_argument("--dynamic", "-dyn", type=bool, default=False, nargs="?", const=True,
                           help="Advanced feature. If true, 2D Walsh functions will be sorted dynamically according to 1D coefficients."
                                "This is only available for 2D functions")

    # Collect all arguments
    args = madeleine.parse_args()
    n: int = args.n
    boundary_n: int = args.boundary_n
    dimensionality: int = args.dimensionality
    function: str = args.function
    method: str = args.method
    f: TestFunction = None
    function_type: TestFunctionType = None
    dynamic: bool = args.dynamic

    cli_arguments: dict[str, int | str] = {
        "n": 0,
        "boundary_n": 0,
        "dimensionality": 0,
        "function": "",
        "method": "",
        "dynamic": False,
    }

    # Check all arguments
    if n <= 0:
        sys.exit(f"{pearl.red("ERROR")} The number of base functions -n 2^n per dimension must be specified.")
    if dimensionality not in [1, 2]:
        sys.exit(f"{pearl.red("ERROR")} Dimensionality -d must be specified. Can be either 1 or 2.")
    if function == "":
        sys.exit(
            f"{pearl.red("ERROR")} Function -f to be transformed must be specified.\n" + helper.available_functions(
                dimensionality) + "Exiting.")
    if method not in ["integration", "interpolation"]:
        sys.exit(f"{pearl.red("ERROR")} Method -m must be either 'interpolation' or 'integration'.")

    print(helper.sol_string + f"Dimensionality: {pearl.green(str(dimensionality) + "D")}")
    print(helper.sol_string + f"Number of base functions per dimension: {pearl.green(str(2 ** n))}")
    print(helper.sol_string + f"Method for computing coefficients: {pearl.green(method)}")
    if boundary_n > n:
        cli_arguments["boundary_n"] = boundary_n
        print(helper.sol_string + f"Number of boundaries to be checked: {pearl.green(str(2 ** boundary_n))}")
    else:
        boundary_n = n

    cli_arguments["n"] = n
    cli_arguments["dimensionality"] = dimensionality
    cli_arguments["method"] = method
    cli_arguments["dynamic"] = dynamic

    # All arguments are fine
    # Image is always two-dimensional
    if function in helper.image_map:
        try:
            f: TestFunction = helper.image_map[function]  # No () needed: Image is already initialized
        except KeyError:
            sys.exit("Function must be specified in image map.")
        print(helper.sol_string + f"Image: {pearl.green(f.name_cli)}")
        dimensionality = 2  # Set dimensionality explicitly to avoid "1D" image and connected values
        # boundary_n is required, but will not be used
        transformations_to_initialize = [
            (WalshTransformation2D, n, f, boundary_n),
            (WaveletTransformation2D, n, f, boundary_n),
            (DiscreteCosineTransformation2D, n, f, boundary_n),
        ]
        function_type = TestFunctionType.IMAGE
    else:
        if dimensionality == 1:
            try:
                f: TestFunction = helper.function_map_1d[function]()  # () To initialize class from the map
            except KeyError:
                sys.exit(
                    f"{pearl.red("ERROR")} Function must be specified in function map.\n" + helper.available_functions(
                        dimensionality) + "Exiting.")
            print(helper.sol_string + f"Function: {pearl.green(f.name_cli)}")
            transformations_to_initialize = [
                (WalshTransformation, n, f, boundary_n),
                (WaveletTransformation, n, f, boundary_n),
                (DiscreteCosineTransformation, n, f, boundary_n),
            ]
        elif dimensionality == 2:
            try:
                f: TestFunction = helper.function_map_2d[function]()  # () To initialize object from the map
            except KeyError:
                sys.exit(
                    f"{pearl.red("ERROR")} Function must be specified in function map.\n" + helper.available_functions(
                        dimensionality) + "Exiting.")
            print(helper.sol_string + f"Function: {pearl.green(f.name_cli)}")
            transformations_to_initialize = [
                (WalshTransformation2D, n, f, boundary_n, dynamic),
                (WaveletTransformation2D, n, f, boundary_n, False),
                (DiscreteCosineTransformation2D, n, f, boundary_n, False),
            ]
        function_type = TestFunctionType.FUNCTION

    cli_arguments["function"] = f.name_cli

    with ProcessPoolExecutor(max_workers=3) as executor:
        transformation_objects = list(
            executor.map(helper.initialize_transformation, *zip(*transformations_to_initialize))
        )

    violet = transformation_objects[0]
    helena = transformation_objects[1]
    keira = transformation_objects[2]

    # Save transformation object to dict
    transformations["walsh"] = violet
    transformations["wavelet"] = helena
    transformations["cosine"] = keira

    theresa.lap()
    print(f"Setup complete. Elapsed time: {theresa.get_last_interval_string()} seconds.")

    # Calculate coefficients
    print("Calculating transformation coefficients...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        coefficient_arrays = list(executor.map(helper.calculate_coefficients, transformations.values()))

    walsh_coefficients = coefficient_arrays[0]
    wavelet_coefficients = coefficient_arrays[1]
    cosine_coefficients = coefficient_arrays[2]

    coefficients["walsh"] = walsh_coefficients
    coefficients["wavelet"] = wavelet_coefficients
    coefficients["cosine"] = cosine_coefficients

    theresa.lap()
    print(f"Calculation done. Elapsed time: {theresa.get_last_interval_string()} seconds.")

    samples: int = 0
    if function_type == TestFunctionType.IMAGE:
        samples = f.resolution  # Sample the image no more than its resolution
    elif dimensionality == 1:
        # If boundary_n is used, more samples may have to be collected
        samples = 1024 if 2 ** boundary_n < 1024 else 2 ** boundary_n
    elif dimensionality == 2:
        samples = 256 if 2 ** boundary_n < 256 else 2 ** boundary_n

    history_information["samples"] = samples
    # Technically, not all transformations need increased samples. However, it is more convenient to do so :)
    # It also makes it easier in the future if other transforms also want more samples.
    f_vals: np.ndarray = f.sample(samples=samples)

    function_values["walsh"] = f_vals
    function_values["wavelet"] = f_vals
    function_values["cosine"] = f_vals

    transformations_to_sample = [
        (violet, walsh_coefficients, samples),
        (helena, wavelet_coefficients, samples),
        (keira, cosine_coefficients, samples),
    ]

    with ProcessPoolExecutor(max_workers=3) as executor:
        sampled_transformation_values = list(
            executor.map(helper.sample_transformation, *zip(*transformations_to_sample)))

    walsh_transformation_values = sampled_transformation_values[0]
    wavelet_transformation_values = sampled_transformation_values[1]
    cosine_transformation_values = sampled_transformation_values[2]

    # Save transformation values in dict
    values["walsh"] = walsh_transformation_values
    values["wavelet"] = wavelet_transformation_values
    values["cosine"] = cosine_transformation_values

    # Copy to be able to use "modified" values immediately
    modified_coefficients = coefficients.copy()
    modified_values = values.copy()

    # Variables for cli output
    input_from_cli: str = ""
    hidden: bool = False
    success: bool = False
    theresa.lap()
    print(f"Final preparations done. Elapsed time: {theresa.get_last_interval_string()} seconds.")
    # Loop to allow plots, different epsilons, et cetera
    while True:
        # A randomly colored, ANSI-corrected string
        sol = ANSI(helper.get_sols(1)[0])
        if not hidden:
            print("Enter next command...")
        input_from_cli: str = session.prompt(message=sol)
        hidden = False
        theresa.lap()
        if "exit" in input_from_cli.lower() or "quit" in input_from_cli.lower():
            break
        elif input_from_cli == "":
            hidden = True
            helper.do_something_suspicious()
            continue
        elif "help" in input_from_cli.lower():
            helper.cli_help()
            success = False
        elif input_from_cli in helper.mysterious_commands:
            helper.do_something_mysterious(input_from_cli)
        else:
            success = handle_input(input_from_cli)
        theresa.lap()
        if success:
            print(f"Command executed. Elapsed time: {theresa.get_last_interval_string()} seconds.")

    theresa.stop()
    sys.exit(f"Exiting. Total time spent: {theresa.get_total_time_string()} seconds.")
