# Walsh-Tranformator

This program emerged as part of my bachelors thesis "Walsh-Funktionen als Ansatzfunktionen für dünne Gitter" ("(Walsh Functions as Ansatz Functions for Sparse Grids") in 2025 at the University of Stuttgart. 
It allows to compute Walsh-Transforms of one and two dimensional functions, manipulate their coefficients and calculate transform errors and compare them to other transforms, which also use piecewise-constant base functions. 
To obtain higher-dimensional base functions, a tensor product of one-dimensional base functions is used.

> [!NOTE]
> Importantly, efficiency or speed were not the primary concern during implementation, which is why some code may seem redundant, inefficient or otherwise unexplainable to some user.

If you encounter bugs or have other suggestions or wishes, please open an issue in this repository.

## Requirements

- Python 3.12 or newer
- pip for installing dependencies
- git for cloning this repository :)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/orlandoisepic/walsh-tranformator
cd walsh-transformator
```
2. Create a new Python Virtual Environment (venv; optional, but recommended):
```bash
python -m venv venv
source venv/bin/activate
```
3. Install required dependencies:
```bash
pip install .
```

## Usage

All transformations can be used stand-alone, by instantiating a new object, in their respective definition files or in any other file. 

To compare transforms, it is recommended to use the CLI. It can be accessed through 

```python
python cli.py -d ... -n ... -f ... [-b ... ] [-dyn]
```

The requiered arguments are as follows: 
* ```-d``` is the dimensionality of the function to be transformed, i.e., 1 or 2
* ```-n``` 2^n is the number of base functions per dimension
* ```-f``` is the name of the of function according the function maps (```utility/cli_helper.py```'s ```function_map_1d```, ```function_map_2d``` or ```image_map```)

Optional arguments are:
* ```-b``` 2^b is the number of boundaries which will be calculated for Walsh-Functions. See the paragraph "Trick" in _Modifications for Walsh Transform_ for more details.
* ```-dyn``` If true, the 2D Walsh tranformation will be use a dynamic ordering of base functions based on the absolute values of the 1D coefficients. See paragraph "Dynamic Order" in _Modifications for Walsh Transform_ for more details.

For available functions and commands in the CLI, use ```help```.

## Project Structure

```
walsh-transformator/
├── cli.py
├── images/
│   └── ...                          # Images for transformations
├── pyproject.toml
├── pytest.ini
├── tests/
│   └── ...                          # A few tests
├── transformations/
│   ├── some_transform/
│   │   ├── some_base_function.py
│   │   └── some_transform.py
│   └── ...                          # More transformations
└── utility
    ├── cli_helper.py                
    ├── cli_input.py
    ├── color.py
    ├── templates                    # Abstract base classes
    │   ├── base_functions.py
    │   ├── base_transformations.py
    │   └── test_functions.py
    ├── test_functions_1d.py        # Functions to be transformed
    ├── test_functions_2d.py
    ├── timer.py
    └── utils.py                    # Utility functions
```

## Features

Functions to be transformed can be of type ```TestFunction1D```, ```TestFunction2D``` or ```Ìmage``` which inherit some methods, such as ```plot()``` from an asbtract parent class.
Arbitrary one- and two-dimensional functions and images f can be defined in ```utility/test_functions_1d.py``` and ```utility/test_functions_2d.py```, respectively. To use them in the CLI, they have to added to ```utility/cli_helper.py```'s ```function_map_1d```, ```function_map_2d``` or ```image_map```, depending on their type.
A function f needs to define methods ```evaluate(...)```, ```evaluate_integral(...)```, and, if one wishes to compute errors, ```evaluate_integral_squared(...)```. 

Base functions are always piecewise constant (meaning their values can be stored in an array), hierarchical in the sense that for 2^n (total) base functions, there are n+1 levels, with 2^{s-1} base functions per level, and one scale function if s=0, and orthogonal, in the sense of the L^2 scalar product, to each other.
Base functions can currently be one- or two-dimensional and inherit methods from an abstract base class. A new base function needs to define the ```__init__(order, n)``` method, to define the properties ```values``` (an array of all-values) or ```x_values``` and ```y_values``` for two-dimensional base functions. 
Here, ```n``` defines the total number of base functions, i.e. 2^n, which is also the length of the ```values``` array. The order is the number of the base function if all 2^n base functions are enumerated in an arbitrary way.

Transforms inherit from an abstract parent class. Methods a new transform has to define include ```__init__(n, f, [boundary_n])```, where ```n``` defines the number of base functions (i.e., 2^n per dimension), ```f``` is the function to be transformed and the optional parameter ```boundary_n``` defines number of boundaries which will be calculated for Walsh-Functions. See the paragraph "Trick" in _Modifications for Walsh-Transform_ for more details. This method is used to define a property ```base_functions```, which is a d-dimensional array for storing all base functions, in some ordering. 
Transforms have access to all kinds of methods, such as discarding coefficients in multiple ways, plotting the transformations and errors, as well as calculating different L^p norms of the error. 

### Modifications for Walsh Transform 

As the objective of my work was to evaluate Walsh-Functions behaviors, they have a few additional features, to improve their rather wild coefficient decay.

#### Trick

Following [this paper](https://doi.org/10.1016/j.jat.2015.12.002), it is possible to define boundaries for the coeficients of Walsh functions. Ordering the coefficients of the base functions according to these boundaries will result in a very smooth coefficient decay.
This can be taken further, by calculating the boundaries for 2^m (more base functions than will actually be used) and then using the only the first 2^n base functions.
Currently, this is implemented through the parameter ```boundary_n``` in the ```__init(...)__``` method of Walsh transformations and the one-dimensional exponential function e^x. 
For many functions, this results in better approximations than the standard (dyadic or sequential) ordering. 

#### Dynamic Coefficient Order

Another way to order base functions is by calculating a dynamic order, based on the absolute values of the coefficients of the 1D base functions. 
For separable functions f, i.e., f(x,y)=g(x)*h(y), this improves coefficient decay for sparse grids by sorting the base functions according to their coefficient values. 
Howerver, for non-separable functions, this seems to rather be detrimental. 
The dynamic ordering is implemented in ```dynamic_order_sepdim()``` in ```transformations/walsh_transformation_2d/walsh_transformation_2d.py```.
