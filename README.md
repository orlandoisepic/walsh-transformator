# Walsh-Tranformator

This program emerged as part of my bachelors thesis "Walsh-Funktionen als Ansatzfunktionen für dünne Gitter" ("(Walsh Functions as Ansatz Functions for Sparse Grids") in 2025 at the University of Stuttgart. 
It allows to compute Walsh-Transforms of one and two dimensional functions, manipulate their coefficients and calculate transform errors and compare them to other transforms, which also use piecewise-constant base functions. 

> [!NOTE] Importantly, efficiency and speed not the primary concern during implementation, which is why some code may seem redundant, inefficient or otherwise unexplainable to some user.

## Requirements

## Installation

## Usage

All transformations can be used stand-alone, by instantiating a new object, in their respective definition files or in any other file. 

To compare transforms, it is best to use the CLI. It can be accessed through 

```python
python cli.py -d ... -n ... -f ... [-b ... ]
```

The requiered arguments are as follows: 
* ```-d``` is the dimensionality of the function to be transformed, i.e., 1 or 2
* ```-n``` 2^n is the number of base functions per dimension
* ```-f``` is the name of the of function according the function maps (```utility/cli_helper.py```'s ```function_map_1d```, ```function_map_2d``` or ```image_map```)
The optional arguments are:
* ```-b``` 2^b is the number of boundaries which will be calculated for Walsh-Functions. See the paragraph "Trick" in _Modifications for Walsh-Transform_ for more details.



## Features

Functions to be transformed can be of type ```TestFunction1D```, ```TestFunction2D``` or ```Ìmage``` which inherit some methods, such as ```plot()``` from an asbtract parent class.
Arbitrary one- and two-dimensional functions and images f can be defined in ```utility/test_functions_1d.py``` and ```utility/test_functions_2d.py```, respectively. To use them in the CLI, they have to added to ```utility/cli_helper.py```'s ```function_map_1d```, ```function_map_2d``` or ```image_map```, depending on their type.
A function f needs to define methods ```evaluate(...)```, ```evaluate_integral(...)```, and, if one wishes to compute errors, ```evaluate_integral_squared(...)```. 

Base functions are always piecewise constant (meaning their values can be stored in an array), hierarchical in the sense that for 2^n (total) base functions, there are n+1 levels, with 2^{s-1} base functions per level, and one scale function if s=0, and orthogonal, in the sense of the L^2 scalar product, to each other.
Base functions can currently be one- or two-dimensional and inherit methods from an abstract base class. A new base function needs to define the ```__init__(order, n)``` method, to define the properties ```values``` (an array of all-values) or ```x_values``` and ```y_values``` for two-dimensional base functions. 
Here, ```n``` defines the total number of base functions, i.e. 2^n, which is also the length of the ```values``` array. The order is the number of the base function if all 2^n base functions are enumerated in an arbitrary way.

Transforms inherit from an abstract parent class. Methods a new transform has to define include ```__init__(n, f, [boundary_n])```, where ```n``` defines the number of base functions (i.e., 2^n per dimension), ```f``` is the function to be transformed and the optional parameter ```boundary_n``` defines number of boundaries which will be calculated for Walsh-Functions. See the paragraph "Trick" in _Modifications for Walsh-Transform_ for more details. This method is used to define a property ```base_functions```, which is a d-dimensional array for storing all base functions, in some ordering. 
Transforms have access to all kinds of methods, such as discarding coefficients in multiple ways, plotting the transformations and errors, as well as calculating different L^p norms of the error. 

### Modifications for Walsh-Transform 

As the objective of my work was to evaluate Walsh-Functions behaviors, they have a few extra requirements. 

#### Trick

Following [this paper](https://doi.org/10.1016/j.jat.2015.12.002), it is possible to define boundaries for the coeficients of Walsh functions. Ordering the coefficients of the base functions according to these boundaries will result in a very smooth coefficient decay.
This can be taken further, by calculating the boundaries for 2^m (more base functions than will actually be used) and then using the only the first 2^n base functions.
Currently, this is implemented through the parameter ```boundary_n``` in the ```__init(...)__``` method of Walsh transformations and the one-dimensional exponential function e^x. 
For many functions, this results in better approximations than the standard ordering. 

#### Dynamic Coefficient Order


## Project Structure
