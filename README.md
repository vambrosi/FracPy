# FracPy

## How to use

Clone this repository and install sympy, numpy, matplotlib, and numba.

To explore the Mandelbrot and Julia sets given by the family $f(z) = z^2 + C$ just run:
```
python ./fracpy.py
```

To explore the plots of other functions or 1-parameter families of functions, you need to open the python interpreter or a jupyter notebook and run the following. First, you have to import `fracpy`'s main class `DSystem` and some symbols from `sympy`:
```
>>> from fracpy import DSystem  # class to store dynamical systems
>>> from sympy.abc import z, c  # z and c are now sympy symbols and can be used in expressions
```
Then you have to create a dynamical system using those variables and then call its `view` method.
```
>>> expr = z**2 - c * z
>>> pol = DSystem(z, expr, crit=c / 2)
>>> pol.view(mandel_center=-1.0, mandel_diam=8.0)
```
`DSystem` takes as arguments the function variable, the expression that determines the functions (it can have at most one parameter), and a critical value to plot the bifurcation locus (that can depend on the parameter).

More examples and details can be found on the [`examples.ipynb`](./examples.ipynb) file.

__REMINDER:__ sympy uses python conventions for math expressions, so:
* `**` is the notation for exponentiation not `^`;
*  All multiplications must be explicit, _e.g_ ``2(1+z)`` is not valid, but ``2*(1+z)`` is.

Also, you can import some basic functions and some math constants via sympy, for example
```
from sympy import E, I, pi, GoldenRatio, sin, cos, sqrt
```
imports the euler constant, the imaginary unit, $\pi$, the golden ratio, sine, cosine, and the square root.

## Shortcuts

* `z + <LeftClick>`: Zooms in on the plot
* `x + <LeftClick>`: Zooms out on the plot
* `s + <LeftClick>`: Centers on pointer coordinates
* `c + <LeftClick>`: Chooses the parameter (only on Mandelbrot plot)
* `t + <LeftClick>`: Draws orbit starting at pointer coordinates (only on Julia plot)
* `r`: Resets view
* `d`: Hides orbit
* `1`: Color only escaping orbits
* `2`: Color bounded orbits by period
* `3`: Color bounded orbits using $|z_n - z_{2n}|$
* `4`: Color bounded orbits by preperiod
* `5`: Color bounded orbits by absolute value of the derivative at attracting cycle
* `6`: Color bounded orbits by argument of the derivative at attracting cycle
* `<LeftArrow>` and `<RightArrow>`: Shift color gradient

All the other settings can be changed by writing on the entries below the plot and pressing `<Enter>`.

## Some Images

Rabbit:

<img src='images/rabbit.png' width='1000'>

Julia sets of the family $C\cos(z)$:

<img src='images/cos_julia.png' width='1000'>

<img src='images/cos_julia1.png' width='1000'>

<img src='images/cos_julia2.png' width='1000'>
