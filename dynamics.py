import re

import numpy as np
from sympy import lambdify
from numba import jit, prange


def parse(expr):
    """
    Takes an expression using usual math conventions and outputs an expression
    using python math conventions.
    """
    # Add explicity multiplications to match python conventions

    # The cases below add product symbols to products of variables, constants,
    # functions or parentheses.

    res = re.sub(
        "[0-9.]+[A-Za-z\(]", lambda x: f"{x.group(0)[:-1]}*{x.group(0)[-1]}", expr
    )
    res = re.sub("[zCI][A-Za-z\(]", lambda x: f"{x.group(0)[0]}*{x.group(0)[1]}", res)
    res = re.sub("[A-Za-z\)][zCI]", lambda x: f"{x.group(0)[0]}*{x.group(0)[1]}", res)
    res = res.replace(")(", ")*(")

    # Change power notation to match python conventions.
    res = res.replace("^", "**")
    res = res.replace("I", "1.0j")

    return res


def to_function(expr: str):
    """
    Outputs a numba function given by the input expression. The expression must
    be in the variable z, and can contain a parameter C.
    """
    return jit(nopython=True)(lambdify(["z", "C"], parse(expr), "numpy"))


@jit(nopython=True)
def orbit(f, z, c, max_iter, radius):
    iterates = np.zeros(max_iter, dtype=np.complex128)
    for i in range(max_iter):
        iterates[i] = z
        z = f(z, c)
        if np.abs(z) >= radius:
            iterates = iterates[:i]
            break

    return iterates


@jit(nopython=True)
def escape_time(f, z, c, max_iters, radius):
    z2 = f(z, c)
    inv_radius = 1 / (1000 * radius)

    for i in range(max_iters):
        if abs(z) >= radius:
            return ((1 / 256) * (i + 1 - np.log2(np.log2(abs(z))))) % 1

        if abs(z2 - z) <= inv_radius:
            return ((1 / 256) * (i + 1 - np.log2(-np.log2(abs(z2 - z))))) % 1

        z = f(z, c)
        z2 = f(f(z2, c), c)

    return np.nan


@jit(nopython=True, parallel=True)
def escape_grid(f, center, c, diam, grid, iters, esc_radius, c_space=False):
    h = grid.shape[0]
    w = grid.shape[1]

    # Assumes that diam is the length of the grid in the x direction
    delta = diam / w

    # Computes the complex number in the southwest corner of the grid
    # Assumes dimensions are even so center is not a point in the grid
    # This is why there is an adjustment of half delta on each direction
    z0 = center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)

    # c_space tells if grid is in parameter space or dynamical space
    # This is true for the mandelbrot set and false for Julia sets.

    if c_space:
        for n in prange(w):
            dx = n * delta
            for m in prange(h):
                dy = m * delta
                color = escape_time(f, c, z0 + complex(dx, dy), iters, esc_radius)
                grid[m, n] = color
    else:
        for n in prange(w):
            dx = n * delta
            for m in prange(h):
                dy = m * delta
                color = escape_time(f, z0 + complex(dx, dy), c, iters, esc_radius)
                grid[m, n] = color
