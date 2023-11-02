import re

import numpy as np
from sympy import lambdify
from numba import jit, prange


def jit_function(vars, expr):
    """
    Takes sympy expressions and outputs fast numba jit-functions.
    """
    jitted = jit(nopython=True)(lambdify(vars, expr, "numpy"))

    # Make sure errors in the computation are returned as nan (which can be plotted).
    # Exception is too general, and should be restricted in the future.
    def jitted_no_errors(*args):
        try:
            return jitted(*args)
        except:
            return np.nan

    return jit(nopython=True)(jitted_no_errors)


@jit(nopython=True)
def orbit(f, z, c, max_iter, radius):
    """
    Computes the orbit of a point given a function.
    """
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
    """
    Computes how long it takes for a point to escape to infinity.
    Uses renormalization to make the output continuous.
    """
    for i in range(max_iters):
        if abs(z) >= radius:
            return ((1 / 256) * (i + 1 - np.log2(np.log2(abs(z))))) % 1

        z = f(z, c)

    return np.nan


@jit(nopython=True)
def escape_period(f, z, c, max_iters, radius):
    """
    Computes how long it takes for a point to escape or become close to periodic.
    """
    z2 = f(z, c)
    inv_radius = 1 / (1000 * radius)

    for i in range(max_iters):
        if abs(z) >= radius:
            return (i + 1 - np.log2(np.log2(abs(z)))) / 256

        if abs(z2 - z) <= inv_radius:
            return (i + 1 - np.log2(-np.log2(abs(z2 - z)))) / 256

        z = f(z, c)
        z2 = f(f(z2, c), c)

    return np.nan


@jit(nopython=True, parallel=True)
def mandel_grid(f, center, crit, diam, grid, iters, esc_radius, alg="iter"):
    """
    Find the escape time of a critical point along a grid of parameters.
    The critical point depends on the value of the parameter.
    """
    h = grid.shape[0]
    w = grid.shape[1]

    # Assumes that diam is the length of the grid in the x direction
    delta = diam / w

    # Computes the complex number in the southwest corner of the grid
    # Assumes dimensions are even so center is not a point in the grid
    # This is why there is an adjustment of half delta on each direction
    z0 = center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)

    if alg == "iter":
        for n in prange(w):
            dx = n * delta
            for m in prange(h):
                dy = m * delta
                c = z0 + complex(dx, dy)
                color = escape_time(f, crit(c), c, iters, esc_radius)
                grid[m, n] = color

    elif alg == "period":
        for n in prange(w):
            dx = n * delta
            for m in prange(h):
                dy = m * delta
                c = z0 + complex(dx, dy)
                color = escape_period(f, crit(c), c, iters, esc_radius)
                grid[m, n] = color


@jit(nopython=True, parallel=True)
def julia_grid(f, center, param, diam, grid, iters, esc_radius, alg="iter"):
    """
    Find the escape time of points in a grid, given a function to iterate.
    """
    h = grid.shape[0]
    w = grid.shape[1]

    # Assumes that diam is the length of the grid in the x direction
    delta = diam / w

    # Computes the complex number in the southwest corner of the grid
    # Assumes dimensions are even so center is not a point in the grid
    # This is why there is an adjustment of half delta on each direction
    z0 = center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)

    if alg == "iter":
        for n in prange(w):
            dx = n * delta
            for m in prange(h):
                dy = m * delta
                color = escape_time(f, z0 + complex(dx, dy), param, iters, esc_radius)
                grid[m, n] = color

    elif alg == "period":
        for n in prange(w):
            dx = n * delta
            for m in prange(h):
                dy = m * delta
                color = escape_period(f, z0 + complex(dx, dy), param, iters, esc_radius)
                grid[m, n] = color
