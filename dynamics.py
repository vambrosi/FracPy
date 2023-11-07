import numpy as np
from sympy import lambdify
from numba import jit


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
def escape_time(f, df, z, c, max_iters, radius):
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
def escape_partial_floyd(f, df, z, c, max_iters, radius):
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


@jit(nopython=True)
def escape_floyd(f, df, z, c, max_iters, radius):
    inv_radius = 1 / (1000 * radius)
    tortoise = f(z, c)
    hare = f(f(z, c), c)

    hare_iter = 2
    while abs(tortoise - hare) >= inv_radius and abs(hare) <= radius and hare_iter < max_iters:
        tortoise = f(tortoise, c)
        hare = f(f(hare, c), c)
        hare_iter += 2

    if hare_iter >= max_iters:
        return np.nan

    if abs(hare) > radius:
        return (hare_iter + 1 - np.log2(np.log2(abs(hare)))) / 256

    preperiod = 0
    tortoise = z

    while abs(tortoise - hare) >= inv_radius and preperiod < max_iters:
        tortoise = f(tortoise, c)
        hare = f(hare, c)
        preperiod += 1

    if preperiod >= max_iters:
        return np.nan

    period = 1
    hare = f(tortoise, c)
    while abs(tortoise - hare) >= inv_radius and period <= max_iters:
        hare = f(hare, c)
        period += 1

    if period > max_iters:
        return np.nan

    return preperiod / (256 *period ) - abs(df(hare, c)) / 256
