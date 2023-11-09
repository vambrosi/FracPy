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
            return (i + 1 - np.log2(np.log2(abs(z)))) / 256

        z = f(z, c)

    return np.nan


@jit(nopython=True)
def escape_naive_period(f, df, z, c, max_iters, radius):
    z2 = f(z, c)
    inv_radius = 1 / (1000 * radius)

    for i in range(max_iters):
        if abs(z) >= radius:
            return (i + 1 - np.log2(np.log2(abs(z)))) / 256

        if abs(z2 - z) <= inv_radius:
            return i / 256

        z = f(z, c)
        z2 = f(f(z2, c), c)

    return np.nan

@jit(nopython=True)
def escape_period(f, df, z, c, max_iters, radius):
    inv_radius = 1 / (1000 * radius)

    # Find the first n such that z_n is close to z_{2n} or it escapes.
    # Algorithm stops if n >= max_iters and returns np.nan.
    tortoise = hare = z

    for n in range(max_iters):
        # If tortoise escaped returns usual coloring
        if abs(tortoise) >= radius:
            return (n + 1 - np.log2(np.log2(abs(tortoise)))) / 256

        tortoise = f(tortoise, c)
        hare = f(f(hare, c), c)

        # If distance is small, exit to compute period and preperiod
        # This is at the end so that at least one iteration is computed.
        if abs(hare - tortoise) <= inv_radius:
            period_multiple = n + 1
            break
    else:
        return np.nan

    # Check to see when they get close again
    for n in range(1, period_multiple + 1):
        tortoise = f(tortoise, c)

        if abs(hare - tortoise) <= inv_radius:
            period = n
            break
    else:
        return np.nan

    return period / 32


@jit(nopython=True)
def escape_preperiod(f, df, z, c, max_iters, radius):
    inv_radius = 1 / (1000 * radius)

    # Find the first n such that z_n is close to z_{2n} or it escapes.
    # Algorithm stops if n >= max_iters and returns np.nan.
    tortoise = hare = z

    for n in range(max_iters):
        # If tortoise escaped returns usual coloring
        if abs(tortoise) >= radius:
            return (n + 1 - np.log2(np.log2(abs(tortoise)))) / 256

        tortoise = f(tortoise, c)
        hare = f(f(hare, c), c)

        # If distance is small, exit to compute period and preperiod
        # This is at the end so that at least one iteration is computed.
        if abs(hare - tortoise) <= inv_radius:
            period_multiple = n + 1
            break
    else:
        return np.nan

    # Check to see when they get close again
    for n in range(1, period_multiple + 1):
        tortoise = f(tortoise, c)

        if abs(hare - tortoise) <= inv_radius:
            period = n
            break
    else:
        return np.nan

    # Finds preperiod
    tortoise = z

    for n in range(period_multiple + 1):
        if abs(hare - tortoise) <= inv_radius:
            preperiod = n
            break

        tortoise = f(tortoise, c)
        hare = f(hare, c)
    else:
        return np.nan

    return (preperiod / period) / 256


@jit(nopython=True)
def escape_terminal_diff(f, df, z, c, max_iters, radius):
    inv_radius = 1 / (1000 * radius)

    # Find the first n such that z_n is close to z_{2n} or it escapes.
    # Algorithm stops if n >= max_iters and returns np.nan.
    tortoise = hare = z

    for n in range(max_iters):
        # If tortoise escaped returns usual coloring
        if abs(tortoise) >= radius:
            return (n + 1 - np.log2(np.log2(abs(tortoise)))) / 256

        tortoise = f(tortoise, c)
        hare = f(f(hare, c), c)

        # If distance is small, exit to compute period derivative
        # This is at the end so that at least one iteration is computed.
        if abs(hare - tortoise) <= inv_radius:
            period_multiple = n + 1
            break
    else:
        return np.nan

    # Find period attracting factor
    mult = 1
    for n in range(1, period_multiple + 1):
        tortoise = f(tortoise, c)
        mult *= df(tortoise, c)

        if abs(hare - tortoise) <= inv_radius:
            break
    else:
        return np.nan

    return  abs(mult) / 4

@jit(nopython=True)
def escape_terminal_diff_arg(f, df, z, c, max_iters, radius):
    inv_radius = 1 / (1000 * radius)

    # Find the first n such that z_n is close to z_{2n} or it escapes.
    # Algorithm stops if n >= max_iters and returns np.nan.
    tortoise = hare = z

    for n in range(max_iters):
        # If tortoise escaped returns usual coloring
        if abs(tortoise) >= radius:
            return (n + 1 - np.log2(np.log2(abs(tortoise)))) / 256

        tortoise = f(tortoise, c)
        hare = f(f(hare, c), c)

        # If distance is small, exit to compute period derivative
        # This is at the end so that at least one iteration is computed.
        if abs(hare - tortoise) <= inv_radius:
            period_multiple = n + 1
            break
    else:
        return np.nan

    # Find period attracting factor
    mult = 1
    for n in range(1, period_multiple + 1):
        tortoise = f(tortoise, c)
        mult *= df(tortoise, c)

        if abs(hare - tortoise) <= inv_radius:
            break
    else:
        return np.nan

    return  np.angle(mult) / (4 * 2 * np.pi)
