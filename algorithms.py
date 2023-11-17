import numpy as np
from numba import jit
from sympy import lambdify


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
def external_ray(f, df, degree, N, D, c, max_iter, radius):
    pts_count = 10
    R = np.zeros(pts_count, dtype=np.complex128)
    curve = []

    for i in range(pts_count):
        R[i] = radius ** (2 ** (i / pts_count))

    def times_degree(n, d):
        if d % degree == 0:
            d = d // degree
        else:
            n = n * degree
        n = n % d
        return n, d

    z = radius * np.exp(2.0 * np.pi * 1.0j * N / D)

    for n in range(1, max_iter):
        N, D = times_degree(N, D)

        for i in range(pts_count - 1, -1, -1):
            w = R[i] * np.exp(2.0 * np.pi * 1.0j * N / D)

            for newton_iter in range(60):
                z0 = z

                # Iterate function n times
                dz = 1.0 + 0.0j
                for _ in range(n):
                    dz *= df(z, c)
                    z = f(z, c)

                # Newton iteration
                if not (1e-100 < abs(dz) < 1e100):
                    return curve

                adj = (z - w) / dz
                z = z0 - adj

                if abs(adj) < 1e-35:
                    break

            curve.append(z)

    return curve


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
    mult = 1
    for n in range(1, period_multiple + 1):
        tortoise = f(tortoise, c)
        mult *= df(tortoise, c)

        if abs(hare - tortoise) <= inv_radius:
            period = n
            break
    else:
        return np.nan

    # Finds preperiod
    tortoise = z

    for n in range(max_iters):
        if abs(hare - tortoise) <= inv_radius:
            preperiod = n
            break

        tortoise = f(tortoise, c)
        hare = f(hare, c)
    else:
        return np.nan

    # Hare is close to limit point, so eps stands for the distance to limit point
    eps = abs(hare - tortoise)

    return (preperiod / period + 1 - np.log(eps) / np.log(abs(mult))) / 256


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

    return abs(mult) / 4


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

    return np.angle(mult) / (4 * 2 * np.pi)
