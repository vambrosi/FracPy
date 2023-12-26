import numpy as np
from math import sqrt
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
def orbit(f_u, f_v, z, c, max_iter, radius):
    """
    Computes the orbit of a point given a function.
    """
    iterates = np.zeros(max_iter, dtype=np.complex128)
    inv_radius_sqr = 1 / radius**2

    u2, v2 = u1, v1 = z, 1
    for i in range(max_iter):
        if v1 != 0:
            iterates[i] = u1 / v1
        else:
            iterates = iterates[:i]
            break

        temp = f_u(u1, v1, c)
        v1 = f_v(u1, v1, c)
        u1 = temp

        for j in range(2):
            temp = f_u(u2, v2, c)
            v2 = f_v(u2, v2, c)
            u2 = temp

        dist2, u1, v1, u2, v2 = d2_and_normalize(u1, v1, u2, v2)

        if dist2 <= inv_radius_sqr:
            iterates = iterates[:i]
            break

    return iterates


@jit(nopython=True)
def orbit_proj(f_u, f_v, z, c, max_iter, radius):
    """
    Computes the orbit of a point given a function.
    """
    iterates = np.zeros((max_iter, 2), dtype=np.complex128)
    inv_radius_sqr = 1 / radius**2

    u1 = u2 = z
    v1 = v2 = 1
    for i in range(max_iter):
        iterates[i, 0] = u1
        iterates[i, 1] = v1

        temp = f_u(u1, v1, c)
        v1 = f_v(u1, v1, c)
        u1 = temp

        for j in range(2):
            temp = f_u(u2, v2, c)
            v2 = f_v(u2, v2, c)
            u2 = temp

        dist2, u1, v1, u2, v2 = d2_and_normalize(u1, v1, u2, v2)

        if dist2 <= inv_radius_sqr:
            iterates = iterates[:i, :]
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
                    return np.array(curve, dtype=np.complex128)

                adj = (z - w) / dz
                z = z0 - adj

                if abs(adj) < 1e-35:
                    break

            curve.append(z)

    return np.array(curve, dtype=np.complex128)


@jit(nopython=True)
def external_ray_mandel(f, df, df_c, degree, N, D, c, max_iter, radius):
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
                c = z

                # Iterate function n times
                dz = 1.0 + 0.0j
                for _ in range(n):
                    dz *= df(z, c)
                    dz += df_c(z, c)
                    z = f(z, c)

                # Newton iteration
                if not (1e-100 < abs(dz) < 1e100):
                    return np.array(curve, dtype=np.complex128)

                adj = (z - w) / dz
                z = z0 - adj

                if abs(adj) < 1e-35:
                    break

            curve.append(z)

    return np.array(curve, dtype=np.complex128)


@jit(nopython=True)
def naive_period(f_u, f_v, z, c, max_iters, radius):
    u2, v2 = u1, v1 = z, 1
    inv_radius_sqr = 1 / radius**2

    for i in range(max_iters):
        temp = f_u(u1, v1, c)
        v1 = f_v(u1, v1, c)
        u1 = temp

        for j in range(2):
            temp = f_u(u2, v2, c)
            v2 = f_v(u2, v2, c)
            u2 = temp

        dist2, u1, v1, u2, v2 = d2_and_normalize(u1, v1, u2, v2)

        if dist2 <= inv_radius_sqr:
            return (i + 1 - np.log2(-np.log(dist2)/2)) / 128 % 1

    return np.nan

@jit(nopython=True)
def escape_preperiod(f_u, f_v, df_u, z, c, max_iters, radius):
    inv_radius = 1 / (1000 * radius)

    # Find the first n such that z_n is close to z_{2n} or it escapes.
    # Algorithm stops if n >= max_iters and returns np.nan.
    tortoise = hare = z, 1

    for n in range(max_iters):
        tortoise = f_u(*tortoise, c), f_v(*tortoise, c)
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
def abs2(z):
    return z.real * z.real + z.imag * z.imag


@jit(nopython=True)
def d2_and_normalize(u1, v1, u2, v2):
    # Temporary variables
    diff = v2 * u1 - u2 * v1
    dist2 = abs2(diff)

    u1_2 = abs2(u1)
    v1_2 = abs2(v1)
    pt1_2 = u1_2 + v1_2

    u2_2 = abs2(u2)
    v2_2 = abs2(v2)
    pt2_2 = u2_2 + v2_2

    # Normalize points
    u1_norm = u1 / sqrt(pt1_2)
    v1_norm = v1 / sqrt(pt1_2)
    u2_norm = u2 / sqrt(pt2_2)
    v2_norm = v2 / sqrt(pt2_2)

    return dist2 / (pt1_2 + pt2_2), u1_norm, v1_norm, u2_norm, v2_norm
