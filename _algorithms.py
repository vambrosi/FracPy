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

        u1 = f_u(u1, v1, c)
        v1 = f_v(u1, v1, c)

        u2 = f_u(u2, v2, c)
        v2 = f_v(u2, v2, c)
        u2 = f_u(u2, v2, c)
        v2 = f_v(u2, v2, c)

        dist2, pt1_2, pt2_2 = d2(u1, v1, u2, v2)

        if dist2 <= inv_radius_sqr:
            iterates = iterates[:i]
            break

        u1 /= sqrt(pt1_2)
        v1 /= sqrt(pt1_2)
        u2 /= sqrt(pt2_2)
        v2 /= sqrt(pt2_2)

    return iterates


@jit(nopython=True)
def orbit_proj(f_u, f_v, z, c, max_iter, radius):
    """
    Computes the orbit of a point given a function.
    """
    iterates = np.zeros((max_iter, 2), dtype=np.complex128)
    inv_radius_sqr = 1 / radius**2

    u2, v2 = u1, v1 = z, 1
    for i in range(max_iter):
        iterates[i, 0] = u1
        iterates[i, 1] = v1

        u1 = f_u(u1, v1, c)
        v1 = f_v(u1, v1, c)

        u2 = f_u(u2, v2, c)
        v2 = f_v(u2, v2, c)
        u2 = f_u(u2, v2, c)
        v2 = f_v(u2, v2, c)

        dist2, pt1_2, pt2_2 = d2(u1, v1, u2, v2)

        if dist2 <= inv_radius_sqr:
            iterates = iterates[:i, :]
            break

        u1 /= sqrt(pt1_2)
        v1 /= sqrt(pt1_2)
        u2 /= sqrt(pt2_2)
        v2 /= sqrt(pt2_2)

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
        u1 = f_u(u1, v1, c)
        v1 = f_v(u1, v1, c)

        u2 = f_u(u2, v2, c)
        v2 = f_v(u2, v2, c)
        u2 = f_u(u2, v2, c)
        v2 = f_v(u2, v2, c)

        dist2, pt1_2, pt2_2 = d2(u1, v1, u2, v2)

        if dist2 <= inv_radius_sqr:
            return i / 128 % 1

        u1 /= sqrt(pt1_2)
        v1 /= sqrt(pt1_2)
        u2 /= sqrt(pt2_2)
        v2 /= sqrt(pt2_2)

    return np.nan


@jit(nopython=True)
def d2(u1, v1, u2, v2):
    cp = v2 * u1 - u2 * v1
    cp_2 = cp.real * cp.real + cp.imag * cp.imag

    u1_2 = u1.real * u1.real + u1.imag * u1.imag
    v1_2 = v1.real * v1.real + v1.imag * v1.imag
    pt1_2 = u1_2 + v1_2

    u2_2 = u2.real * u2.real + u2.imag * u2.imag
    v2_2 = v2.real * v2.real + v2.imag * v2.imag
    pt2_2 = u2_2 + v2_2

    return cp_2 / (pt1_2 * pt2_2), pt1_2, pt2_2
