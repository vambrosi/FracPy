import numpy as np
from numba import jit, vectorize, prange, int64, float64, complex128


@vectorize([float64(complex128, complex128, int64, float64, float64)])
def escape_time(z, c, max_iters, radius_sqr, gradient_speed):
    i = 0
    for i in range(max_iters):
        z = z * z + c
        abs2 = z.real * z.real + z.imag * z.imag
        if abs2 >= radius_sqr:
            return gradient_speed * (i + 1 - np.log2(np.log2(abs2) / 2)) % 1

    return np.nan


@jit(nopython=True)
def mandelbrot(z0, delta, image, iters, radius, gradient_speed):
    height = image.shape[0]
    width = image.shape[1]
    radius_sqr = radius**2

    for n in prange(width):
        dx = n * delta
        for m in prange(height):
            dy = m * delta
            color = escape_time(
                0.0j, z0 + complex(dx, dy), iters, radius_sqr, gradient_speed
            )
            image[m, n] = color


@jit(nopython=True)
def julia(z0, c, delta, image, iters, radius, gradient_speed):
    height = image.shape[0]
    width = image.shape[1]
    radius_sqr = radius**2

    for n in prange(width):
        dx = n * delta
        for m in prange(height):
            dy = m * delta
            color = escape_time(
                z0 + complex(dx, dy), c, iters, radius_sqr, gradient_speed
            )
            image[m, n] = color


@jit(nopython=True)
def escape_plot(type, z0, c, delta, image, iters, radius, gradient_speed):
    if type == "mandel":
        mandelbrot(z0, delta, image, iters, radius, gradient_speed)
    elif type == "julia":
        julia(z0, c, delta, image, iters, radius, gradient_speed)
