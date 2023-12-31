import numpy as np
from numba import jit, prange

import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.figure import Figure

import algorithms


@jit(nopython=True)
def color_shift_scale(img, shift, scale):
    return (scale * img + shift) % 1


@jit(nopython=True, parallel=True)
def mandel_grid(alg, f, df, d2f, center, crit, diam, grid, iters, esc_radius):
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

    for n in prange(w):
        dx = n * delta
        for m in prange(h):
            dy = m * delta
            c = z0 + complex(dx, dy)
            grid[m, n] = alg(f, df, crit(c), c, iters, esc_radius)


@jit(nopython=True, parallel=True)
def julia_grid(alg, f, df, d2f, center, param, diam, grid, iters, esc_radius):
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

    for n in prange(w):
        dx = n * delta
        for m in prange(h):
            dy = m * delta
            grid[m, n] = alg(f, df, z0 + complex(dx, dy), param, iters, esc_radius)


class FigureWrapper:
    """
    Stores the settings and the Figure object of the main figure.
    """

    def __init__(self) -> None:
        # Initialize all settings with default values

        # Graphics settings and figure object
        self.width_pxs = 1000
        self.height_pxs = 1000
        self.color_shift = 0.0
        self.color_speed = 4
        self.fig = Figure(figsize=(20, 10), layout="compressed")
        self.cmap = mpl.colormaps.get_cmap("twilight")
        self.cmap.set_bad(color=self.cmap(0.5))

        # Dynamical parameters
        self.max_iter = 256
        self.esc_radius = 100.0


class SetView:
    """
    Wrapper for plot of a set in the complex plane (Julia or Mandelbrot).
    """

    def __init__(
        self,
        fig_wrap,
        ax,
        d_system,
        alg,
        center,
        diam,
        param_space=False,
        init_param=0.0j,
    ):
        # Initialize all settings
        self.d_system = d_system
        self.init_center = center
        self.init_diam = diam
        self.param_space = param_space

        self.fig_wrap = fig_wrap
        self.img = np.zeros(
            (self.fig_wrap.height_pxs, self.fig_wrap.width_pxs), dtype=np.float64
        )
        self.ax = ax
        self.ax.set_axis_off()
        self.alg = getattr(algorithms, alg)

        if param_space:
            mandel_grid(
                self.alg,
                self.d_system.f,
                self.d_system.df,
                self.d_system.d2f,
                self.center,
                self.d_system.crit,
                self.diam,
                self.img,
                fig_wrap.max_iter,
                fig_wrap.esc_radius,
            )
            (self.overlay,) = self.ax.plot([], [], "ro-", alpha=0.75, markersize=3)
        else:
            self.z_iter = 20
            (self.orbit_plt,) = self.ax.plot([], [], "ro-", alpha=0.75)
            self.angles = []
            self.rays = []

            self.param = init_param
            julia_grid(
                self.alg,
                self.d_system.f,
                self.d_system.df,
                self.d_system.d2f,
                self.center,
                self.param,
                self.diam,
                self.img,
                fig_wrap.max_iter,
                fig_wrap.esc_radius,
            )

        self.plt = self.ax.imshow(
            color_shift_scale(
                self.img, self.fig_wrap.color_shift, self.fig_wrap.color_speed
            ),
            cmap=self.fig_wrap.cmap,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            interpolation_stage="rgba",
        )

    @property
    def init_center(self):
        return self._init_center

    @init_center.setter
    def init_center(self, center):
        self._init_center = center
        self.center = center

    @property
    def init_diam(self):
        return self._init_diam

    @init_diam.setter
    def init_diam(self, diam):
        self._init_diam = diam
        self.diam = diam

    def orbit(self, z):
        return algorithms.orbit(
            self.d_system.f,
            z,
            self.param,
            self.fig_wrap.max_iter,
            self.fig_wrap.esc_radius,
        )

    def add_external_ray(self, N, D):
        self.angles.append((N, D))
        zs = algorithms.external_ray(
            self.d_system.f,
            self.d_system.df,
            self.d_system.degree,
            N,
            D,
            self.param,
            self.fig_wrap.max_iter,
            self.fig_wrap.esc_radius,
        )
        xs, ys = self.z_to_img_coords(zs)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        (ray,) = self.ax.plot(
            xs,
            ys,
            c=mpl.colormaps["twilight"](N / D),
            lw=1,
            path_effects=[
                pe.Stroke(linewidth=2, foreground="b"),
                pe.Stroke(linewidth=1.5, foreground="w"),
                pe.Normal(),
            ],
        )
        self.rays.append(ray)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def update_external_rays(self):
        for ray in self.rays:
            ray.remove()
        self.rays = []

        angles = self.angles.copy()
        self.angles = []

        for N, D in angles:
            self.add_external_ray(N, D)

    def erase_last_ray(self):
        if self.rays:
            self.rays.pop().remove()
            self.angles.pop()

    def update_plot(self, all=True):
        """
        Plots the set in self.ax (plot reference is stored in self.plt).
        """
        if all:
            if self.param_space:
                mandel_grid(
                    self.alg,
                    self.d_system.f,
                    self.d_system.df,
                    self.d_system.d2f,
                    self.center,
                    self.d_system.crit,
                    self.diam,
                    self.img,
                    self.fig_wrap.max_iter,
                    self.fig_wrap.esc_radius,
                )
            else:
                julia_grid(
                    self.alg,
                    self.d_system.f,
                    self.d_system.df,
                    self.d_system.d2f,
                    self.center,
                    self.param,
                    self.diam,
                    self.img,
                    self.fig_wrap.max_iter,
                    self.fig_wrap.esc_radius,
                )

        self.plt.set_data(
            color_shift_scale(
                self.img, self.fig_wrap.color_shift, self.fig_wrap.color_speed
            )
        )

    def update_resolution(self):
        self.img = np.zeros(
            (self.fig_wrap.height_pxs, self.fig_wrap.width_pxs), dtype=np.float64
        )

        if self.param_space:
            mandel_grid(
                self.alg,
                self.d_system.f,
                self.d_system.df,
                self.d_system.d2f,
                self.center,
                self.d_system.crit,
                self.diam,
                self.img,
                self.fig_wrap.max_iter,
                self.fig_wrap.esc_radius,
            )
        else:
            julia_grid(
                self.alg,
                self.d_system.f,
                self.d_system.df,
                self.d_system.d2f,
                self.center,
                self.param,
                self.diam,
                self.img,
                self.fig_wrap.max_iter,
                self.fig_wrap.esc_radius,
            )

        self.plt = self.ax.imshow(
            color_shift_scale(
                self.img, self.fig_wrap.color_shift, self.fig_wrap.color_speed
            ),
            cmap=self.fig_wrap.cmap,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            interpolation_stage="rgba",
        )
        self.ax.set_xlim(-0.5, self.fig_wrap.width_pxs -0.5)
        self.ax.set_ylim(-0.5, self.fig_wrap.height_pxs -0.5)

    def img_to_z_coords(self, xdata, ydata):
        h = self.img.shape[0]
        w = self.img.shape[1]
        delta = self.diam / w

        sw = self.center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)
        return sw + xdata * delta + ydata * delta * 1.0j

    def z_to_img_coords(self, z):
        h = self.img.shape[0]
        w = self.img.shape[1]
        delta = self.diam / w

        sw = self.center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)
        img_coords = (z - sw) / delta

        return img_coords.real, img_coords.imag
