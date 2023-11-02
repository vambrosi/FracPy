import numpy as np
from numba import jit

import matplotlib as mpl
from matplotlib.figure import Figure

from dynamics import mandel_grid, julia_grid, orbit


@jit(nopython=True)
def color_shift_scale(img, shift, scale):
    return (scale * img + shift) % 1


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
        self, fig_wrap, ax, d_system, center, diam, param_space=False, init_param=0.0j
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
        self.alg = "iter"

        if param_space:
            mandel_grid(
                self.d_system.f,
                self.center,
                self.d_system.crit,
                self.diam,
                self.img,
                fig_wrap.max_iter,
                fig_wrap.esc_radius,
                alg=self.alg,
            )
        else:
            self.z_iter = 20
            (self.orbit_plt,) = self.ax.plot([], [], "ro-", alpha=0.75)

            self.param = init_param
            julia_grid(
                self.d_system.f,
                self.center,
                self.param,
                self.diam,
                self.img,
                fig_wrap.max_iter,
                fig_wrap.esc_radius,
                alg=self.alg,
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
        return orbit(
            self.d_system.f, z, self.param, self.fig_wrap.max_iter, self.fig_wrap.esc_radius
        )

    def update_plot(self, all=True):
        """
        Plots the set in self.ax (plot reference is stored in self.plt).
        """
        if self.param_space:
            mandel_grid(
                self.d_system.f,
                self.center,
                self.d_system.crit,
                self.diam,
                self.img,
                self.fig_wrap.max_iter,
                self.fig_wrap.esc_radius,
                alg=self.alg,
            )
        else:
            julia_grid(
                self.d_system.f,
                self.center,
                self.param,
                self.diam,
                self.img,
                self.fig_wrap.max_iter,
                self.fig_wrap.esc_radius,
                alg=self.alg,
            )

        self.plt.set_data(
            color_shift_scale(
                self.img, self.fig_wrap.color_shift, self.fig_wrap.color_speed
            )
        )

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
