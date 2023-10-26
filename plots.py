import numpy as np

import matplotlib as mpl
from matplotlib.figure import Figure

from dynamics import to_function, escape_grid


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

    def __init__(self, fig_wrap, ax, expr="z^2+C", c_space=False):
        # Initialize all settings with default values
        self.c_space = c_space  # 'mandel' or 'julia'
        self.diam = 4.0  # width of the plot

        if c_space:
            self.init_center = -0.5 + 0.0j
            self.c = 0.0j
        else:
            self.init_center = 0.0j
            self.c = 1.0j

        self.f = to_function(expr)

        self.fig_wrap = fig_wrap
        self.img = np.zeros(
            (self.fig_wrap.height_pxs, self.fig_wrap.width_pxs), dtype=np.float64
        )
        self.ax = ax
        self.ax.set_axis_off()

        escape_grid(
            self.f,
            self.center,
            self.c,
            self.diam,
            self.img,
            fig_wrap.max_iter,
            fig_wrap.esc_radius,
            c_space=self.c_space,
        )

        self.plt = self.ax.imshow(
            (self.fig_wrap.color_speed * self.img) % 1,
            cmap=self.fig_wrap.cmap,
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

    def update_plot(self, all=True):
        """
        Plots the set in self.ax (plot reference is stored in self.plt).
        """
        escape_grid(
            self.f,
            self.center,
            self.c,
            self.diam,
            self.img,
            self.fig_wrap.max_iter,
            self.fig_wrap.esc_radius,
            c_space=self.c_space,
        )

        self.plt.set_data(
            (self.fig_wrap.color_speed * self.img + self.fig_wrap.color_shift) % 1
        )

    def pointer_z(self, xdata, ydata):
        h = self.img.shape[0]
        w = self.img.shape[1]
        delta = self.diam / w

        sw = self.center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)
        return sw + xdata * delta + ydata * delta * 1.0j
