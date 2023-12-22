import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as pe

from numba import jit, prange
from typing import Optional

import _algorithms
from _algorithms import naive_period
from _dynamics import RationalMap


class Settings:
    def __init__(self) -> None:
        # Graphics settings
        self.width_pxs: int = 1000
        self.height_pxs: int = 1000
        self.cmap = mpl.colormaps.get_cmap("twilight")
        self.cmap.set_bad(color=self.cmap(0.5))

        # Dynamical parameters
        self.max_iter: int = 256
        self.radius: float = 1000.0
        self.on_sphere: bool = False


class SetView:
    def __init__(
        self,
        d_system: RationalMap,
        coloring,
        ax: mpl.axes.Axes,
        center: complex = 0.0j,
        diam: float = 4.0,
        param: Optional[complex] = 0.0j,
        settings: Optional[Settings] = None,
    ) -> None:
        # Store parameters
        self.d_system = d_system
        self.init_center = center
        self.init_diam = diam
        self.param = param

        self.settings = settings if settings != None else Settings()

        # Coloring algorithm can be supplied directly or by name
        self.coloring = (
            getattr(_algorithms, coloring) if isinstance(coloring, str) else coloring
        )

        # Initialize image
        self.ax = ax
        self.ax.set_axis_off()

        self.img = np.zeros(
            (self.settings.height_pxs, self.settings.width_pxs), dtype=np.float64
        )

        # Lists of angles and rays currently plotted
        self.angles = []
        self.rays = []

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

    def to_complex(self, xdata: float, ydata: float) -> complex:
        h = self.img.shape[0]
        w = self.img.shape[1]
        delta = self.diam / w

        sw = self.center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)
        return sw + complex(xdata, ydata) * delta

    def from_complex(self, z: complex) -> tuple[float, float]:
        h = self.img.shape[0]
        w = self.img.shape[1]
        delta = self.diam / w

        sw = self.center - delta * complex(w, h) / 2 + delta * (0.5 + 0.5j)
        img_coords = (z - sw) / delta

        return img_coords.real, img_coords.imag

    def add_external_ray(self, N: int, D: int) -> None:
        pass

    def update_external_rays(self) -> None:
        for ray in self.rays:
            ray.remove()
        self.rays = []

        angles = self.angles.copy()
        self.angles = []

        for N, D in angles:
            self.add_external_ray(N, D)

    def erase_last_ray(self) -> None:
        if self.rays:
            self.rays.pop().remove()
            self.angles.pop()

    def erase_all_rays(self) -> None:
        while self.rays:
            self.rays.pop().remove()
            self.angles.pop()


class MandelView(SetView):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

        mandel_grid(
            self.coloring,
            self.d_system._f[0],
            self.d_system._f[1],
            self.center,
            self.d_system._crit,
            self.diam,
            self.img,
            self.settings.max_iter,
            self.settings.radius,
        )

        self.plt = self.ax.imshow(
            self.img,
            cmap=self.settings.cmap,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            interpolation_stage="rgba",
        )

        self.ax.set_xlim(-0.5, self.settings.width_pxs - 0.5)
        self.ax.set_ylim(-0.5, self.settings.height_pxs - 0.5)

    def add_external_ray(self, N: int, D: int) -> None:
        self.angles.append((N, D))
        zs = _algorithms.external_ray_mandel(
            self.d_system.f,
            self.d_system.df,
            self.d_system.df_c,
            self.d_system.degree,
            N,
            D,
            self.param,
            self.settings.max_iter,
            self.settings.radius,
        )
        xs, ys = self.from_complex(zs)
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

    def update_plot(self) -> None:
        mandel_grid(
            self.coloring,
            self.d_system._f[0],
            self.d_system._f[1],
            self.center,
            self.d_system._crit,
            self.diam,
            self.img,
            self.settings.max_iter,
            self.settings.radius,
        )

        self.plt.set_data(self.img)
        self.update_external_rays()


class JuliaView(SetView):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

        # Initialize main plot
        julia_grid(
            self.coloring,
            self.d_system._f[0],
            self.d_system._f[1],
            self.center,
            self.param,
            self.diam,
            self.img,
            self.settings.max_iter,
            self.settings.radius,
        )

        self.plt = self.ax.imshow(
            self.img,
            cmap=self.settings.cmap,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            interpolation_stage="rgba",
        )

        self.ax.set_xlim(-0.5, self.settings.width_pxs - 0.5)
        self.ax.set_ylim(-0.5, self.settings.height_pxs - 0.5)

        # Initialize orbits plot
        self.orbit_lenght = 20
        self.orbit_start = None
        (self.orbit_plt,) = self.ax.plot([], [], "ro-", alpha=0.75)

    def add_external_ray(self, N: int, D: int) -> None:
        self.angles.append((N, D))
        zs = _algorithms.external_ray(
            self.d_system.f,
            self.d_system.df,
            self.d_system.degree,
            N,
            D,
            self.param,
            self.settings.max_iter,
            self.settings.radius,
        )
        xs, ys = self.from_complex(zs)
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

    def orbit(self, z: complex, iters: int):
        return _algorithms.orbit(
            self.d_system._f[0],
            self.d_system._f[1],
            z,
            self.param,
            iters,
            self.settings.radius,
        )

    def orbit_proj(self, z: complex, iters: int):
        return _algorithms.orbit_proj(
            self.d_system._f[0],
            self.d_system._f[1],
            z,
            self.param,
            iters,
            self.settings.radius,
        )

    def plot_orbit(self, z: Optional[complex] = None) -> None:
        if z is None and not self.orbit_start is None:
            orbit = self.orbit(self.orbit_start, self.orbit_lenght)
        elif isinstance(z, complex):
            orbit = self.orbit(z, self.orbit_lenght)
            self.orbit_start = z
        else:
            return

        pts = self.from_complex(orbit)
        self.orbit_plt.set_data(pts[0], pts[1])

    def erase_orbit(self):
        self.last_orbit = np.array([], dtype=np.complex128)
        self.orbit_plt.set_data([], [])

    def update_plot(self) -> None:
        julia_grid(
            self.coloring,
            self.d_system._f[0],
            self.d_system._f[1],
            self.center,
            self.param,
            self.diam,
            self.img,
            self.settings.max_iter,
            self.settings.radius,
        )

        self.plt.set_data(self.img)
        self.update_external_rays()
        self.plot_orbit()


@jit(nopython=True, parallel=True)
def mandel_grid(alg, f_u, f_v, center, crit, diam, grid, iters, radius):
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
            grid[m, n] = naive_period(f_u, f_v, crit(c), c, iters, radius)


@jit(nopython=True, parallel=True)
def julia_grid(alg, f_u, f_v, center, param, diam, grid, iters, radius):
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
            grid[m, n] = naive_period(f_u, f_v, z0 + complex(dx, dy), param, iters, radius)
