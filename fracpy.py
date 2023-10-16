import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import setplot as sp


class FigureWrapper:
    """
    Stores the settings and the Figure object of the main figure.
    """

    def __init__(self) -> None:
        # Initialize all settings with default values

        # Graphics settings and figure object
        self.diam_pxs = 1000
        self.color_shift = 0.0
        self.color_speed = 1 / 128
        self.fig = Figure(dpi=100, layout="compressed")
        self.cmap = mpl.colormaps.get_cmap("twilight")
        self.cmap.set_bad(color=self.cmap(0.5))

        # Dynamical parameters
        self.max_iter = 256
        self.esc_radius = 100.0


# There should be only one instance of the class above
fig_wrap = FigureWrapper()


class SetView:
    """
    Wrapper for plot of a set in the complex plane (Julia or Mandelbrot).
    """

    def __init__(self, set_type, ax) -> None:
        # Initialize all settings with default values
        self.set_type = set_type  # 'mandel' or 'julia'
        self._diam = 4.0  # width of the plot
        self.c = 1.0j  # Parameter for julia (z^2+c)

        if self.set_type == "julia":
            self._center = 0.0j
        elif self.set_type == "mandel":
            self._center = -0.5 + 0.0j

        self.delta = self.diam / fig_wrap.diam_pxs
        self.sw = self.center + (self.delta / 2 - self.diam / 2) * (1.0 + 1.0j)

        self.img = np.zeros((fig_wrap.diam_pxs, fig_wrap.diam_pxs), dtype=np.float64)
        self.ax = ax
        self.ax.set_axis_off()

        sp.escape_plot(
            self.set_type,
            self.sw,
            self.c,
            self.delta,
            self.img,
            fig_wrap.max_iter,
            fig_wrap.esc_radius,
            fig_wrap.color_speed,
        )

        self.plt = self.ax.imshow(
            self.img, cmap=fig_wrap.cmap, origin="lower", interpolation_stage="rgba"
        )

    @property
    def diam(self) -> float:
        return self._diam

    @diam.setter
    def diam(self, value):
        self._diam = value
        self.delta = self.diam / fig_wrap.diam_pxs
        self.sw = self.center + (self.delta / 2 - self.diam / 2) * (1.0 + 1.0j)

    @property
    def center(self) -> complex:
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self.sw = self.center + (self.delta / 2 - self.diam / 2) * (1.0 + 1.0j)

    def update_plot(self):
        """
        Plots the set in self.ax (plot reference is stored in self.plt).
        """
        sp.escape_plot(
            self.set_type,
            self.sw,
            self.c,
            self.delta,
            self.img,
            fig_wrap.max_iter,
            fig_wrap.esc_radius,
            fig_wrap.color_speed,
        )
        self.plt.set_data((self.img + fig_wrap.color_shift) % 1)


# INITIALIZE MANDELBROT AND JULIA PLOTS

mandel = SetView("mandel", fig_wrap.fig.add_subplot(1, 2, 1))
julia = SetView("julia", fig_wrap.fig.add_subplot(1, 2, 2))

mandel.update_plot()
julia.update_plot()


# CREATING GUI AND DISPLAYING RESULTS

root = tk.Tk()
root.wm_title("FracPy Mandelbrot")
root.geometry("1400x750")

canvas = FigureCanvasTkAgg(fig_wrap.fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(anchor=tk.CENTER, fill=tk.BOTH, expand=True)


# FUNCTIONS THAT UPDATE VIEW

shortcuts = {"z", "x", "r", "s"}


def shortcut_handler(event):
    key = event.key

    if key in shortcuts and event.inaxes != None:
        view = julia if julia.ax == event.inaxes else mandel
        view.center = view.sw + (
            event.xdata * view.delta + event.ydata * view.delta * 1.0j
        )

        # 's' is not listed because it doesn't change center or diam
        if key == "z":  # zooms in
            view.diam /= 2
        elif key == "x":  # zooms out
            view.diam *= 2
        elif key == "r" and view == julia:  # resets center and diam
            view.center = 0.0j
            view.diam = 4.0
        elif key == "r" and view == mandel:  # resets center and diam
            view.center = -0.5 + 0.0j
            view.diam = 4.0

        view.update_plot()
        canvas.draw()

    elif key == "c" and event.inaxes == mandel.ax:
        julia.c = mandel.sw + (
            event.xdata * mandel.delta + event.ydata * mandel.delta * 1.0j
        )

        julia.update_plot()
        canvas.draw()


def update_julia_center(event):
    if event.inaxes != None:
        view = julia if julia.ax == event.inaxes else mandel

        pointer = view.sw + (event.xdata * view.delta + event.ydata * view.delta * 1.0j)

        entry_pointer_x.delete(0, tk.END)
        entry_pointer_x.insert(0, pointer.real)
        entry_pointer_y.delete(0, tk.END)
        entry_pointer_y.insert(0, pointer.imag)


def update_color_shift(shift_text):
    fig_wrap.color_shift = np.float64(shift_text)
    mandel.plt.set_data((mandel.img + fig_wrap.color_shift) % 1)
    julia.plt.set_data((julia.img + fig_wrap.color_shift) % 1)
    canvas.draw()


def update_color_speed(speed_text):
    fig_wrap.color_speed = np.float64(speed_text)
    mandel.update_plot()
    julia.update_plot()
    canvas.draw()


def update_esc_radius(event):
    fig_wrap.esc_radius = np.float64(event.widget.get())
    mandel.update_plot()
    julia.update_plot()
    canvas.draw()


def update_max_iter(event):
    fig_wrap.max_iter = np.int64(event.widget.get())
    mandel.update_plot()
    julia.update_plot()
    canvas.draw()


# GUI OBJECTS AND EVENT HANDLERS

label_pointer_x = ttk.Label(root, text="Pointer x-coordinate:")
label_pointer_x.pack(side=tk.LEFT, padx=5)
entry_pointer_x = ttk.Entry(root, width=25)
entry_pointer_x.pack(side=tk.LEFT, padx=5)

label_pointer_y = ttk.Label(root, text="Pointer y-coordinate:")
label_pointer_y.pack(side=tk.LEFT, padx=5)
entry_pointer_y = ttk.Entry(root, width=25)
entry_pointer_y.pack(side=tk.LEFT, padx=5)

label_esc_radius = ttk.Label(root, text="Escape Radius:")
label_esc_radius.pack(side=tk.LEFT, padx=5)
entry_esc_radius = ttk.Entry(root, width=15)
entry_esc_radius.insert(0, fig_wrap.esc_radius)
entry_esc_radius.bind("<Return>", update_esc_radius)
entry_esc_radius.pack(side=tk.LEFT, padx=5)

label_max_iter = ttk.Label(root, text="Max Iterations:")
label_max_iter.pack(side=tk.LEFT, padx=5)
entry_max_iter = ttk.Entry(root, width=10)
entry_max_iter.insert(0, fig_wrap.max_iter)
entry_max_iter.bind("<Return>", update_max_iter)
entry_max_iter.pack(side=tk.LEFT, padx=5)

label_color_shift = ttk.Label(root, text="Color Gradient Shift:")
label_color_shift.pack(side=tk.LEFT, padx=5)
color_shift_slider = ttk.Scale(
    root, from_=0.0, to=1.0, length=50, command=update_color_shift
)
color_shift_slider.pack(side=tk.LEFT, padx=5)

label_gradient_speed = ttk.Label(root, text="Color Gradient Speed:")
label_gradient_speed.pack(side=tk.LEFT, padx=5)
gradient_speed = ttk.Scale(
    root,
    from_=1 / 512,
    to=1 / 32,
    value=fig_wrap.color_speed,
    length=50,
    command=update_color_speed,
)
gradient_speed.pack(side=tk.LEFT, padx=20)

canvas.mpl_connect("key_press_event", shortcut_handler)
canvas.mpl_connect("motion_notify_event", update_julia_center)
tk.mainloop()
