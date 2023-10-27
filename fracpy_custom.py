from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
from numpy import sin, cos, tan, exp, log
from numba import jit, vectorize, prange, int64, float64, complex128

import re
import os

if os.name == "nt":
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(2)

I = 1.0j

# Code that will compute Julia set. (Not the most elegant solution.)
# We will insert the formula for f(z) in the middle of code1 and code2.
# We will run a default version below.

code1 = """
@vectorize([float64(complex128, int64, float64, float64)])
def escape_time(z, max_iters, radius_sqr, gradient_speed):
    i = 0
    for i in range(max_iters):
        z = """

code2 = """
        abs2 = z.real * z.real + z.imag * z.imag
        if abs2 >= radius_sqr:
            return gradient_speed * (i + 1 - np.log2(np.log2(abs2) / 2)) % 1

    return np.nan

@jit(nopython=True, parallel=True)
def julia_plot(z0, delta, image, iters, radius, gradient_speed):
    height = image.shape[0]
    width = image.shape[1]
    radius_sqr = radius**2

    for n in prange(width):
        dx = n * delta
        for m in prange(height):
            dy = m * delta
            color = escape_time(z0 + complex(dx, dy), iters, radius_sqr, gradient_speed)
            image[m, n] = color
"""

code3 = """@jit(nopython=True)
def orbit(z, max_iter, radius):
    iterates = np.zeros(max_iter, dtype=np.complex128)
    for i in range(max_iter):
        iterates[i] = z
        z ="""

code4 = """
        if np.abs(z) >= radius:
            iterates = iterates[:i]
            break

    return iterates"""


@vectorize([float64(complex128, int64, float64, float64)])
def escape_time(z, max_iters, radius_sqr, gradient_speed):
    i = 0
    for i in range(max_iters):
        z = z**2 + 1.0j
        abs2 = z.real * z.real + z.imag * z.imag
        if abs2 >= radius_sqr:
            return gradient_speed * (i + 1 - np.log2(np.log2(abs2) / 2)) % 1

    return np.nan


@jit(nopython=True, parallel=True)
def julia_plot(z0, delta, image, iters, radius, gradient_speed):
    height = image.shape[0]
    width = image.shape[1]
    radius_sqr = radius**2

    for n in prange(width):
        dx = n * delta
        for m in prange(height):
            dy = m * delta
            color = escape_time(z0 + complex(dx, dy), iters, radius_sqr, gradient_speed)
            image[m, n] = color


@jit(nopython=True)
def orbit(z, max_iter, radius):
    iterates = np.zeros(max_iter, dtype=np.complex128)
    for i in range(max_iter):
        iterates[i] = z
        z = z**2 + 1.0j
        if np.abs(z) >= radius:
            iterates = iterates[:i]
            break

    return iterates


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
        self.fig = Figure(figsize=(8, 8), layout="compressed")
        self.cmap = mpl.colormaps.get_cmap("twilight")
        self.cmap.set_bad(color=self.cmap(0.5))
        self.stop_pointer = False

        # Dynamical parameters
        self.max_iter = 256
        self.esc_radius = 100.0


fig_wrap = FigureWrapper()


class JuliaSetView:
    """
    Wrapper for plot of a set in the complex plane (Julia or Mandelbrot).
    """

    def __init__(self, julia_plot, ax) -> None:
        # Initialize all settings with default values
        self._diam = 4.0  # width of the plot
        self.c = 1.0j  # Parameter for julia (z^2+c)
        self._center = 0.0j

        self.delta = self.diam / fig_wrap.diam_pxs
        self.sw = self.center + (self.delta / 2 - self.diam / 2) * (1.0 + 1.0j)

        self.img = np.zeros((fig_wrap.diam_pxs, fig_wrap.diam_pxs), dtype=np.float64)
        self.ax = ax
        self.ax.set_axis_off()

        self.julia_plot = julia_plot

        self.julia_plot(
            self.sw,
            self.delta,
            self.img,
            fig_wrap.max_iter,
            fig_wrap.esc_radius,
            fig_wrap.color_speed,
        )

        self.plt = self.ax.imshow(
            self.img, cmap=fig_wrap.cmap, origin="lower", interpolation_stage="rgba"
        )

        (self.orbit_plt,) = self.ax.plot([], [], "ro-", alpha=0.75)
        self.z_iter = 20

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
        self.julia_plot(
            self.sw,
            self.delta,
            self.img,
            fig_wrap.max_iter,
            fig_wrap.esc_radius,
            fig_wrap.color_speed,
        )
        self.plt.set_data((self.img + fig_wrap.color_shift) % 1)


def close_store(event):
    root.config(cursor="watch")
    # Gets the formula for f(z)
    f = event.widget.get()

    # Makes usual math notation legible by python

    f = re.sub("[0-9.]+I", lambda matchobj: f"{matchobj.group(0)[:-1]}*I", f)
    f = re.sub("[0-9.]+z", lambda matchobj: f"{matchobj.group(0)[:-1]}*z", f)
    f = f.replace("^", "**")
    f = f.replace("Iz", "I*z")
    f = f.replace(")z", ")*z")
    f = f.replace(")I", ")*I")
    f = f.replace("I(", "I*(")
    f = f.replace("z(", "z*(")

    # Defines julia_plot and escape_time using f(z)
    exec(code1 + f + code2, globals())
    exec(code3 + f + code4, globals())

    global julia
    julia = JuliaSetView(julia_plot, fig_wrap.fig.add_subplot(1, 1, 1))
    julia.update_plot()
    canvas.draw()
    root.config(cursor="")
    event.widget.master.destroy()


# FUNCTIONS THAT UPDATE VIEW

shortcuts = {"z", "x", "r", "s"}


def shortcut_handler(event):
    key = event.key

    if key in shortcuts and event.inaxes != None:
        canvas.get_tk_widget().config(cursor="watch")

        if key == "z":  # zooms in
            julia.center = (
                julia.sw
                + julia.delta * complex(event.xdata, event.ydata)
                + julia.center
            ) / 2
            julia.diam /= 2
        elif key == "x":  # zooms out
            julia.center = 2 * julia.center - (
                julia.sw + julia.delta * complex(event.xdata, event.ydata)
            )
            julia.diam *= 2
        elif key == "r":  # resets center and diam
            julia.center = 0.0j
            julia.diam = 4.0
        elif key == "s":
            julia.center = julia.sw + (julia.delta * complex(event.xdata, event.ydata))

        julia.update_plot()
        if hasattr(julia, "zs"):
            zs = (julia.zs[: julia.z_iter] - julia.sw) / julia.delta
            julia.orbit_plt.set_data(zs.real, zs.imag)

        canvas.draw()
        canvas.get_tk_widget().config(cursor="")

    elif key == "t" and event.inaxes != None:
        z = julia.sw + (event.xdata * julia.delta + event.ydata * julia.delta * 1.0j)

        julia.zs = orbit(z, fig_wrap.max_iter, fig_wrap.esc_radius)
        zs = (julia.zs[: julia.z_iter] - julia.sw) / julia.delta
        julia.orbit_plt.set_data(zs.real, zs.imag)

        canvas.draw()

    elif key == "d":
        if hasattr(julia, "zs"):
            delattr(julia, "zs")
            julia.orbit_plt.set_data([], [])
            canvas.draw()

    elif key == "e":
        if fig_wrap.stop_pointer:
            fig_wrap.stop_pointer = False
            global pointer_event
            pointer_event = canvas.mpl_connect(
                "motion_notify_event", update_julia_center
            )
        else:
            fig_wrap.stop_pointer = True
            canvas.mpl_disconnect(pointer_event)


def update_julia_center(event):
    if event.inaxes != None:
        pointer = julia.sw + (
            event.xdata * julia.delta + event.ydata * julia.delta * 1.0j
        )

        entry_pointer_x.delete(0, END)
        entry_pointer_x.insert(0, pointer.real)
        entry_pointer_y.delete(0, END)
        entry_pointer_y.insert(0, pointer.imag)


def update_color_shift(shift_text):
    fig_wrap.color_shift = np.float64(shift_text)
    julia.plt.set_data((julia.img + fig_wrap.color_shift) % 1)
    canvas.draw()
    canvas.get_tk_widget().focus_set()


def update_color_speed():
    canvas.get_tk_widget().config(cursor="watch")
    root.update()
    fig_wrap.color_speed = 1 / (1 << (7 - int(entry_gradient_speed.get())))
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().config(cursor="")
    canvas.get_tk_widget().focus_set()


def update_esc_radius(event):
    canvas.get_tk_widget().config(cursor="watch")
    fig_wrap.esc_radius = np.float64(event.widget.get())
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().config(cursor="")
    canvas.get_tk_widget().focus_set()


def update_max_iter(event):
    canvas.get_tk_widget().config(cursor="watch")
    fig_wrap.max_iter = np.int64(event.widget.get())
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().config(cursor="")
    canvas.get_tk_widget().focus_set()


def update_z_iter(*args):
    canvas.get_tk_widget().config(cursor="watch")
    julia.z_iter = int(entry_z_iter.get())
    if hasattr(julia, "zs"):
        zs = (julia.zs[: julia.z_iter] - julia.sw) / julia.delta
        julia.orbit_plt.set_data(zs.real, zs.imag)
    canvas.draw()
    canvas.get_tk_widget().config(cursor="")
    canvas.get_tk_widget().focus_set()


# MENU FUNCTIONS


def get_formula():
    f_window = Toplevel(root)
    f_window.title("Function for Iteration")
    Label(f_window, text="f(z) =").pack(side=LEFT, padx=5, pady=10)
    f_entry = Entry(f_window, width=60)
    f_entry.pack(side=LEFT, padx=5, pady=10)
    f_entry.bind("<Return>", close_store)


def save_fig_julia():
    filetypes = [("All Files", "*.*"), ("PNG", "*.png"), ("JPEG Image", "*.jpg")]

    filename = filedialog.asksaveasfilename(
        initialfile="julia.png",
        defaultextension=".png",
        filetypes=filetypes,
    )
    extent = julia.ax.get_window_extent().transformed(
        fig_wrap.fig.dpi_scale_trans.inverted()
    )
    fig_wrap.fig.savefig(filename, bbox_inches=extent)


# GUI OBJECTS AND EVENT HANDLERS

root = Tk()
root.wm_title("FracPy")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

julia = JuliaSetView(julia_plot, fig_wrap.fig.add_subplot(1, 1, 1))
julia.update_plot()

canvas = FigureCanvasTkAgg(fig_wrap.fig, master=root)
canvas.get_tk_widget().rowconfigure(0, weight=1)
canvas.get_tk_widget().columnconfigure(0, weight=1)
canvas.get_tk_widget().grid(row=0, column=0, columnspan=8)
canvas.draw()

root.option_add("*tearOff", FALSE)
menu = Menu(root)
root.config(menu=menu)

m_file = Menu(menu)
menu.add_cascade(menu=m_file, label="File")
m_file.add_command(label="Input Julia set function", command=get_formula)
m_file.add_command(label="Save Julia plot", command=save_fig_julia)

options = Frame(root)
options.grid(row=1, column=0)

label_pointer_x = Label(options, text="Pointer x-coordinate:")
label_pointer_x.grid(row=0, column=0, padx=5, pady=5)
entry_pointer_x = Entry(options, width=25)
entry_pointer_x.grid(row=0, column=1, padx=5, pady=5)

label_pointer_y = Label(options, text="Pointer y-coordinate:")
label_pointer_y.grid(row=1, column=0, padx=5, pady=5)
entry_pointer_y = Entry(options, width=25)
entry_pointer_y.grid(row=1, column=1, padx=5, pady=5)

label_esc_radius = Label(options, text="Escape Radius:")
label_esc_radius.grid(row=0, column=2, padx=5, pady=5)
entry_esc_radius = Entry(options, width=10)
entry_esc_radius.insert(0, fig_wrap.esc_radius)
entry_esc_radius.bind("<Return>", update_esc_radius)
entry_esc_radius.grid(row=0, column=3, padx=5, pady=5)

label_max_iter = Label(options, text="Max Iterations:")
label_max_iter.grid(row=1, column=2, padx=5, pady=5)
entry_max_iter = Entry(options, width=10)
entry_max_iter.insert(0, fig_wrap.max_iter)
entry_max_iter.bind("<Return>", update_max_iter)
entry_max_iter.grid(row=1, column=3, padx=5, pady=5)

label_color_shift = Label(options, text="Color Gradient Shift:")
label_color_shift.grid(row=0, column=4, padx=5, pady=5)
color_shift_slider = Scale(
    options, from_=0.0, to=1.0, length=50, command=update_color_shift
)
color_shift_slider.grid(row=0, column=5, padx=5, pady=5)

label_gradient_speed = Label(options, text="Color Gradient Speed:")
label_gradient_speed.grid(row=1, column=4, padx=5, pady=5)
entry_gradient_speed = Spinbox(
    options,
    values=[-2, -1, 0, 1, 2],
    width=5,
    command=update_color_speed,
)
entry_gradient_speed.insert(0, 0)
entry_gradient_speed.grid(row=1, column=5, padx=5, pady=5)

Label(options, text="Point Iterations:").grid(row=0, column=6, padx=5, pady=5)
entry_z_iter = Spinbox(
    options,
    values=list(range(fig_wrap.max_iter)),
    width=5,
    command=update_z_iter,
)
entry_z_iter.insert(0, 20)
entry_z_iter.grid(row=0, column=7, padx=5, pady=5)
entry_z_iter.bind("<Return>", update_z_iter)

canvas.mpl_connect("key_press_event", shortcut_handler)
pointer_event = canvas.mpl_connect("motion_notify_event", update_julia_center)

mainloop()
