from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
from numba import jit, vectorize, int64, float64, complex128


# Code that will compute Julia set. (Not the most elegant solution.)
# We will insert the formula for f(z) in the middle of code1 and code2.

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

@jit(nopython=True)
def julia_plot(z0, delta, image, iters, radius, gradient_speed):
    height = image.shape[0]
    width = image.shape[1]
    radius_sqr = radius**2

    for n in range(width):
        dx = n * delta
        for m in range(height):
            dy = m * delta
            color = escape_time(z0 + complex(dx, dy), iters, radius_sqr, gradient_speed)
            image[m, n] = color
"""


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
    # julia_plot will be defined in the exec below.
    exec(code1 + event.widget.get() + code2, globals())

    global julia
    julia = JuliaSetView(julia_plot, fig_wrap.fig.add_subplot(1, 1, 1))
    julia.update_plot()
    canvas.draw()
    event.widget.master.destroy()


# CREATING GUI AND DISPLAYING RESULTS

root = Tk()
root.wm_title("FracPy Mandelbrot")
root.geometry("1400x750")

canvas = FigureCanvasTkAgg(fig_wrap.fig, master=root)
canvas.draw()

# FUNCTIONS THAT UPDATE VIEW

shortcuts = {"z", "x", "r", "s"}


def shortcut_handler(event):
    key = event.key

    if key in shortcuts and event.inaxes != None:
        view = julia
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

        view.update_plot()
        canvas.draw()


def update_julia_center(event):
    if event.inaxes != None:
        view = julia

        pointer = view.sw + (event.xdata * view.delta + event.ydata * view.delta * 1.0j)

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
    fig_wrap.color_speed = 1 / (1 << (7 - int(entry_gradient_speed.get())))
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().focus_set()


def update_esc_radius(event):
    fig_wrap.esc_radius = np.float64(event.widget.get())
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().focus_set()


def update_max_iter(event):
    fig_wrap.max_iter = np.int64(event.widget.get())
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().focus_set()


# MENU FUNCTIONS


def get_formula():
    f_window = Toplevel(root)
    f_window.title("Function for Iteration")
    Label(f_window, text="f(z) =").pack(side=LEFT, padx=5)
    f_entry = Entry(f_window, width=100)
    f_entry.pack(side=LEFT, padx=5)
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

canvas.get_tk_widget().pack(anchor=CENTER, fill=BOTH, expand=True)

root.option_add("*tearOff", FALSE)
menu = Menu(root)
root.config(menu=menu)

m_file = Menu(menu)
menu.add_cascade(menu=m_file, label="File")
m_file.add_command(label="Input Julia set function", command=get_formula)
m_file.add_command(label="Save Julia plot", command=save_fig_julia)

label_pointer_x = Label(root, text="Pointer x-coordinate:")
label_pointer_x.pack(side=LEFT, padx=5)
entry_pointer_x = Entry(root, width=25)
entry_pointer_x.pack(side=LEFT, padx=5)

label_pointer_y = Label(root, text="Pointer y-coordinate:")
label_pointer_y.pack(side=LEFT, padx=5)
entry_pointer_y = Entry(root, width=25)
entry_pointer_y.pack(side=LEFT, padx=5)

label_esc_radius = Label(root, text="Escape Radius:")
label_esc_radius.pack(side=LEFT, padx=5)
entry_esc_radius = Entry(root, width=15)
entry_esc_radius.insert(0, fig_wrap.esc_radius)
entry_esc_radius.bind("<Return>", update_esc_radius)
entry_esc_radius.pack(side=LEFT, padx=5)

label_max_iter = Label(root, text="Max Iterations:")
label_max_iter.pack(side=LEFT, padx=5)
entry_max_iter = Entry(root, width=10)
entry_max_iter.insert(0, fig_wrap.max_iter)
entry_max_iter.bind("<Return>", update_max_iter)
entry_max_iter.pack(side=LEFT, padx=5)

label_color_shift = Label(root, text="Color Gradient Shift:")
label_color_shift.pack(side=LEFT, padx=5)
color_shift_slider = Scale(
    root, from_=0.0, to=1.0, length=50, command=update_color_shift
)
color_shift_slider.pack(side=LEFT, padx=5)

label_gradient_speed = Label(root, text="Color Gradient Speed:")
label_gradient_speed.pack(side=LEFT, padx=5)
entry_gradient_speed = Spinbox(
    root,
    values=[-2, -1, 0, 1, 2],
    width=50,
    command=update_color_speed,
)
entry_gradient_speed.insert(0, 0)
entry_gradient_speed.pack(side=LEFT, padx=20)

canvas.mpl_connect("key_press_event", shortcut_handler)
canvas.mpl_connect("motion_notify_event", update_julia_center)

mainloop()