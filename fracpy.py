from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
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

        (self.orbit_plt,) = self.ax.plot([], [], "ro-", linewidth=1, alpha=0.75)
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

root = Tk()
root.wm_title("FracPy Mandelbrot")
root.geometry("1500x750")

canvas = FigureCanvasTkAgg(fig_wrap.fig, master=root)
canvas.draw()

# FUNCTIONS THAT UPDATE VIEW

shortcuts = {"z", "x", "r", "s"}


def shortcut_handler(event):
    key = event.key

    if key in shortcuts and event.inaxes != None:
        canvas.get_tk_widget().config(cursor="watch")
        view = julia if julia.ax == event.inaxes else mandel

        if key == "z":  # zooms in
            view.center = (
                view.sw + view.delta * complex(event.xdata, event.ydata) + view.center
            ) / 2
            view.diam /= 2
        elif key == "x":  # zooms out
            view.center = 2 * view.center - (
                view.sw + view.delta * complex(event.xdata, event.ydata)
            )
            view.diam *= 2
        elif key == "r" and view == julia:  # resets center and diam
            view.center = 0.0j
            view.diam = 4.0
        elif key == "r" and view == mandel:  # resets center and diam
            view.center = -0.5 + 0.0j
            view.diam = 4.0
        elif key == "s":
            view.center = view.sw + (view.delta * complex(event.xdata, event.ydata))

        view.update_plot()
        if view == julia and hasattr(julia, "zs"):
            zs = (julia.zs[: julia.z_iter] - julia.sw) / julia.delta
            julia.orbit_plt.set_data(zs.real, zs.imag)
        canvas.draw()
        canvas.get_tk_widget().config(cursor="")

    elif key == "c" and event.inaxes == mandel.ax:
        canvas.get_tk_widget().config(cursor="watch")
        julia.c = mandel.sw + (
            event.xdata * mandel.delta + event.ydata * mandel.delta * 1.0j
        )

        julia.update_plot()

        if hasattr(julia, "zs"):
            delattr(julia, "zs")
            julia.orbit_plt.set_data([], [])

        canvas.draw()
        canvas.get_tk_widget().config(cursor="")

    elif key == "t" and event.inaxes == julia.ax:
        z = julia.sw + (event.xdata * julia.delta + event.ydata * julia.delta * 1.0j)

        julia.zs = sp.orbit(z, julia.c, fig_wrap.max_iter, fig_wrap.esc_radius)
        zs = (julia.zs[: julia.z_iter] - julia.sw) / julia.delta
        julia.orbit_plt.set_data(zs.real, zs.imag)

        canvas.draw()

    elif key == "d" and event.inaxes == julia.ax:
        if hasattr(julia, "zs"):
            delattr(julia, "zs")
            julia.orbit_plt.set_data([], [])

        canvas.draw()


def update_julia_center(event):
    if event.inaxes != None:
        view = julia if julia.ax == event.inaxes else mandel

        pointer = view.sw + (event.xdata * view.delta + event.ydata * view.delta * 1.0j)

        entry_pointer_x.delete(0, END)
        entry_pointer_x.insert(0, pointer.real)
        entry_pointer_y.delete(0, END)
        entry_pointer_y.insert(0, pointer.imag)


def update_color_shift(shift_text):
    fig_wrap.color_shift = np.float64(shift_text)
    mandel.plt.set_data((mandel.img + fig_wrap.color_shift) % 1)
    julia.plt.set_data((julia.img + fig_wrap.color_shift) % 1)
    canvas.draw()
    canvas.get_tk_widget().focus_set()


def update_color_speed():
    canvas.get_tk_widget().config(cursor="watch")
    fig_wrap.color_speed = 1 / (1 << (7 - int(entry_gradient_speed.get())))
    mandel.update_plot()
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().config(cursor="")
    canvas.get_tk_widget().focus_set()


def update_esc_radius(event):
    canvas.get_tk_widget().config(cursor="watch")
    fig_wrap.esc_radius = np.float64(event.widget.get())
    mandel.update_plot()
    julia.update_plot()
    canvas.draw()
    canvas.get_tk_widget().config(cursor="")
    canvas.get_tk_widget().focus_set()


def update_max_iter(event):
    canvas.get_tk_widget().config(cursor="watch")
    fig_wrap.max_iter = np.int64(event.widget.get())
    mandel.update_plot()
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


def save_fig_mandel():
    filetypes = [("All Files", "*.*"), ("PNG", "*.png"), ("JPEG Image", "*.jpg")]

    filename = filedialog.asksaveasfilename(
        initialfile="mandel.png",
        defaultextension=".png",
        filetypes=filetypes,
    )
    extent = mandel.ax.get_window_extent().transformed(
        fig_wrap.fig.dpi_scale_trans.inverted()
    )
    fig_wrap.fig.savefig(filename, bbox_inches=extent)


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
m_file.add_command(label="Save Mandelbrot plot", command=save_fig_mandel)
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
    width=5,
    command=update_color_speed,
)
entry_gradient_speed.insert(0, 0)
entry_gradient_speed.pack(side=LEFT, padx=5)

Label(root, text="Point Iterations:").pack(side=LEFT, padx=5)
entry_z_iter = Spinbox(
    root,
    values=list(range(fig_wrap.max_iter)),
    width=5,
    command=update_z_iter,
)
entry_z_iter.insert(0, 20)
entry_z_iter.pack(side=LEFT, padx=5)
entry_z_iter.bind("<Return>", update_z_iter)

canvas.mpl_connect("key_press_event", shortcut_handler)
canvas.mpl_connect("motion_notify_event", update_julia_center)
mainloop()
