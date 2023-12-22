import os

from tkinter import Tk, Toplevel, messagebox, END
from tkinter.ttk import Frame, Label, Entry

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from _plots import Settings, MandelView, JuliaView
from _dynamics import RationalMap

if os.name == "nt":
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(2)


class SetViewer(Tk):
    def __init__(
        self,
        d_system: RationalMap = RationalMap(),
        coloring="naive_period",
        init_julia_center: complex = 0.0j,
        init_julia_diam: float = 4.0,
        init_mandel_center: complex = 0.0j,
        init_mandel_diam: float = 4.0,
        init_param: complex = 0.0j,
    ):
        Tk.__init__(self)
        self.wm_title("FracPy")

        # Make it resizable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Store reference to dynamical system
        self.d_system = d_system
        self.coloring = coloring

        # Add universal shortcuts
        self.shortcuts = {
            "z": self.zoom_in,
            "x": self.zoom_out,
            "s": self.pan,
            "r": self.reset_view,
            "t": self.plot_orbit,
            "d": self.erase_orbit,
            "e": self.add_ray,
            "w": self.erase_last_ray,
            "q": self.erase_all_rays,
        }

        # Initialize GUI
        # Figure and plot objects
        self.fig = Figure(figsize=(20, 10), layout="compressed")
        self.settings = Settings()

        if self.d_system.is_family:
            self.geometry("650x420")
            self.mandel = MandelView(
                d_system,
                coloring,
                self.fig.add_subplot(1, 2, 1),
                center=init_mandel_center,
                diam=init_mandel_diam,
                settings=self.settings,
            )
            julia_fig = self.fig.add_subplot(1, 2, 2)
            self.shortcuts["c"] = self.update_param
        else:
            self.geometry("650x700")
            julia_fig = self.fig.add_subplot(1, 1, 1)

        self.julia = JuliaView(
            d_system,
            coloring,
            julia_fig,
            center=init_julia_center,
            diam=init_julia_diam,
            param=init_param,
            settings=self.settings,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # Make it resizable
        self.canvas.get_tk_widget().rowconfigure(0, weight=1)
        self.canvas.get_tk_widget().columnconfigure(0, weight=1)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=6)

        # Add event loops
        self.canvas.mpl_connect("motion_notify_event", self.update_pointer)
        self.canvas.mpl_connect("key_press_event", self.shortcut_handler)

        # Add input entries below figure
        self.put_options()

    def put_options(self):
        self.options = Frame(self)
        self.options.grid(row=1, column=0)

        # Pointer coordinates
        Label(self.options, text="Pointer:").grid(row=0, column=0, padx=5, sticky="w")
        self.options.pointer_x = Entry(self.options, width=20)
        self.options.pointer_x.grid(row=0, column=1, padx=5, pady=5)
        self.options.pointer_x["state"] = "readonly"

        self.options.pointer_y = Entry(self.options, width=20)
        self.options.pointer_y.grid(row=0, column=2, padx=5, pady=5)
        self.options.pointer_y["state"] = "readonly"

        # Parameter coordinates
        Label(self.options, text="Parameter:").grid(row=1, column=0, padx=5, sticky="w")
        self.options.param_x = Entry(self.options, width=20)
        self.options.param_x.grid(row=1, column=1, padx=5, pady=5)
        self.options.param_y = Entry(self.options, width=20)
        self.options.param_y.grid(row=1, column=2, padx=5, pady=5)

        if self.d_system.is_family:
            self.options.param_x.insert(0, self.julia.param.real)
            self.options.param_x.bind("<Return>", self.update_param)
            self.options.param_y.insert(0, self.julia.param.imag)
            self.options.param_y.bind("<Return>", self.update_param)
        else:
            self.options.param_x.insert(0, "0.0")
            self.options.param_x["state"] = "readonly"
            self.options.param_y.insert(0, "0.0")
            self.options.param_y["state"] = "readonly"

        # Dynamical parameters
        self.options.radius_label = Label(self.options, text="Radius:")
        self.options.radius_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.options.radius = Entry(self.options, width=10)
        self.options.radius.insert(0, self.settings.radius)
        self.options.radius.bind("<Return>", self.update_radius)
        self.options.radius.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.options.max_iter_label = Label(self.options, text="Max Iter:")
        self.options.max_iter_label.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.options.max_iter = Entry(self.options, width=10)
        self.options.max_iter.insert(0, self.settings.max_iter)
        self.options.max_iter.bind("<Return>", self.update_max_iter)
        self.options.max_iter.grid(row=1, column=4, padx=5, pady=5, sticky="w")

    def shortcut_handler(self, event):
        # Exit if not a shortcut
        if not event.key in self.shortcuts:
            return

        # Actions are listed in a dictionary
        self.canvas.get_tk_widget().config(cursor="watch")
        self.shortcuts[event.key](event)
        self.canvas.get_tk_widget().config(cursor="")

    def zoom_in(self, event):
        # You can only zoom inside a plot
        if event.inaxes == None:
            return

        # Get point
        view = self.julia if self.julia.ax == event.inaxes else self.mandel
        pointer = view.to_complex(event.xdata, event.ydata)

        # Change parameters
        view.diam /= 2
        view.center = 0.5 * view.center + 0.5 * pointer

        # Redraw view
        view.update_plot()
        self.canvas.draw_idle()

    def zoom_out(self, event):
        # You can only zoom inside a plot
        if event.inaxes == None:
            return

        # Get point
        view = self.julia if self.julia.ax == event.inaxes else self.mandel
        pointer = view.to_complex(event.xdata, event.ydata)

        # Change parameters
        view.diam *= 2
        view.center = 2 * view.center - pointer

        # Redraw view
        view.update_plot()
        self.canvas.draw_idle()

    def pan(self, event):
        # You can only pan inside a view
        if event.inaxes == None:
            return

        # Get point
        view = self.julia if self.julia.ax == event.inaxes else self.mandel
        pointer = view.to_complex(event.xdata, event.ydata)

        # Change parameters
        view.center = pointer

        # Redraw view
        view.update_plot()
        self.canvas.draw_idle()

    def reset_view(self, event):
        # Pointer needs to be inside a view
        if event.inaxes == None:
            return

        # Pointer tells which view to reset
        view = self.julia if self.julia.ax == event.inaxes else self.mandel

        # Change parameters
        view.center = view.init_center
        view.diam = view.init_diam

        # Redraw view
        view.update_plot()
        self.canvas.draw_idle()

    def plot_orbit(self, event):
        if event.inaxes != self.julia.ax:
            return

        pointer = self.julia.to_complex(event.xdata, event.ydata)
        self.julia.plot_orbit(pointer)
        self.canvas.draw_idle()

    def erase_orbit(self, event):
        self.julia.erase_orbit()
        self.canvas.draw_idle()

    def add_ray(self, event):
        if event.inaxes is None:
            return

        view = self.julia if self.julia.ax == event.inaxes else self.mandel

        if not hasattr(self.d_system, "degree"):
            messagebox.showinfo("Error", "Only works if f(z) is a polynomial!")
            return

        w_angle = Toplevel(self)
        w_angle.title("Input Angle")
        w_angle.columnconfigure((0, 1, 2), weight=1)

        Label(w_angle, text="Angle (as a fraction of a turn):", justify="right").grid(
            row=0, column=0, padx=5, pady=5
        )
        N = Entry(w_angle)
        N.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        D = Entry(w_angle)
        D.grid(row=0, column=2, padx=5, pady=5, sticky="we")
        N.focus_set()

        def close_store(event=None):
            view.add_external_ray(int(N.get()), int(D.get()))
            w_angle.destroy()
            self.canvas.draw_idle()

        w_angle.bind("<Return>", close_store)

    def erase_last_ray(self, event):
        if event.inaxes is None:
            return

        view = self.julia if self.julia.ax == event.inaxes else self.mandel
        view.erase_last_ray()
        self.canvas.draw_idle()

    def erase_all_rays(self, event):
        if event.inaxes is None:
            return

        view = self.julia if self.julia.ax == event.inaxes else self.mandel
        view.erase_all_rays()
        self.canvas.draw_idle()

    def update_pointer(self, event):
        # Only tracks pointer inside the axes
        if event.inaxes is None:
            return

        self.options.pointer_x["state"] = "active"
        self.options.pointer_y["state"] = "active"

        view = self.julia if self.julia.ax == event.inaxes else self.mandel

        pointer = view.to_complex(event.xdata, event.ydata)
        self.options.pointer_x.delete(0, END)
        self.options.pointer_x.insert(0, pointer.real)
        self.options.pointer_y.delete(0, END)
        self.options.pointer_y.insert(0, pointer.imag)

        self.options.pointer_x["state"] = "readonly"
        self.options.pointer_y["state"] = "readonly"

    def update_param(self, event=None):
        self.canvas.get_tk_widget().config(cursor="watch")

        # If it was triggered by shortcut, gets coordinates
        if hasattr(event, "key") and event.inaxes == self.mandel.ax:
            self.julia.param = self.mandel.to_complex(event.xdata, event.ydata)
            self.options.param_x.delete(0, END)
            self.options.param_x.insert(0, self.julia.param.real)
            self.options.param_y.delete(0, END)
            self.options.param_y.insert(0, self.julia.param.imag)
        # If not get coordinates from GUI input
        else:
            self.julia.param = complex(
                float(self.options.param_x.get()), float(self.options.param_y.get())
            )

        self.julia.update_plot()
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_radius(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.settings.radius = float(event.widget.get())
        self.update_plot()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_max_iter(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.settings.max_iter = int(event.widget.get())
        self.update_plot()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_plot(self) -> None:
        if hasattr(self, "mandel"):
            self.mandel.update_plot()
        self.julia.update_plot()
        self.canvas.draw_idle()
