import os
from functools import partial

from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from plots import FigureWrapper, SetView
import dynamics

if os.name == "nt":
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(2)


class SetViewer(Tk):
    """
    Creates a window to explore a dynamical system (DSystem).
    """

    def __init__(
        self,
        d_system,
        alg,
        julia_center,
        julia_diam,
        mandel_center,
        mandel_diam,
        init_param=0.0j,
    ):
        super().__init__()

        self.wm_title("FracPy")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.fig_wrap = FigureWrapper()

        self.shortcuts = {
                "z": None,
                "x": None,
                "r": None,
                "s": None,
                "t": None,
                "d": None,
                "left": None,
                "right": None,
                "1": "escape_time",
                "2": "escape_period",
                "3": "escape_naive_period",
                "4": "escape_preperiod",
                "5": "escape_terminal_diff",
                "6": "escape_terminal_diff_arg",
            }

        if d_system.is_family:
            self.geometry("650x420")
            self.mandel = SetView(
                self.fig_wrap,
                self.fig_wrap.fig.add_subplot(1, 2, 1),
                d_system,
                alg,
                mandel_center,
                mandel_diam,
                param_space=True,
            )
            self.julia = SetView(
                self.fig_wrap,
                self.fig_wrap.fig.add_subplot(1, 2, 2),
                d_system,
                alg,
                julia_center,
                julia_diam,
                init_param=init_param,
            )
            self.shortcuts["c"] = None

        else:
            self.geometry("650x700")
            self.julia = SetView(
                self.fig_wrap,
                self.fig_wrap.fig.add_subplot(1, 1, 1),
                d_system,
                alg,
                julia_center,
                julia_diam,
                init_param=init_param,
            )

        self.m_shortcuts = {"z", "x", "r", "s"}

        self.put_figure()
        self.put_options(uses_param=d_system.is_family)
        self.put_menu()

        self.canvas.draw_idle()
        self.mainloop()

    def put_figure(self):
        self.canvas = FigureCanvasTkAgg(self.fig_wrap.fig, master=self)
        self.canvas.get_tk_widget().rowconfigure(0, weight=1)
        self.canvas.get_tk_widget().columnconfigure(0, weight=1)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=6)
        self.canvas.mpl_connect("key_press_event", self.shortcut_handler)
        self.canvas.mpl_connect("motion_notify_event", self.update_pointer)

    def put_options(self, uses_param=True):
        self.options = Frame(self)
        self.options.grid(row=1, column=0)

        # Pointer coordinates
        Label(self.options, text="Pointer:").grid(row=0, column=0, padx=5, sticky="w")
        self.pointer_x = Entry(self.options, width=20)
        self.pointer_x.grid(row=0, column=1, padx=5, pady=5)
        self.pointer_x["state"] = "readonly"

        self.pointer_y = Entry(self.options, width=20)
        self.pointer_y.grid(row=0, column=2, padx=5, pady=5)
        self.pointer_y["state"] = "readonly"

        # Parameter coordinates
        if uses_param:
            Label(self.options, text="Parameter:").grid(
                row=1, column=0, padx=5, sticky="w"
            )
            self.c_x = Entry(self.options, width=20)
            self.c_x.insert(0, self.julia.param.real)
            self.c_x.bind("<Return>", self.update_c)
            self.c_x.grid(row=1, column=1, padx=5, pady=5)

            self.c_y = Entry(self.options, width=20)
            self.c_y.insert(0, self.julia.param.imag)
            self.c_y.bind("<Return>", self.update_c)
            self.c_y.grid(row=1, column=2, padx=5, pady=5)

        # Orbit z0 coordinates
        Label(self.options, text="Orbit start:").grid(
            row=1 + uses_param, column=0, padx=5, sticky="w"
        )
        self.z0_x = Entry(self.options, width=20)
        self.z0_x.bind("<Return>", self.update_z0)
        self.z0_x.grid(row=1 + uses_param, column=1, padx=5, pady=5)

        self.z0_y = Entry(self.options, width=20)
        self.z0_y.bind("<Return>", self.update_z0)
        self.z0_y.grid(row=1 + uses_param, column=2, padx=5, pady=5)

        # Dynamical parameters
        self.esc_radius_label = Label(self.options, text="Esc Radius:")
        self.esc_radius_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.esc_radius = Entry(self.options, width=10)
        self.esc_radius.insert(0, self.fig_wrap.esc_radius)
        self.esc_radius.bind("<Return>", self.update_esc_radius)
        self.esc_radius.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.max_iter_label = Label(self.options, text="Max Iter:")
        self.max_iter_label.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.max_iter = Entry(self.options, width=10)
        self.max_iter.insert(0, self.fig_wrap.max_iter)
        self.max_iter.bind("<Return>", self.update_max_iter)
        self.max_iter.grid(row=1, column=4, padx=5, pady=5, sticky="w")

        # Point parameters
        self.z_iter_label = Label(self.options, text="Point Iter:")
        self.z_iter_label.grid(row=0, column=5, padx=5, sticky="w")
        self.z_iter = Spinbox(
            self.options,
            values=list(range(self.fig_wrap.max_iter)),
            width=5,
            command=self.update_z_iter,
        )
        self.z_iter.insert(0, 20)
        self.z_iter.grid(row=0, column=6, padx=5, pady=5, sticky="w")
        self.z_iter.bind("<Return>", self.update_z_iter)

        # Color parameters
        self.gradient_speed_label = Label(self.options, text="Color Speed:")
        self.gradient_speed_label.grid(row=1, column=5, padx=5, sticky="w")
        self.gradient_speed = Spinbox(
            self.options,
            values=list(range(-2, 6)),
            width=5,
            command=self.update_color_speed,
        )
        self.gradient_speed.insert(0, 0)
        self.gradient_speed.bind("<Return>", self.update_color_speed)
        self.gradient_speed.grid(row=1, column=6, padx=5, pady=5, sticky="w")

    def put_menu(self):
        self.option_add("*tearOff", FALSE)
        self.menu = Menu(self)
        self.config(menu=self.menu)

        self.m_file = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_file, label="File")

        if hasattr(self, "mandel"):
            self.m_file.add_command(
                label="Save Mandelbrot plot", command=self.save_fig_mandel
            )

        self.m_file.add_command(label="Save Julia plot", command=self.save_fig_julia)

        self.m_color = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_color, label="Coloring")

        self.m_color.add_command(
            label="Escape Time (ET)",
            command=partial(self.pick_algorithm, name="escape_time"),
            accelerator="1",
        )
        self.m_color.add_command(
            label="ET + Period",
            command=partial(self.pick_algorithm, name="escape_period"),
            accelerator="2",
        )
        self.m_color.add_command(
            label="ET + Naive Preperiod",
            command=partial(self.pick_algorithm, name="escape_naive_period"),
            accelerator="3",
        )
        self.m_color.add_command(
            label="ET + Preperiod",
            command=partial(self.pick_algorithm, name="escape_preperiod"),
            accelerator="4",
        )
        self.m_color.add_command(
            label="ET + Derivative modulus",
            command=partial(self.pick_algorithm, name="escape_terminal_diff"),
            accelerator="5",
        )
        self.m_color.add_command(
            label="ET + Derivative argument",
            command=partial(self.pick_algorithm, name="escape_terminal_diff_arg"),
            accelerator="6",
        )

    def pick_algorithm(self, name, view=None):
        if view is None:
            self.julia.alg = getattr(dynamics, name)
            if hasattr(self, "mandel"):
                self.mandel.alg = getattr(dynamics, name)
            self.update_plot()

        elif view == self.julia or view == self.mandel:
            view.alg = getattr(dynamics, name)
            view.update_plot()
            self.canvas.draw_idle()

    def shortcut_handler(self, event):
        key = event.key

        if not key in self.shortcuts:
            return None

        if key in self.m_shortcuts and event.inaxes != None:
            self.canvas.get_tk_widget().config(cursor="watch")
            view = self.julia if self.julia.ax == event.inaxes else self.mandel
            pointer = view.img_to_z_coords(event.xdata, event.ydata)

            if hasattr(self.julia, "pts"):
                x, y = self.julia.pts[0][0], self.julia.pts[1][0]
                z = self.julia.img_to_z_coords(x, y)

            # 's' is not listed because it doesn't change center or diam
            if key == "z":  # zooms in
                view.diam /= 2
                view.center = 0.5 * view.center + 0.5 * pointer
            elif key == "x":  # zooms out
                view.diam *= 2
                view.center = 2 * view.center - pointer
            elif key == "s":  # zooms out
                view.center = pointer
            elif key == "r":  # resets center and diam
                view.center = view.init_center
                view.diam = view.init_diam

            if hasattr(self.julia, "pts"):
                self.julia.pts = self.julia.z_to_img_coords(self.julia.orbit(z))
                xs = self.julia.pts[0][: self.julia.z_iter + 1]
                ys = self.julia.pts[1][: self.julia.z_iter + 1]

                self.julia.orbit_plt.set_data(xs, ys)

            view.update_plot()
            self.canvas.draw_idle()
            self.canvas.get_tk_widget().config(cursor="")

        elif key == "c" and event.inaxes == self.mandel.ax:
            self.canvas.get_tk_widget().config(cursor="watch")
            self.julia.param = self.mandel.img_to_z_coords(event.xdata, event.ydata)
            self.julia.update_plot()

            self.c_x.delete(0, END)
            self.c_x.insert(0, self.julia.param.real)

            self.c_y.delete(0, END)
            self.c_y.insert(0, self.julia.param.imag)

            if hasattr(self.julia, "pts"):
                x, y = self.julia.pts[0][0], self.julia.pts[1][0]
                z = self.julia.img_to_z_coords(x, y)

                self.julia.pts = self.julia.z_to_img_coords(self.julia.orbit(z))
                xs = self.julia.pts[0][: self.julia.z_iter + 1]
                ys = self.julia.pts[1][: self.julia.z_iter + 1]

                self.julia.orbit_plt.set_data(xs, ys)

            self.canvas.draw_idle()
            self.canvas.get_tk_widget().config(cursor="")

        elif key == "t" and event.inaxes == self.julia.ax:
            z = self.julia.img_to_z_coords(event.xdata, event.ydata)

            self.julia.pts = self.julia.z_to_img_coords(self.julia.orbit(z))
            xs = self.julia.pts[0][: self.julia.z_iter + 1]
            ys = self.julia.pts[1][: self.julia.z_iter + 1]

            self.julia.orbit_plt.set_data(xs, ys)
            self.canvas.draw_idle()

            self.z0_x.delete(0, END)
            self.z0_x.insert(0, z.real)

            self.z0_y.delete(0, END)
            self.z0_y.insert(0, z.imag)

        elif key == "d":
            if hasattr(self.julia, "pts"):
                delattr(self.julia, "pts")
                self.julia.orbit_plt.set_data([], [])

            self.canvas.draw_idle()

        elif key == "left":
            self.update_color_shift(pressed_left=True)

        elif key == "right":
            self.update_color_shift(pressed_left=False)

        elif key in {"1", "2", "3", "4", "5", "6"}:
            if event.inaxes == self.julia.ax:
                view = self.julia
            elif event.inaxes != None:
                view = self.mandel
            else:
                view = None

            self.pick_algorithm(name=self.shortcuts[key], view=view)

    def update_plot(self, which="both", all=True):
        # In case there is only one plot
        if which == "both" and not hasattr(self, "mandel"):
            which = "julia"

        if which == "both":
            self.mandel.update_plot(all=all)
            self.julia.update_plot(all=all)
            self.canvas.draw_idle()
            self.canvas.get_tk_widget().focus_set()
        elif which == "julia":
            self.julia.update_plot(all=all)
            self.canvas.draw_idle()
            self.canvas.get_tk_widget().focus_set()
        elif which == "mandel":
            self.mandel.update_plot(all=all)
            self.canvas.draw_idle()
            self.canvas.get_tk_widget().focus_set()

    def update_pointer(self, event):
        if event.inaxes != None:
            self.pointer_x["state"] = self.pointer_y["state"] = "active"
            view = self.julia if self.julia.ax == event.inaxes else self.mandel

            pointer = view.img_to_z_coords(event.xdata, event.ydata)

            self.pointer_x.delete(0, END)
            self.pointer_x.insert(0, pointer.real)
            self.pointer_y.delete(0, END)
            self.pointer_y.insert(0, pointer.imag)
            self.pointer_x["state"] = self.pointer_y["state"] = "readonly"

    def update_c(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.julia.param = complex(float(self.c_x.get()), float(self.c_y.get()))
        self.update_plot(which="julia")

        if hasattr(self.julia, "pts"):
            x, y = self.julia.pts[0][0], self.julia.pts[1][0]
            z = self.julia.img_to_z_coords(x, y)

            self.julia.pts = self.julia.z_to_img_coords(self.julia.orbit(z))
            xs = self.julia.pts[0][: self.julia.z_iter + 1]
            ys = self.julia.pts[1][: self.julia.z_iter + 1]

            self.julia.orbit_plt.set_data(xs, ys)
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_z0(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        z = complex(float(self.z0_x.get()), float(self.z0_y.get()))

        self.julia.pts = self.julia.z_to_img_coords(self.julia.orbit(z))
        xs = self.julia.pts[0][: self.julia.z_iter + 1]
        ys = self.julia.pts[1][: self.julia.z_iter + 1]

        self.julia.orbit_plt.set_data(xs, ys)
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_color_shift(self, pressed_left):
        if pressed_left:
            self.fig_wrap.color_shift -= 1 / 32
        else:
            self.fig_wrap.color_shift += 1 / 32
        self.update_plot(which="both", all=False)

    def update_color_speed(self, *args):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.fig_wrap.color_speed = 1 << (2 + int(self.gradient_speed.get()))
        self.update_plot(which="both", all=False)
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_esc_radius(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.fig_wrap.esc_radius = np.float64(event.widget.get())
        self.update_plot(which="both")
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_max_iter(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.fig_wrap.max_iter = np.int64(event.widget.get())
        self.update_plot(which="both")
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_z_iter(self, *args):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.julia.z_iter = np.int64(self.z_iter.get())

        if hasattr(self.julia, "pts"):
            xs = self.julia.pts[0][: self.julia.z_iter + 1]
            ys = self.julia.pts[1][: self.julia.z_iter + 1]

            self.julia.orbit_plt.set_data(xs, ys)

        self.canvas.draw_idle()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def save_fig_mandel(self):
        filetypes = [("PNG", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]

        filename = filedialog.asksaveasfilename(
            initialfile="mandel.png",
            defaultextension=".png",
            filetypes=filetypes,
        )
        extent = self.mandel.ax.get_window_extent().transformed(
            self.fig_wrap.fig.dpi_scale_trans.inverted()
        )
        self.fig_wrap.fig.savefig(filename, bbox_inches=extent)

    def save_fig_julia(self):
        filetypes = [("PNG", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]

        filename = filedialog.asksaveasfilename(
            initialfile="julia.png",
            defaultextension=".png",
            filetypes=filetypes,
        )
        extent = self.julia.ax.get_window_extent().transformed(
            self.fig_wrap.fig.dpi_scale_trans.inverted()
        )
        self.fig_wrap.fig.savefig(filename, bbox_inches=extent)
