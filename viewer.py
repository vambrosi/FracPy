import os
from functools import partial

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox, filedialog

import numpy as np
from sympy import sympify
from sympy.abc import z, c
from sympy.core.sympify import SympifyError

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from plots import FigureWrapper, SetView
from dynamics import DSystem
import algorithms

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
        Tk.__init__(self)

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
            "w": partial(self.erase_ray, which="last"),
            "q": partial(self.erase_ray, which="all"),
            "left": None,
            "right": None,
            "ctrl+f": self.pick_function,
            "ctrl+r": self.pick_resolution,
            "e": self.pick_angle,
            "1": {"desc": "Escape Time (ET)", "alg": algorithms.escape_time},
            "2": {"desc": "Stop Time", "alg": algorithms.stop_time},
            "3": {"desc": "ET + Period", "alg": algorithms.escape_period},
            "4": {
                "desc": "ET + Naive Preperiod",
                "alg": algorithms.escape_naive_period,
            },
            "5": {"desc": "ET + Preperiod", "alg": algorithms.escape_preperiod},
            "6": {
                "desc": "ET + Derivative modulus",
                "alg": algorithms.escape_terminal_diff,
            },
            "7": {
                "desc": "ET + Derivative argument",
                "alg": algorithms.escape_terminal_diff_arg,
            },
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

            x, y = self.mandel.z_to_img_coords(self.julia.param)
            self.mandel.overlay.set_data(x, y)

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

        self.update_idletasks()

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
        # Menu bar
        self.option_add("*tearOff", FALSE)
        self.menu = Menu(self)
        self.config(menu=self.menu)

        # File menu
        self.m_file = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_file, label="File")

        if hasattr(self, "mandel"):
            self.m_file.add_command(
                label="Save Mandelbrot plot", command=self.save_fig_mandel
            )

        self.m_file.add_command(label="Save Julia plot", command=self.save_fig_julia)
        self.m_file.add_separator()
        self.m_file.add_command(label="Quit", command=self.destroy)

        # Parameters menu
        self.m_params = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_params, label="Parameters")

        self.m_params.add_command(
            label="Choose function",
            command=self.pick_function,
            accelerator="Ctrl-f",
        )

        self.m_params.add_command(
            label="Choose resolution",
            command=self.pick_resolution,
            accelerator="Ctrl-r",
        )

        # Parameters menu
        self.m_params = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_params, label="Annotations")

        self.m_params.add_command(
            label="Draw external ray",
            command=self.shortcuts["e"],
            accelerator="e",
        )

        self.m_params.add_command(
            label="Erase last external ray",
            command=self.shortcuts["w"],
            accelerator="w",
        )

        self.m_params.add_command(
            label="Erase all external rays",
            command=self.shortcuts["q"],
            accelerator="q",
        )

        # Coloring menu
        self.m_color = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_color, label="Coloring")

        for i in range(1, 8):
            key = str(i)
            color_alg = self.shortcuts[key]

            self.m_color.add_command(
                label=color_alg["desc"],
                command=partial(self.pick_algorithm, alg=color_alg["alg"]),
                accelerator=key,
            )

    def refresh(self):
        self.update_plot()
        self.julia.update_external_rays()

    def erase_ray(self, which="last"):
        if which == "last":
            self.julia.erase_last_ray()
        elif which == "all":
            while self.julia.rays:
                self.julia.erase_last_ray()

        self.canvas.draw_idle()

    def pick_algorithm(self, alg, view=None):
        if view is None:
            self.julia.alg = alg
            if hasattr(self, "mandel"):
                self.mandel.alg = alg
            self.update_plot()

        elif view == self.julia or view == self.mandel:
            view.alg = alg
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

            if hasattr(self, "mandel") and view == self.mandel:
                x, y = self.mandel.z_to_img_coords(self.julia.param)
                self.mandel.overlay.set_data(x, y)

            self.julia.update_external_rays()

            view.update_plot()
            self.canvas.draw_idle()
            self.canvas.get_tk_widget().config(cursor="")

        elif key == "c" and event.inaxes == self.mandel.ax:
            self.julia.param = self.mandel.img_to_z_coords(event.xdata, event.ydata)
            self.julia.update_plot()

            self.c_x.delete(0, END)
            self.c_x.insert(0, self.julia.param.real)

            self.c_y.delete(0, END)
            self.c_y.insert(0, self.julia.param.imag)

            self.update_c()

            self.canvas.draw_idle()

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

        elif key in {"1", "2", "3", "4", "5", "6", "7"}:
            if event.inaxes == self.julia.ax:
                view = self.julia
            elif event.inaxes != None:
                view = self.mandel
            else:
                view = None

            self.pick_algorithm(alg=self.shortcuts[key]["alg"], view=view)

        else:
            if not self.shortcuts[key] is None:
                self.shortcuts[key]()

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

    def update_resolution(self):
        if hasattr(self, "mandel"):
            self.mandel.update_resolution()
        if hasattr(self.julia, "pts"):
            x, y = self.julia.pts[0][0], self.julia.pts[1][0]
            z = self.julia.img_to_z_coords(x, y)

        self.julia.update_resolution()
        if hasattr(self.julia, "pts"):
            self.julia.pts = self.julia.z_to_img_coords(self.julia.orbit(z))
            xs = self.julia.pts[0][: self.julia.z_iter + 1]
            ys = self.julia.pts[1][: self.julia.z_iter + 1]
            self.julia.orbit_plt.set_data(xs, ys)

        self.julia.update_external_rays()
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

    def update_c(self, event=None):
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

        x, y = self.mandel.z_to_img_coords(self.julia.param)
        self.mandel.overlay.set_data(x, y)

        self.julia.update_external_rays()
        self.canvas.draw_idle()
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

    def update_color_speed(self, event=None):
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

    def update_z_iter(self, event=None):
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

    def pick_function(self):
        w_function = Toplevel(self)
        w_function.columnconfigure((1, 3), weight=1)

        Label(
            w_function,
            text="Choose a function or a family of functions (depending on c)",
        ).grid(row=0, column=0, columnspan=4)

        Label(w_function, text="Function:    f(z) =", justify="right").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        function = Entry(w_function, width=60)
        function.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="we")

        Label(w_function, text="Mandelbrot center:", justify="right").grid(
            row=2, column=0, padx=5, sticky="e"
        )
        mandel_center = Entry(w_function)
        mandel_center.grid(row=2, column=1, padx=5, pady=5, sticky="we")

        Label(w_function, text="Mandelbrot diameter:", justify="right").grid(
            row=3, column=0, padx=5, sticky="e"
        )
        mandel_diam = Entry(w_function)
        mandel_diam.grid(row=3, column=1, padx=5, pady=5, sticky="we")

        Label(w_function, text="Critical Point:", justify="right").grid(
            row=4, column=0, padx=5, sticky="e"
        )
        crit = Entry(w_function)
        crit.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        Label(w_function, text="Julia center:", justify="right").grid(
            row=2, column=2, padx=5, sticky="e"
        )
        julia_center = Entry(w_function)
        julia_center.grid(row=2, column=3, padx=5, pady=5, sticky="we")

        Label(w_function, text="Julia diameter:", justify="right").grid(
            row=3, column=2, padx=5, sticky="e"
        )
        julia_diam = Entry(w_function)
        julia_diam.grid(row=3, column=3, padx=5, pady=5, sticky="we")

        Label(w_function, text="Initial Parameter:", justify="right").grid(
            row=4, column=2, padx=5, sticky="e"
        )
        init_param = Entry(w_function)
        init_param.grid(row=4, column=3, padx=5, pady=5, sticky="we")

        # Insert current parameters on entries
        function.insert(0, self.julia.d_system.expr)

        def disp(z):
            return f"{z.real} + {z.imag}*I"

        if hasattr(self, "mandel"):
            mandel_center.insert(0, disp(self.mandel.init_center))
            mandel_diam.insert(0, self.mandel.init_diam)
            crit.insert(0, self.mandel.d_system.crit_expr)
        else:
            mandel_center.insert(0, "-0.5")
            mandel_diam.insert(0, "4.0")
            crit.insert(0, "0.0")

        julia_center.insert(0, disp(self.julia.init_center))
        julia_diam.insert(0, self.julia.init_diam)
        init_param.insert(0, 0.0)

        def close_store(event=None):
            expr = sympify(function.get())
            try:
                crit_expr = sympify(crit.get())
                d_system = DSystem(z, expr, crit=crit_expr)
            except SympifyError:
                d_system = DSystem(z, expr)

            julia_center_expr = complex(sympify(julia_center.get()))
            julia_diam_expr = float(sympify(julia_diam.get()))

            if d_system.is_family:
                mandel_center_expr = complex(sympify(mandel_center.get()))
                mandel_diam_expr = float(sympify(mandel_diam.get()))
                init_param_expr = complex(sympify(init_param.get()))
                d_system.view(
                    julia_center=julia_center_expr,
                    julia_diam=julia_diam_expr,
                    mandel_center=mandel_center_expr,
                    mandel_diam=mandel_diam_expr,
                    init_param=init_param_expr,
                )
            else:
                d_system.view(
                    julia_center=julia_center_expr,
                    julia_diam=julia_diam_expr,
                )
            self.destroy()

        w_function_exit = Button(w_function, text="Plot sets!", command=close_store)
        w_function_exit.grid(row=5, column=0, columnspan=4, padx=5, pady=5)

        w_function.bind("<Return>", close_store)

    def pick_resolution(self):
        w_res = Toplevel(self)
        w_res.columnconfigure((0, 1), weight=1)

        Label(w_res, text="Width (in points):", justify="right").grid(
            row=0, column=0, padx=5, pady=5
        )
        width = Entry(w_res, width=20)
        width.insert(0, self.fig_wrap.width_pxs)
        width.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        Label(w_res, text="Height (in points):", justify="right").grid(
            row=1, column=0, padx=5, pady=5
        )
        height = Entry(w_res, width=20)
        height.insert(0, self.fig_wrap.height_pxs)
        height.grid(row=1, column=1, padx=5, pady=5, sticky="we")

        def close_store(event=None):
            self.fig_wrap.width_pxs = int(width.get())
            self.fig_wrap.height_pxs = int(height.get())
            w_res.destroy()
            self.update_resolution()

        w_res_exit = Button(w_res, text="Plot sets!", command=close_store)
        w_res_exit.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        w_res.bind("<Return>", close_store)

    def pick_angle(self):
        if not hasattr(self.julia.d_system, "degree"):
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
            self.julia.add_external_ray(int(N.get()), int(D.get()))
            w_angle.destroy()
            self.canvas.draw_idle()

        w_angle.bind("<Return>", close_store)
