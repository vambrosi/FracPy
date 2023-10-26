import os
from tkinter import *
from tkinter.ttk import *

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from plots import FigureWrapper, SetView
from dynamics import to_function, parse

if os.name == "nt":
    from ctypes import windll

    windll.shcore.SetProcessDpiAwareness(2)


class App(Tk):
    def __init__(self):
        super().__init__()

        self.wm_title("FracPy")
        self.geometry("1700x900")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.fig_wrap = FigureWrapper()
        self.mandel = SetView(
            self.fig_wrap, self.fig_wrap.fig.add_subplot(1, 2, 1), c_space=True
        )
        self.julia = SetView(self.fig_wrap, self.fig_wrap.fig.add_subplot(1, 2, 2))

        self.m_shortcuts = {"z", "x", "r", "s"}

        self.put_figure()
        self.put_options()
        self.put_menu()

        self.mainloop()

    def put_figure(self):
        self.canvas = FigureCanvasTkAgg(self.fig_wrap.fig, master=self)
        self.canvas.get_tk_widget().rowconfigure(0, weight=1)
        self.canvas.get_tk_widget().columnconfigure(0, weight=1)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=6)
        self.canvas.mpl_connect("key_press_event", self.shortcut_handler)
        self.canvas.mpl_connect("motion_notify_event", self.update_pointer)
        self.canvas.draw()

    def put_options(self):
        self.options = Frame(self)
        self.options.grid(row=1, column=0)

        Label(self.options, text="Pointer x-coordinate:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.pointer_x = Entry(self.options, width=25)
        self.pointer_x.grid(row=0, column=1, padx=5, pady=5)

        Label(self.options, text="Pointer y-coordinate:").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.pointer_y = Entry(self.options, width=25)
        self.pointer_y.grid(row=1, column=1, padx=5, pady=5)

        Label(self.options, text="Escape Radius:").grid(row=0, column=2, padx=5, pady=5)
        self.esc_radius = Entry(self.options, width=10)
        self.esc_radius.insert(0, self.fig_wrap.esc_radius)
        self.esc_radius.bind("<Return>", self.update_esc_radius)
        self.esc_radius.grid(row=0, column=3, padx=5, pady=5)

        Label(self.options, text="Max Iterations:").grid(
            row=1, column=2, padx=5, pady=5
        )
        self.max_iter = Entry(self.options, width=10)
        self.max_iter.insert(0, self.fig_wrap.max_iter)
        self.max_iter.bind("<Return>", self.update_max_iter)
        self.max_iter.grid(row=1, column=3, padx=5, pady=5)

        Label(self.options, text="Color Gradient Shift:").grid(
            row=0, column=4, padx=5, pady=5
        )
        self.color_shift_slider = Scale(
            self.options, from_=0.0, to=1.0, length=50, command=self.update_color_shift
        )
        self.color_shift_slider.grid(row=0, column=5, padx=5, pady=5)

        Label(self.options, text="Color Gradient Speed:").grid(
            row=1, column=4, padx=5, pady=5
        )
        self.gradient_speed = Spinbox(
            self.options,
            values=[-2, -1, 0, 1, 2],
            width=5,
            command=self.update_color_speed,
        )
        self.gradient_speed.insert(0, 0)
        self.gradient_speed.grid(row=1, column=5, padx=5, pady=5)

    def put_menu(self):
        self.option_add("*tearOff", FALSE)
        self.menu = Menu(self)
        self.config(menu=self.menu)

        self.m_input = Menu(self.menu)
        self.menu.add_cascade(menu=self.m_input, label="Input")
        self.m_input.add_command(label="Family of functions", command=self.get_formula)

    def get_formula(self):
        f_window = Toplevel(self)
        f_window.title("Input 1-parameter family of functions")

        Label(f_window, text="f(z, C) =").grid(row=0, column=0, padx=5, pady=5)
        f_entry = Entry(f_window, width=80)
        f_entry.insert(0, "z^2 + C")
        f_entry.grid(row=0, column=1, columnspan=5, padx=5, pady=10)

        Label(f_window, text="Initial C:").grid(row=1, column=0, padx=5, pady=5)
        c_entry = Entry(f_window, width=10)
        c_entry.insert(0, "I")
        c_entry.grid(row=1, column=1, padx=5, pady=5)

        Label(f_window, text="Bifurcation locus center:").grid(
            row=1, column=2, padx=5, pady=10
        )
        mandel_center_entry = Entry(f_window, width=10)
        mandel_center_entry.insert(0, "-0.5")
        mandel_center_entry.grid(row=1, column=3, padx=5, pady=5)

        Label(f_window, text="Filled Julia set center:").grid(
            row=1, column=4, padx=5, pady=10
        )
        julia_center_entry = Entry(f_window, width=10)
        julia_center_entry.insert(0, "0.0")
        julia_center_entry.grid(row=1, column=5, padx=5, pady=5)

        def close_store(*args):
            expr = f_entry.get()
            self.mandel.f = to_function(expr)
            self.mandel.init_center = eval(parse(mandel_center_entry.get()))
            self.mandel.diam = 4.0
            self.julia.f = to_function(expr)
            self.julia.init_center = eval(parse(julia_center_entry.get()))
            self.julia.diam = 4.0
            self.julia.c = eval(parse(c_entry.get()))

            self.config(cursor="watch")
            self.julia.update_plot()
            self.mandel.update_plot()
            self.canvas.draw()
            self.config(cursor="")
            f_window.destroy()

        f_window.bind("<Return>", close_store)

        Button(f_window, text="Plot sets", command=close_store).grid(
            row=2, column=0, columnspan=6, pady=5
        )

    def shortcut_handler(self, event):
        key = event.key

        if key in self.m_shortcuts and event.inaxes != None:
            self.canvas.get_tk_widget().config(cursor="watch")
            view = self.julia if self.julia.ax == event.inaxes else self.mandel
            pointer = view.pointer_z(event.xdata, event.ydata)

            # 's' is not listed because it doesn't change center or diam
            if key == "z":  # zooms in
                view.diam /= 2
                view.center = 0.5 * view.center + 0.5 * pointer
            elif key == "x":  # zooms out
                view.diam *= 2
                view.center = 2 * view.center - pointer
            elif key == "s":  # zooms out
                view.center = pointer
            elif key == "r" and view == self.julia:  # resets center and diam
                view.center = view.init_center
                view.diam = 4.0
            elif key == "r" and view == self.mandel:  # resets center and diam
                view.center = view.init_center
                view.diam = 4.0

            view.update_plot()
            self.canvas.draw()
            self.canvas.get_tk_widget().config(cursor="")

        elif key == "c" and event.inaxes == self.mandel.ax:
            self.canvas.get_tk_widget().config(cursor="watch")
            self.julia.c = self.mandel.pointer_z(event.xdata, event.ydata)

            self.julia.update_plot()
            self.canvas.draw()
            self.canvas.get_tk_widget().config(cursor="")

    def update_pointer(self, event):
        if event.inaxes != None:
            view = self.julia if self.julia.ax == event.inaxes else self.mandel

            pointer = view.pointer_z(event.xdata, event.ydata)

            self.pointer_x.delete(0, END)
            self.pointer_x.insert(0, pointer.real)
            self.pointer_y.delete(0, END)
            self.pointer_y.insert(0, pointer.imag)

    def update_color_shift(self, shift_text):
        self.fig_wrap.color_shift = np.float64(shift_text)
        self.mandel.update_plot(all=False)
        self.julia.update_plot(all=False)
        self.canvas.draw()
        self.canvas.get_tk_widget().focus_set()

    def update_color_speed(self):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.fig_wrap.color_speed = 1 << (2 + int(self.gradient_speed.get()))
        self.mandel.update_plot(all=False)
        self.julia.update_plot(all=False)
        self.canvas.draw()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_esc_radius(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.fig_wrap.esc_radius = np.float64(event.widget.get())
        self.mandel.update_plot()
        self.julia.update_plot()
        self.canvas.draw()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()

    def update_max_iter(self, event):
        self.canvas.get_tk_widget().config(cursor="watch")
        self.fig_wrap.max_iter = np.int64(event.widget.get())
        self.mandel.update_plot()
        self.julia.update_plot()
        self.canvas.draw()
        self.canvas.get_tk_widget().config(cursor="")
        self.canvas.get_tk_widget().focus_set()


if __name__ == "__main__":
    App()
