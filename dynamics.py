from sympy import degree
from sympy.abc import z, c
from algorithms import jit_function


class DSystem:
    """
    Stores a function f(z) or a one-parameter family of functions f_c(z) as numba functions. The fi
    If it is a one-parameter family, crit encodes a critical point for f_c(z).
    (crit can depend on c and it is used to plot the bifurcation locus).
    """

    def __init__(self, var=z, expr=z**2 + c, crit=0):
        # First questions are: Are there parameters? If so, how many?
        if len(expr.free_symbols) > 2:
            raise Exception("Expression can have only one parameter.")

        # If there are two symbols the one that is not the variable is the parameter
        elif len(expr.free_symbols) == 2:
            param = (expr.free_symbols - {var}).pop()
            self.f = jit_function([var, param], expr)
            self.df = jit_function([var, param], expr.diff(var))
            self.d2f = jit_function([var, param], expr.diff(var, 2))
            self.crit = jit_function(param, crit)
            self.is_family = True
            self.param = param

        # Or else there are no parameters and we use a default value
        else:
            self.f = jit_function([var, "c"], expr)
            self.df = jit_function([var, "c"], expr.diff(var))
            self.d2f = jit_function([var, "c"], expr.diff(var, 2))
            self.is_family = False

        self.var = var
        self.expr = expr
        self.crit_expr = crit

        if expr.is_polynomial():
            self.degree = int(degree(expr, gen=var))

    def view(
        self,
        alg="escape_time",
        julia_center=0.0j,
        julia_diam=4.0,
        mandel_center=0.0j,
        mandel_diam=4.0,
        init_param=0.0j,
    ):
        from viewer import SetViewer

        return SetViewer(
            self,
            alg=alg,
            julia_center=julia_center,
            julia_diam=julia_diam,
            mandel_center=mandel_center,
            mandel_diam=mandel_diam,
            init_param=init_param,
        )
