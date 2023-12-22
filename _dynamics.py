from sympy import together, cancel, fraction, sympify
from sympy.abc import z, u, v, c

from _algorithms import jit_function


class RationalMap:
    def __init__(self, var=z, expr=z**2 + c, proj_vars=(u, v), crit=0):
        # First questions are: Are there parameters? If so, how many?
        if len(expr.free_symbols) > 2:
            raise Exception("Expression can have only one parameter.")

        # If there are two symbols the one that is not the variable is the parameter
        elif len(expr.free_symbols) == 2:
            param = (expr.free_symbols - {var}).pop()

            # Store rational map in projective coordinates
            self.f = projectivize(var, proj_vars, expr)
            self.df = projectivize(var, proj_vars, expr.diff(var))
            self.crit = crit

            inputs = list(proj_vars) + [param]

            # Store numba versions of the functions
            n, d = fraction(self.f)
            self._f = jit_function(inputs, n), jit_function(inputs, d)
            n, d = fraction(self.df)
            self._df = jit_function(inputs, n), jit_function(inputs, d)
            n, d = fraction(self.crit)
            self._crit = jit_function(param, crit)

            self.is_family = True
            self.var = var
            self.param = param

        # Or else there are no parameters and we use a default value
        else:
            self.f = projectivize(var, c, expr)
            self.df = projectivize(var, c, expr.diff(var))

            inputs = list(proj_vars).append("c")

            n, d = fraction(self.f)
            self._fn = jit_function(inputs, n)
            self._fd = jit_function(inputs, d)

            n, d = fraction(self.df)
            self._dfn = jit_function(inputs, n)
            self._dfd = jit_function(inputs, d)

            self.is_family = False
            self.var = var
            self.param = None


def projectivize(var, proj_vars, rational_expr):
    expr = sympify(rational_expr)
    expr = expr.subs({var: proj_vars[0] / proj_vars[1]})
    expr = together(expr)
    expr = cancel(expr)
    n, d = fraction(expr)

    if not n.is_polynomial() or not d.is_polynomial():
        raise TypeError("Expression is not a rational function.")

    return expr