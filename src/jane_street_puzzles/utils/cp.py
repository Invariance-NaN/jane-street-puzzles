"""
Some `cpmpy` values and functions with stronger typing.
"""

import cpmpy as cp
from cpmpy.expressions.core import Expression, BoolVal
from typing import Iterable

zero = cp.intvar(0, 0)
false = BoolVal(False)
true = BoolVal(True)

def cp_wrapper(fn, identity):
    def wrapped(xs: Iterable[Expression]) -> Expression:
        xs = list(xs)
        return fn(xs) if len(xs) > 0 else identity
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = f"Wrapper around `cp`'s `{fn.__name__}` that works on empty lists and is typed for only cpmpy `Expressions`."
    return wrapped

sum = cp_wrapper(cp.sum, zero)
any = cp_wrapper(cp.any, false)
all = cp_wrapper(cp.all, true)
