from typing import Callable

import numpy as np

from ux import UX

def count_x(dx: float) -> np.array:
    x = []
    counter = 0
    while counter < 1:
        x.append(counter)
        counter += dx

    return np.array(x)

def count_ux(dx: float, func: Callable[[float], float]) -> UX:
    x = count_x(dx)
    U = np.array([func(x_i) for x_i in x])
    U[0] = float("inf")
    U[-1] = float("inf")
    return UX(U, x)
