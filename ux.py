import numpy as np

class UX:
    def __init__(self, U: np.array, x: np.array):
        self.x = x
        self.U = U
        self.dx = x[1] - x[0]
        self.n = len(x)
