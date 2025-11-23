import numpy as np

from constants import *
from epsi import EPsi
from ux import UX

class AnalyticCounter:
    def __init__(self, ux: UX):
        self.ux = ux

    def count_epsi(self, levels: int) -> EPsi:
        E = np.empty(levels)
        psi = [np.array(self.ux.n)] * levels
        dpsi = [np.array(self.ux.n)] * levels
        for n in range(levels):
            E[n] = ((h_cross ** 2) * ((n + 1) ** 2) * (pi ** 2)) / (2 * m * a)
            psi[n] = np.sqrt(2 / a) * np.sin((n + 1) * pi * self.ux.x / (a ** 2))
            dpsi[n] = np.sqrt(2 / a) * pi * (n + 1) / a * np.cos((n + 1) * pi * self.ux.x / (a ** 2))

        return EPsi(E, psi, dpsi)