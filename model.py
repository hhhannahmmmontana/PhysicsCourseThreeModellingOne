import numpy as np
from scipy.sparse import diags, dia_matrix
from scipy.sparse.linalg import eigsh

from constants import *
from epsi import EPsi
from ux import UX

class Model:
    def __init__(self, ux: UX):
        self.ux = ux
        self.dx = ux.dx
        self.U = ux.U
        self.x = ux.x
        self.n = ux.n

    def count_psi(self, levels: int) -> EPsi:
        hamiltonian = self.__count_hamiltonian()
        E, denormalized_psi = eigsh(hamiltonian, k=levels, which="SM")
        idx = np.argsort(E)
        E = E[idx]
        denormalized_psi = denormalized_psi[:, idx]
        psi = self.__normalize_psi(denormalized_psi, levels)
        dpsi = [np.gradient(psi[i], self.dx) for i in range(levels)]

        return EPsi(E, psi, dpsi)

    def __count_hamiltonian(self) -> dia_matrix:
        t = (h_cross ** 2) / (2 * m * (self.dx ** 2))
        diagonals = [
            np.full(self.n - 3, -t),
            2 * t + self.U[1:self.n - 1],
            np.full(self.n - 3, -t)
        ]
        return diags(diagonals, [-1, 0, 1], format="csr")

    def __normalize_psi(self, denormalized_psi: np.array, levels: int) -> np.array:
        psi = []
        for i in range(levels):
            psi_i = denormalized_psi[:, i]
            norm = np.sqrt(np.sum(psi_i ** 2) * self.dx)
            psi_i /= norm

            # Убираем случайность
            if psi_i[0] > psi_i[1]:
                psi_i *= -1

            t_psi = np.zeros(self.n)
            t_psi[1:-1] = psi_i
            psi.append(t_psi)

        return psi