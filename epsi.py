import numpy as np
from dataclasses import dataclass

@dataclass
class EPsi:
    E: np.array
    psi: np.array
    dpsi: np.array