import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.krylov_based import *

class GMRES(Arnoldi):
    """
    Charge sector of the result depends on the initial vec psi0
    options :: Parameters for building the Krylov space
    """
    """
    H should realize the method `matvec`
    """
    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self._psi0_norm = npc.norm(psi0)
    
    def run(self):
        dim = self._build_krylov()
        h = self._h_krylov[:dim+1, :dim] # dim = k+1, h: (k+2)*(k+1)
        basis_0 = np.zeros(shape=dim+1, dtype = np.complex128)
        basis_0[0] = 1. + 0.j
        b = self._psi0_norm * basis_0
        y, _, _, _ = np.linalg.lstsq(h, b, rcond=None) # (dim,)-ndarray
        x = y[0] * self._cache[0]
        for i in range(1, dim):
            x += y[i] * self._cache[i]
        return x
    
