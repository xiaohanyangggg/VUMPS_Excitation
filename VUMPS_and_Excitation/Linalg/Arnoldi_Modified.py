import numpy as np

import tenpy.linalg.np_conserved as npc

from tenpy.linalg.krylov_based import LanczosGroundState, Arnoldi
from tenpy.tools.misc import argsort

class Arnoldi_Modified(Arnoldi):
    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self.hermitian = options.get("hermitian", False)

    def _calc_result_krylov(self, k):
        """calculate ground state of _h_krylov[:k+1, :k+1]"""
        h = self._h_krylov
        if k == 0:
            self.Es[0, 0] = h[0, 0]
            self._result_krylov = np.ones([1, 1], self._dtype_h_krylov)
        else:
            # Diagonalize h
            if self.hermitian:
                E_kr, v_kr = np.linalg.eigh(h[:k + 1, :k + 1])
            elif not self.hermitian:
                E_kr, v_kr = np.linalg.eig(h[:k + 1, :k + 1]) # not hermitian!
            sort = argsort(E_kr, self.which)
            self.Es[k, :k + 1] = E_kr[sort]
            self._result_krylov = v_kr[:, sort]  # ground state of _h_krylov 

