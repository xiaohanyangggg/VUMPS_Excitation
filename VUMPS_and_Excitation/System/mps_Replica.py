import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import *

class MPS_Replica(MPS):
    r"""A matrix product state include different canonical forms and bond tensors for VUMPS.

    Enlagre the MPS class to include left canonical form, center canonical form and bond tensors.
    meth:`get_B` and `set_B` are overrided,
    in order to directly read from and write to `MPS_Replica` obj BL, BR, Ac tensors.

    Added attributes
    ----------------
    _BL : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'matrices' of the MPS. Labels are ``vL, vR, p`` (in any order).
        In Left canonical form.
    _Bc : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'matrices' of the MPS. Labels are ``vL, vR, p`` (in any order).
        In center canonical form.
    _C : list of :class:`~tenpy.linalg.np_conserved.Array`
        The 'matrices' on the MPS bonds. Labels are ``vL, vR`` (in any order).
    """
    def __init__(self, sites, Bs, SVs, bc='finite', form='B', norm=1):
        super().__init__(sites, Bs, SVs, bc, form, norm)
        dtype = np.find_common_type([B.dtype for B in Bs], [])
        self._BL = [B.astype(dtype, copy=True).itranspose(self._B_labels) for B in Bs]
        self._C = []
        self._Bc = []
        # len(_C) = self.L because in VUMPs the first leg is identified with the last one.
        for i in range(len(Bs)):
            legL = self._BL[self._to_valid_index(i-1)].get_leg('vR').conj()
            legR = self._B[self._to_valid_index(i)].get_leg('vL').conj()
            C_i = npc.Array.from_ndarray(np.diag(SVs[i]), legcharges=[legL, legR], labels=['vL', 'vR'])
            self._C.append(C_i)
            self._Bc.append(npc.tensordot(C_i, self._B[i], axes=('vR', 'vL')))

    @classmethod
    def from_MPS(cls, psi):
        obj = cls.__new__(cls)
        obj.sites = list(psi.sites)
        obj.chinfo = psi.sites[0].leg.chinfo
        obj.dtype = psi.dtype
        obj.form = [(0, 1)] * len(psi._B)
        obj.bc = psi.bc
        obj.norm = psi.norm
        obj.grouped = psi.grouped
        obj.segment_boundaries = psi.segment_boundaries
        obj.valid_umps = False # Need to check that AL[n] C[n+1] = AC[n] and C[n] AR[n] = AC[n]
        
        # make copies of 4 types of tensors
        obj._B = [psi.get_B(i, form='B').astype(obj.dtype, copy=True).itranspose(obj._B_labels) for i in range(psi.L)]
        obj._Bc = [psi.get_B(i, form='Th').astype(obj.dtype, copy=True).itranspose(obj._B_labels) for i in range(psi.L)]
        obj._BL = [psi.get_B(i, form='A').astype(obj.dtype, copy=True).itranspose(obj._B_labels) for i in range(psi.L)]
        obj._C = []
        obj._S = []
        for i in range(psi.L):
            C = npc.diag(psi.get_SL(i), obj._BL[i].get_leg('vL'), labels=['vL', 'vR']) # center matrix on the left of site `i`
            obj._C.append(C.astype(obj.dtype, copy=True).itranspose(['vL', 'vR']))
            obj._S.append(psi.get_SL(i))
        obj._S.append(psi.get_SR(psi.L))
        obj._transfermatrix_keep = psi._transfermatrix_keep
        return obj


    def get_B(self, i, form='B', copy=False, cutoff=1e-16, label_p=None):
        i = self._to_valid_index(i)
        # The TeNPy do axis scaling, which means multiple Gamma by some power of Lambda, which will lose precision 
        if self._to_valid_form(form) == self._to_valid_form('B'):
            return super().get_B(i, form, copy, cutoff, label_p)
        elif self._to_valid_form(form) == self._to_valid_form('A'):
            """ try:
                ty = self._BL[i].dtype
                except AttributeError:
                print("The left canonical mps has not be well defined!")
            else: """
            B = self._BL[i]
            if label_p is not None:
                B = self._replace_p_label(B, label_p)
            return B
        elif self._to_valid_form(form) == self._to_valid_form('Th'):
            B = self._Bc[i]
            if label_p is not None:
                B = self._replace_p_label(B, label_p)
            return B
        elif form is None:
            return super().get_B(i, form, copy, cutoff, label_p)
            # Need further check

    def set_B(self, i, B, form='B'):
        i = self._to_valid_index(i)
        if self._to_valid_form(form) == self._to_valid_form('B'):
            super().set_B(i, B, form)
        elif self._to_valid_form(form) == self._to_valid_form('A'):
            self.dtype = np.find_common_type([self.dtype, B.dtype], [])
            self._BL[i] = B.itranspose(self._B_labels)
        elif self._to_valid_form(form) == self._to_valid_form('Th'):
            self.dtype = np.find_common_type([self.dtype, B.dtype], [])
            self._Bc[i] = B.itranspose(self._B_labels)

    def get_C(self, i):
        i = self._to_valid_index(i)
        return self._C[i]

    def set_C(self, i, C):
        i = self._to_valid_index(i)
        self._C[i] = C

    def Calc_Bc(self, i):
        i = self._to_valid_index(i)
        return npc.tensordot(self._C[i], self._B[i], axes=('vR', 'vL'))

    def get_Bc(self, i):
        i = self._to_valid_index(i)
        return self._Bc[i]

    def set_Bc(self, i, Bc):
        i = self._to_valid_index(i)
        self._Bc[i] = Bc
            
    def convert_form_old(self, new_form='B'):
        """ Since meth:`set_B()` is overrided, this method can be discarded. """
        new_forms = self._parse_form(new_form)
        for i, new_form in enumerate(new_forms):
            new_B = super().get_B(i, form=new_form, copy=False)  # calculates the desired form.
            super().set_B(i, new_B, form=new_form)

    def norm_test(self):
        return 0