import numpy as np
import cmath

import tenpy.linalg.np_conserved as npc

__all__ = ['puMPS', 'H_eff_MultiSite', 'N_eff_MultiSite', 'NDressed_H_eff', 'Masked_H_Masked_eff']

class puMPS:
    def __init__(self, _As, _Bs, momentum_int, num_UC) -> None:
        self._As = _As
        self._Bs = _Bs
        self.len_UC = len(_Bs)
        self.num_UC = num_UC
        self.Bs_len = []
        self.Bs_slice = [0]
        self.Bs_legs = []
        # print("Bs[0] has qtotal:{}".format(_Bs[0].qtotal))
        if isinstance(_Bs[0].qtotal, np.ndarray) and _Bs[0].size == 0:
            self.q_sector = None
        elif isinstance(_Bs[0].qtotal, np.ndarray) and _Bs[0].size > 0:
            self.q_sector = _Bs[0].qtotal[0] - _As[0].qtotal[0]
        else:
            raise ValueError("The qtotal of Bs should be either `list` or `np.array`.")
        # The total charge sector of the excitation
        for i in range(self.len_UC):
            B_i_copy = _Bs[i].copy(deep=True)
            if self.q_sector is not None and self._As[i].qtotal[0] != 0:
                self.i_filling = i
                B_i_copy = B_i_copy.gauge_total_charge(axis='p', newqtotal=np.array([self.q_sector]), new_qconj=None)
            B_i_com = B_i_copy.combine_legs(['vL', 'p', 'vR'], qconj=+1)
            self.Bs_len.append(B_i_com.shape[0])
            self.Bs_slice.append(self.Bs_slice[i] + self.Bs_len[i])
            self.Bs_legs.append(B_i_com.get_leg("(vL.p.vR)"))
        
        self.momentum_int = momentum_int
        self.momentum = 2.*cmath.pi * momentum_int / self.num_UC
        B_Vec = self.List_to_Vec(_Bs)
        self.B_Vec = B_Vec
        self.dtype = _Bs[0].dtype

    def List_to_Vec(self, Bs_List):
        B_Vec = Bs_List[0].copy(deep=True)
        if self.q_sector is not None and self.i_filling == 0:
            B_Vec = B_Vec.gauge_total_charge(axis='p', newqtotal=np.array([self.q_sector]), new_qconj=None)
        B_Vec = B_Vec.combine_legs(['vL', 'p', 'vR'], qconj=+1)
        for i in range(1, self.len_UC):
            B_i = Bs_List[i].copy(deep=True)
            if self.q_sector is not None and i == self.i_filling:
                B_i = B_i.gauge_total_charge(axis='p', newqtotal=np.array([self.q_sector]), new_qconj=None)
            B_i = B_i.combine_legs(['vL', 'p', 'vR'], qconj=+1)
            B_Vec = npc.concatenate([B_Vec, B_i], axis=0)
        return B_Vec

    def Vec_to_List(self, Bs_Vec):
        Bs_List = []
        for i in range(self.len_UC):
            B_old_i = Bs_Vec[self.Bs_slice[i] : self.Bs_slice[i+1]]    # Take the slices corresponding to the X tensor at the i site
            if not B_old_i.qtotal:
                qtotal_i = None
            else:
                qtotal_i = B_old_i.qtotal
            B_old_i_NPC = npc.Array.from_ndarray(data_flat=B_old_i.to_ndarray(), legcharges=[self.Bs_legs[i]],
                                              dtype=self.dtype, qtotal=qtotal_i, labels=["(vL.p.vR)"]).split_legs()
            if self.q_sector is not None and i == self.i_filling:
                B_old_i_NPC = B_old_i_NPC.gauge_total_charge(axis='p', newqtotal=np.array([self._As[i].qtotal[0]+self.q_sector]), new_qconj=None)
            Bs_List.append(B_old_i_NPC)
            # print("B_old_i has legcharge information {} after splitting.".format(B_old_i_NPC))
        return Bs_List
    
class H_eff_MultiSite:
    def __init__(self, psi, H_MPO, Lx, Ly, x=0) -> None:
        self._Ws = H_MPO._W
        # Since TeNPy uses OBC with LR interaction between the boundary sites to represent
        # the PBC model, we need to fix the outerest MPO legs to exploit the PBC Hamiltonian.
        # The solution is to project the first and last legs of the MPO.
        if self._Ws[0].get_leg('wL').ind_len > 1 or self._Ws[-1].get_leg('wR').ind_len > 1:
            IdL = H_MPO.IdL[0]
            IdR = H_MPO.IdR[-1]
            self._Ws[0].iproject(IdL, axes='wL')
            self._Ws[-1].iproject(IdR, axes='wR')
        self.psi = psi
        self.num_UC = Lx
        self.len_UC = Ly
        self.len_tot = Lx*Ly
        self.dtype = psi.dtype
        self.x = x
        # Initialize the environments for the onsite terms.
        # And the list of environments for center terms.
        self.env_onsite_0 = self.init_onsite_env(x)
        self.env_onsite_c = self.init_onsite_env_c(x)

    def matvec(self, vec):
        """
        Default to calculate the effective Hamiltonian for the 0th unit cell
        i.e. x = 0
        """
        Labels = vec.get_leg_labels()
        x = self.x
        vec_List = self.psi.Vec_to_List(vec)
        B_env = self.get_B_env(x, vec_List)
        B_new_List = []
        for y in range(0, self.len_UC):
            B_onsite = self.Onsite_term(x, y, Bs_list=vec_List)
            B_offsite = self.Offsite_term(x, y, Bs_list=vec_List, B_env=B_env)
            B_y = B_onsite + B_offsite
            B_new_List.append(B_y)
        vec_new = self.psi.List_to_Vec(B_new_List)
        return vec_new.itranspose(Labels)
    
    def Onsite_term(self, x, y, Bs_list):
        Labels = Bs_list[y].get_leg_labels()
        x = self.x_to_valid(x)
        i_W_xy = self.x_y_to_W_index(x, y)
        
        B_c = npc.tensordot(self.env_onsite_c[y], Bs_list[y], axes=[['vR', 'vL'], ['vL', 'vR']])
        B_c = npc.tensordot(B_c, self._Ws[i_W_xy], axes=[['wR', 'p', 'wL'], ['wL', 'p*', 'wR']])
        B_c.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        return B_c.itranspose(Labels)
    
    def Offsite_term(self, x, y, Bs_list, B_env):
        Labels = self.psi._As[0].get_leg_labels()
        x = self.x_to_valid(x)
        i_W_x_y = self.x_y_to_W_index(x, y)
        """
        The terms in which the up B tensor doesn't coincide with the down B tensor.
        """
        env_site = self.env_onsite_0.copy(deep=True)
        env_B = B_env.copy(deep=True)
        for y_i in range(0, y):
            W_i = self._Ws[self.x_y_to_W_index(x, y_i)]
            env_B = npc.tensordot(env_B, self.psi._As[y_i], axes=[["vR"], ["vL"]])
            env_B = npc.tensordot(env_B, W_i, axes=[['wR', 'p'], ['wL', 'p*']])
            env_B = npc.tensordot(env_B, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            
            env_add = npc.tensordot(env_site, Bs_list[y_i], axes=[["vR"], ["vL"]])
            env_add = npc.tensordot(env_add, W_i, axes=[['wR', 'p'], ['wL', 'p*']])
            env_add = npc.tensordot(env_add, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            env_B += env_add

            env_site = npc.tensordot(env_site, self.psi._As[y_i], axes=[["vR"], ["vL"]])
            env_site = npc.tensordot(env_site, W_i, axes=[['wR', 'p'], ['wL', 'p*']])
            env_site = npc.tensordot(env_site, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

        for y_i in range(self.len_UC-1, y, -1):
            W_i = self._Ws[self.x_y_to_W_index(x, y_i)]
            env_B = npc.tensordot(env_B, self.psi._As[y_i], axes=[["vL"], ["vR"]])
            env_B = npc.tensordot(env_B, W_i, axes=[['wL', 'p'], ['wR', 'p*']])
            env_B = npc.tensordot(env_B, self.psi._As[y_i].conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            
            env_add = npc.tensordot(env_site, Bs_list[y_i], axes=[["vL"], ["vR"]])
            env_add = npc.tensordot(env_add, W_i, axes=[['wL', 'p'], ['wR', 'p*']])
            env_add = npc.tensordot(env_add, self.psi._As[y_i].conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

            env_B += env_add

            env_site = npc.tensordot(env_site, self.psi._As[y_i], axes=[["vL"], ["vR"]])
            env_site = npc.tensordot(env_site, W_i, axes=[['wL', 'p'], ['wR', 'p*']])
            env_site = npc.tensordot(env_site, self.psi._As[y_i].conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

        B_offsite = npc.tensordot(env_B, self.psi._As[y], axes=[['vR', 'vL'], ['vL', 'vR']])
        B_offsite = npc.tensordot(B_offsite, self._Ws[i_W_x_y], axes=[['wR', 'p', 'wL'], ['wL', 'p*', 'wR']])
        B_offsite.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        return B_offsite.itranspose(Labels)

    def get_B_env(self, x, Bs_list):
        """
        Get the environment with B tensors around the x unit cell.
        """
        Id_env_0 = self.generate_initial_eye(x)
        env_site = Id_env_0
        x_start = self.x_to_valid(x+1)
        envB = self.Apply_TB_from_Right(vec=Id_env_0, x=x_start, Bs_list=Bs_list) * cmath.exp(1.j*self.psi.momentum)
        for i in range(2, self.num_UC):
            x_i = self.x_to_valid(x+i)
            envB = self.Apply_T(vec=envB, x=x_i, transpose=True)
            env_site = self.Apply_T(vec=env_site, x=self.x_to_valid(x+i-1), transpose=True) * cmath.exp(1.j*self.psi.momentum)
            env_add = self.Apply_TB_from_Right(vec=env_site, x=x_i, Bs_list=Bs_list) * cmath.exp(1.j*self.psi.momentum)

            envB += env_add
        return envB
    
    def init_onsite_env(self, x):
        """
        Generate the environment for the case in which two B unit cells overlap.
        i.e. return (Transfer_Matrix)^(self.num_UC-1)
        """

        x = self.x_to_valid(x)
        env_onsite = self.generate_initial_eye(x)
        
        for i in range(1, self.num_UC):
            x_i =self.x_to_valid(x+i)
            env_onsite = self.Apply_T(env_onsite, x_i, transpose=True)
        return env_onsite
    
    def init_onsite_env_c(self, x):
        env_onsite_c = []
        for y_i in range(0, self.len_UC):
            env_y = self.env_onsite_0.copy(deep=True)
            for y_r in range(self.len_UC-1, y_i, -1):
                i_W = self.x_y_to_W_index(x, y_r)
                env_y = npc.tensordot(env_y, self.psi._As[y_r], axes=[['vL'], ['vR']])
                env_y = npc.tensordot(env_y, self._Ws[i_W], axes=[['wL', 'p'], ['wR', 'p*']])
                env_y = npc.tensordot(env_y, self.psi._As[y_r].conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])
            for y_l in range(0, y_i, +1):
                i_W = self.x_y_to_W_index(x, y_l)
                env_y = npc.tensordot(env_y, self.psi._As[y_l], axes=[['vR'], ['vL']])
                env_y = npc.tensordot(env_y, self._Ws[i_W], axes=[['wR', 'p'], ['wL', 'p*']])
                env_y = npc.tensordot(env_y, self.psi._As[y_l].conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
            env_onsite_c.append(env_y)
        return env_onsite_c

    def Apply_T(self, vec, x, transpose=False):
        """
        Apply the x unit cell transfer matrix on x-1 uc or x+1 uc
        """
        Labels = vec.get_leg_labels()
        x = self.x_to_valid(x)
        if not transpose:
            for y_i in range(self.len_UC-1, -1, -1):
                A_u = self.psi._As[y_i]
                A_d = self.psi._As[y_i]
                W = self._Ws[self.x_y_to_W_index(x, y_i)]
                vec = npc.tensordot(vec, A_u, axes=["vL", "vR"])
                vec = npc.tensordot(vec, W, axes=[["wL", "p"], ["wR", "p*"]])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        elif transpose:
            for y_i in range(0, self.len_UC, +1):
                A_u = self.psi._As[y_i]
                A_d = self.psi._As[y_i]
                W = self._Ws[self.x_y_to_W_index(x, y_i)]
                vec = npc.tensordot(vec, A_u, axes=["vR", "vL"])
                vec = npc.tensordot(vec, W, axes=[["wR", "p"], ["wL", "p*"]])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        return vec.itranspose(Labels)
    
    def Apply_TB_from_Right(self, vec, x, Bs_list):
        Labels = vec.get_leg_labels()
        x = self.x_to_valid(x)
        i_W_start = self.x_y_to_W_index(x, 0)
        Lp_site = vec.copy(deep=True)
        L_B = npc.tensordot(vec, Bs_list[0], axes=[["vR"], ["vL"]])
        L_B = npc.tensordot(L_B, self._Ws[i_W_start], axes=[["p", "wR"], ["p*", "wL"]])
        L_B = npc.tensordot(L_B, self.psi._As[0].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

        for y_i in range(1, self.len_UC, +1):
            i_W = self.x_y_to_W_index(x, y_i)
            L_B = npc.tensordot(L_B, self.psi._As[y_i], axes=[["vR"], ["vL"]])
            L_B = npc.tensordot(L_B, self._Ws[i_W], axes=[["p", "wR"], ["p*", "wL"]])
            L_B = npc.tensordot(L_B, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            
            Lp_site = npc.tensordot(Lp_site, self.psi._As[y_i-1], axes=[["vR"], ["vL"]])
            Lp_site = npc.tensordot(Lp_site, self._Ws[self.x_y_to_W_index(x, y_i-1)], axes=[["p", "wR"], ["p*", "wL"]])
            Lp_site = npc.tensordot(Lp_site, self.psi._As[y_i-1].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            L_B_add = npc.tensordot(Lp_site, Bs_list[y_i], axes=[["vR"], ["vL"]])
            L_B_add = npc.tensordot(L_B_add, self._Ws[i_W], axes=[["p", "wR"], ["p*", "wL"]])
            L_B_add = npc.tensordot(L_B_add, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            L_B += L_B_add
        return L_B.itranspose(Labels)

    def generate_initial_eye(self, x):
        """
        Generate the three line Identity at the end of x unit cell
        """
        leg_u = self.psi._As[0].get_leg('vL')
        leg_d = self.psi._As[0].conj().get_leg('vL*')
        i_W = self.x_y_to_W_index(x+1, 0)
        leg_m = self._Ws[i_W].get_leg('wL')
        eye_u = npc.diag(1., leg=leg_u, dtype=self.dtype, labels=['vL', 'vR'])
        eye_d = npc.diag(1., leg=leg_d, dtype=self.dtype, labels=['vL*', 'vR*'])
        eye_m = npc.diag(1., leg=leg_m, dtype=self.dtype, labels=['wL', 'wR'])
        eye_init = npc.outer(eye_u, eye_m)
        eye_init = npc.outer(eye_init, eye_d)
        return eye_init

    def Cal_GS_Energy(self):
        E = self.generate_initial_eye(0)
        for x_i in range(1, self.num_UC+1):
            E = self.Apply_T(E, x_i, transpose=True)
        E = npc.trace(E, leg1='vL', leg2='vR')
        E = npc.trace(E, leg1='wL', leg2='wR')
        E = npc.trace(E, leg1='vL*', leg2='vR*')
        
        return E
    
    def x_y_to_W_index(self, x, y):
        i_W = (x*self.len_UC + y) % self.len_tot
        return i_W
    
    def x_to_valid(self, x):
        return x % self.num_UC
    
class N_eff_MultiSite(H_eff_MultiSite):
    def __init__(self, psi, H_MPO, Lx, Ly) -> None:
        super().__init__(psi, H_MPO, Lx, Ly)
        # self.Normalize_As()

    def matvec(self, vec):
        vec_List = self.psi.Vec_to_List(vec)
        B_env = self.get_B_env(vec_List)
        B_new_List = []
        for y in range(0, self.len_UC):
            B_onsite = self.Onsite_term(y, Bs_list=vec_List)
            B_offsite = self.Offsite_term(y, Bs_list=vec_List, B_env=B_env)
            B_y = B_onsite + B_offsite
            B_new_List.append(B_y)
        vec_new = self.psi.List_to_Vec(B_new_List)
        vec_new = vec_new
        return vec_new
    
    def Onsite_term(self, y, Bs_list):
        Labels = Bs_list[y].get_leg_labels()
        B_onsite = self.env_onsite_c[y]
        B_onsite = npc.tensordot(B_onsite, Bs_list[y], axes=[['vR', 'vL'], ['vL', 'vR']])
        B_onsite.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        return B_onsite.itranspose(Labels)
    
    def Offsite_term(self, y, Bs_list, B_env):
        """
        The terms in which the up B tensor doesn't coincide with the down B tensor.
        """
        Labels = Bs_list[0].get_leg_labels()
        env_site = self.env_onsite_0.copy(deep=True)
        env_B = B_env.copy(deep=True)

        for y_i in range(0, y):
            env_B = npc.tensordot(env_B, self.psi._As[y_i], axes=[["vR"], ["vL"]])
            env_B = npc.tensordot(env_B, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            
            env_add = npc.tensordot(env_site, Bs_list[y_i], axes=[["vR"], ["vL"]])
            env_add = npc.tensordot(env_add, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            env_B += env_add

            env_site = npc.tensordot(env_site, self.psi._As[y_i], axes=[["vR"], ["vL"]])
            env_site = npc.tensordot(env_site, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

        for y_i in range(self.len_UC-1, y, -1):
            env_B = npc.tensordot(env_B, self.psi._As[y_i], axes=[["vL"], ["vR"]])
            env_B = npc.tensordot(env_B, self.psi._As[y_i].conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            
            env_add = npc.tensordot(env_site, Bs_list[y_i], axes=[["vL"], ["vR"]])
            env_add = npc.tensordot(env_add, self.psi._As[y_i].conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

            env_B += env_add

            env_site = npc.tensordot(env_site, self.psi._As[y_i], axes=[["vL"], ["vR"]])
            env_site = npc.tensordot(env_site, self.psi._As[y_i].conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

        B_offsite = npc.tensordot(env_B, self.psi._As[y], axes=[['vR', 'vL'], ['vL', 'vR']])
        B_offsite.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
        return B_offsite.itranspose(Labels)
    
    def get_B_env(self, Bs_list):
        Id_env_0 = self.generate_initial_eye()
        env_site = Id_env_0
        envB = self.Apply_TB_from_Right(vec=Id_env_0, Bs_list=Bs_list) * cmath.exp(1.j*self.psi.momentum)
        for i in range(2, self.num_UC):
            envB = self.Apply_T(vec=envB, transpose=True)
            env_site = self.Apply_T(vec=env_site, transpose=True) * cmath.exp(1.j*self.psi.momentum)
            env_add = self.Apply_TB_from_Right(vec=env_site, Bs_list=Bs_list) * cmath.exp(1.j*self.psi.momentum)

            envB += env_add
        return envB
    
    def init_onsite_env(self, x):
        x0 = x
        env_onsite = self.generate_initial_eye()
        for x_i in range(x0+1, x0+self.len_UC):
            env_onsite = self.Apply_T(env_onsite, transpose=True)
        return env_onsite

    def init_onsite_env_c(self, x):
        x0 = x
        env_onsite_c = []
        for y_i in range(0, self.len_UC):
            env_y = self.env_onsite_0.copy(deep=True)
            for y_r in range(self.len_UC-1, y_i, -1):
                env_y = npc.tensordot(env_y, self.psi._As[y_r], axes=[['vL'], ['vR']])
                env_y = npc.tensordot(env_y, self.psi._As[y_r].conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])
            for y_l in range(0, y_i, +1):
                env_y = npc.tensordot(env_y, self.psi._As[y_l], axes=[['vR'], ['vL']])
                env_y = npc.tensordot(env_y, self.psi._As[y_l].conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
            env_onsite_c.append(env_y)
        return env_onsite_c

    def Apply_T(self, vec, transpose=False):
        Labels = vec.get_leg_labels()
        if not transpose:
            for y_i in range(self.len_UC-1, -1, -1):
                A_u = self.psi._As[y_i]
                A_d = self.psi._As[y_i]
                vec = npc.tensordot(vec, A_u, axes=["vL", "vR"])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        elif transpose:
            for y_i in range(0, self.len_UC):
                A_u = self.psi._As[y_i]
                A_d = self.psi._As[y_i]
                vec = npc.tensordot(vec, A_u, axes=["vR", "vL"])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        return vec.itranspose(Labels)
    
    def Apply_TB_from_Right(self, vec, Bs_list):
        Labels = vec.get_leg_labels()
        L_B = npc.tensordot(Bs_list[0], vec, axes=[["vL"], ["vR"]])
        L_B = npc.tensordot(L_B, self.psi._As[0].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

        Lp_site = vec

        for y_i in range(1, self.len_UC, +1):
            L_B = npc.tensordot(L_B, self.psi._As[y_i], axes=[["vR"], ["vL"]])
            L_B = npc.tensordot(L_B, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            
            Lp_site = npc.tensordot(Lp_site, self.psi._As[y_i-1], axes=[["vR"], ["vL"]])
            Lp_site = npc.tensordot(Lp_site, self.psi._As[y_i-1].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            L_B_add = npc.tensordot(Lp_site, Bs_list[y_i], axes=[["vR"], ["vL"]])
            L_B_add = npc.tensordot(L_B_add, self.psi._As[y_i].conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            L_B += L_B_add
        return L_B.itranspose(Labels)

    def generate_initial_eye(self):
        """
        Generate the three two Identity at the end of x unit cell
        """
        leg_u = self.psi._As[0].get_leg('vL')
        leg_d = self.psi._As[0].conj().get_leg('vL*')
        eye_u = npc.diag(1., leg=leg_u, dtype=self.dtype, labels=['vL', 'vR'])
        eye_d = npc.diag(1., leg=leg_d, dtype=self.dtype, labels=['vL*', 'vR*'])
        eye_init = npc.outer(eye_u, eye_d)
        return eye_init

    def Normalize_As(self):
        Norm = self.Apply_T(self.env_onsite_0, transpose=True)
        Norm = npc.trace(Norm, leg1='vL', leg2='vR')
        Norm = npc.trace(Norm, leg1='vL*', leg2='vR*')
        print("Initialize in puMPS, Norm is:{}".format(Norm))
        for y_i in range(0, self.len_UC):
            self.psi._As[y_i] = self.psi._As[y_i] / np.sqrt(Norm)

class NDressed_H_eff(H_eff_MultiSite):
    """
    eig_vals: list of float | list of complex
    eig_vecs: :class: Array | list of Array
    """
    def __init__(self, psi, H_MPO, Lx, Ly, eig_vals, eig_vecs, tol_eig=0.5e-2, x=0) -> None:
        super().__init__(psi, H_MPO, Lx, Ly, x)
        if len(eig_vals) > len(eig_vecs):
            for i in range(len(eig_vals)-1, len(eig_vecs)-1, -1):
                eig_vals = np.delete(eig_vals, i)
        if len(eig_vals) != len(eig_vecs):
            raise ValueError("After deletion, length of eig_vals should be equal to length of eig_vecs!")
        self.dim_N = len(eig_vals)
        for i in range(self.dim_N-1, -1, -1):
            if eig_vals[i] < tol_eig:
                eig_vals = np.delete(eig_vals, i)
                del eig_vecs[i]
            elif eig_vals[i] > tol_eig:
                break
        self.dim_N = len(eig_vals)
        print("Dim_N after {}".format(self.dim_N))
        self.eigs_inv = 1.0/np.real(eig_vals)

        if isinstance(eig_vecs[0], npc.Array):
            self.basis_list = [psi.Vec_to_List(eig_vecs[i]) for i in range(0, self.dim_N)]
        else:
            self.basis_list = eig_vecs
        self.E_shift = np.real(self.Cal_GS_Energy())
        print("E_shift is:{}".format(self.E_shift))

    def matvec(self, vec):
        vec_0 = vec.copy(deep=True)
        H_vec = super().matvec(vec)
        H_vec_list = self.psi.Vec_to_List(H_vec)
        ND_H_vec_list = [H_vec_list[y].zeros_like() for y in range(self.len_UC)]
        for i in range(self.dim_N):
            Amplitude_i = 0.0
            for y_i in range(self.len_UC):
                Amplitude_i += npc.tensordot(H_vec_list[y_i].conj(), self.basis_list[i][y_i], axes=[['vL*', 'p*', 'vR*'], ['vL', 'p', 'vR']])
            for y_i in range(self.len_UC):
                ND_H_vec_list[y_i] += self.eigs_inv[i] * Amplitude_i * self.basis_list[i][y_i]

        ND_H_vec = self.psi.List_to_Vec(ND_H_vec_list)
        return ND_H_vec - vec_0*self.E_shift
    
class Masked_H_Masked_eff(H_eff_MultiSite):
    """
    eig_vals: list of float | list of complex
    eig_vecs: :class: Array | list of Array
    """
    def __init__(self, psi, H_MPO, Lx, Ly, eig_vals, eig_vecs, sort='SR', tol_eig=1.e-3, x=0) -> None:
        super().__init__(psi, H_MPO, Lx, Ly, x)
        if len(eig_vals) > len(eig_vecs):
            for i in range(len(eig_vals)-1, len(eig_vecs)-1, -1):
                eig_vals = np.delete(eig_vals, i)
        if len(eig_vals) != len(eig_vecs):
            raise ValueError("After deletion, length of eig_vals should be equal to length of eig_vecs!")
        self.dim_N = len(eig_vals)

        if sort == 'SR':
            eig_vecs.reverse()
            eig_vals = eig_vals[::-1]
            
        for i in range(self.dim_N-1, -1, -1):
            if np.real(eig_vals[i]) < tol_eig:
                eig_vals = np.delete(eig_vals, i)
                del eig_vecs[i]
            elif eig_vals[i] > tol_eig:
                break
        
            
        self.dim_N = len(eig_vals)
        print("Dim_N after {}".format(self.dim_N))
        self.eigs_inv_root = 1.0/np.sqrt(np.real(eig_vals))
        print(self.eigs_inv_root)

        if isinstance(eig_vecs[0], npc.Array):
            self.basis_list = [psi.Vec_to_List(eig_vecs[i]) for i in range(0, self.dim_N)]
        else:
            self.basis_list = eig_vecs
        self.E_shift = np.real(self.Cal_GS_Energy())
        print("E_shift is:{}".format(self.E_shift))

    def matvec(self, vec):
        """
        Dress the Hamiltonian to $\Lambda_{N}^{-1/2}U_{N}^\dagger H U_{N}\Lambda_{N}^{-1/2}$
        """
        vec_0 = vec.copy(deep=True)
        vec_list = self.psi.Vec_to_List(vec)
        Masked_vec_list = [vec_list[y].zeros_like() for y in range(self.len_UC)]

        for i in range(self.dim_N):
            Amplitude_i = 0.0
            for y_i in range(self.len_UC):
                # Get the overlap of the ith basis with the vec, i.e. the component of the vector on the ith basis
                Amplitude_i += npc.tensordot(vec_list[y_i].conj(), self.basis_list[i][y_i], axes=[['vL*', 'p*', 'vR*'], ['vL', 'p', 'vR']])
            for y_i in range(self.len_UC):
                Masked_vec_list[y_i] += self.eigs_inv_root[i] * Amplitude_i * self.basis_list[i][y_i]

        H_Masked_vec = super().matvec(self.psi.List_to_Vec(Masked_vec_list))
        H_Masked_vec_list = self.psi.Vec_to_List(H_Masked_vec)
        M_H_M_vec_list = [H_Masked_vec_list[y].zeros_like() for y in range(self.len_UC)]

        for i in range(self.dim_N):
            Amplitude_i = 0.0
            for y_i in range(self.len_UC):
                Amplitude_i += npc.tensordot(H_Masked_vec_list[y_i].conj(), self.basis_list[i][y_i], axes=[['vL*', 'p*', 'vR*'], ['vL', 'p', 'vR']])
            for y_i in range(self.len_UC):
                M_H_M_vec_list[y_i] += self.eigs_inv_root[i] * Amplitude_i * self.basis_list[i][y_i]

        M_H_M_vec = self.psi.List_to_Vec(M_H_M_vec_list)
        return M_H_M_vec - vec_0*self.E_shift