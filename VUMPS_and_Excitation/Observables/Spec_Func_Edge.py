import numpy as np
import cmath

import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPSEnvironment, TransferMatrix
from tenpy.linalg.krylov_based import GMRES

from VUMPS_and_Excitation.System.Bond_Control_RE_RE import l_Ortho
from VUMPS_and_Excitation.Algorithm.Single_Mode_Approximation. Exi_Transfer_System import Exi_Transfer_System_MPS

class Spectral_function():
    """Note that in this class, B is chosen to be left-orthogonal."""
    """
    Parameters
    ----------
    psi : :class: `MPS_Replica`
        The ground state wave function
    X_lists : 3d list of :class: `Array`
        The list of X tensors which encode the excitation wave function for different energy levels at different momentum k
        X_lists[k][n] is the list of X tensors of nth enrgy level at momentum k
    Eks : 2d list of `float`
        Energy levels at different momentum k
    sites : list of :class: `tenpy.networks.site.Site`
        The list of sites used to get the operators acting on different sites
    ops_str : list of string
        Names of operators defined by the spectral function chosen
    ops_indices : list of int 
        Site indices which operators act on
    eta : float
        The Lorentzian broadening parameter

    Attributes
    ----------
    overlap_matrix
    """
    def __init__(self, psi, X_lists, Eks, ops_str, ops_indices, eta_w, eta_k, k_space, Kry_Para) -> None:
        self.psi_GS = psi
        # self.X_lists = X_lists
        # Delete X_lists as an attribute since we just need the overlap matrix for further calculation
        self.Eks = Eks
        self.num_exci_k = len(Eks)
        self.k_space = k_space
        self.num_exci_level = len(Eks[0])
        self.L_UC = psi.L
        self.sites = psi.sites
        self.eta_w = eta_w
        self.eta_k = eta_k
        self.L_Ortho = []
        assert self.num_exci_k == k_space.size
        
        for i in range(0, psi.L):
            self.L_Ortho.append(l_Ortho(psi, i).split_legs())
        
        op_n = []
        for n in range(0, len(ops_str)):
            op_n.append(self.sites[ops_indices[n]].get_op(ops_str[n]))

        if len(op_n) != len(ops_indices):
            raise ValueError("Operators list should match the indices list!")
        
        ops_flag = [0] * self.L_UC
        for ind in ops_indices:
            ops_flag[ind] = 1

        self.ops = op_n
        self.ops_indices = ops_indices
        self.ops_flag = ops_flag
        LP_op_init = self.Prepare_initial_LP_op(op_n, ops_flag)

        self.Kry_Para = Kry_Para
        overlap_matrix = []
        """
        The overlap matrix is defined by <k, w_k|O_0|psi_GS>
        """
        n_k = 0
        for momentum in k_space:
            overlap_k = []
            # In the H_eff module, I think I use the name "R_Transfer" for such a transposed operator...
            L_Transfer = Exi_Transfer_System_MPS(self.psi_GS, 
                                                 momentum,
                                                 T_type="LL",
                                                 group_sites=True, 
                                                 transpose=True)
            LP_op_solver = GMRES(A=L_Transfer, 
                                 x=LP_op_init,
                                 b=LP_op_init,
                                 options=Kry_Para)
            LP_op_k, _, _, _ = LP_op_solver.run()
            LP_op_k = cmath.exp(-1.j*momentum*L_Transfer.Delta_x) * LP_op_k
        
            for w in range(self.num_exci_level):
                print("n_k: {}, w: {}".format(n_k, w))
                X_list = X_lists[n_k][w]
                R_B = self.Prepare_RB(X_list)
                overlap_k_w = npc.tensordot(LP_op_k, R_B, axes=[['vR', 'vR*'], ['vL', 'vL*']])
                overlap_k_w += self.Calc_Onsite(X_list)
                overlap_k.append(overlap_k_w)
                # print("At the k point {}, the {} energy level, intensity is:{}".format(momentum, w, np.abs(overlap_k_w)**2))
            overlap_matrix.append(overlap_k)
            n_k = n_k + 1
        self.overlap_matrix = overlap_matrix


    def Calculate_A_k_w(self, k, omega):
        """
        This method is aimed to calculate A(k,w) at points which were not sampled in the single-mode calculation.
        Parameters
        ----------
        k : float | int
            If `float`: The momentum point k \in [0, 2*math.pi] or [-*math.pi, math.pi]
            If `int`: Some k mesh index of our excitation data
        omega : float
            Some energy value w on the (k, w) plane
        """
        A_k_w = 0
        if self.eta_k != 0:
            for n in range(self.num_exci_level):
                for n_k in range(self.num_exci_k):
                    A_k_w += self.Lorentzian_k(self.k_space[n_k], k) * self.Lorentzian_w(self.Eks[n_k][n], omega) * abs(self.overlap_matrix[n_k][n])**2
        if self.eta_k == 0:
            for n in range(self.num_exci_level):
                k = int(k)
                A_k_w += self.Lorentzian_w(self.Eks[k][n], omega) * abs(self.overlap_matrix[k][n])**2
        return A_k_w


    def Calc_Onsite(self, X_list):
        """
        Calculate the term like:
        -----[AL]----
        |     |     |
        |    [op]  [C]
        |     |     |
        |____[B]____|
        """
        start_i = self.ops_indices[0]
        op_order = 0
        L_B = npc.tensordot(self.psi_GS.get_B(start_i, form='A'), self.ops[op_order], axes=[['p'], ['p*']])
        Lp_site = npc.tensordot(L_B, self.psi_GS.get_B(start_i, form='A').conj(), axes=[['p', 'vL'], ['p*', 'vL*']])
        B_first = npc.tensordot(self.L_Ortho[start_i], X_list[start_i], axes=[['vr'], ['vl']])
        L_B = npc.tensordot(L_B, B_first.conj(), axes=[['p', 'vL'], ['p*', 'vL*']])
        for i in range(start_i+1, self.L_UC):
            L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form='A'), axes=[['vR'], ['vL']])
            L_B_add = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form='A'), axes=[['vR'], ['vL']])
            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form='A'), axes=[['vR'], ['vL']])
            if self.ops_flag[i] == 1:
                op_order += 1
                L_B = npc.tensordot(L_B, self.ops[op_order], axes=[['p'], ['p*']])
                L_B_add = npc.tensordot(Lp_site, self.ops[op_order], axes=[['p'], ['p*']])
                Lp_site = npc.tensordot(Lp_site, self.ops[op_order], axes=[['p'], ['p*']])
            L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form='B').conj(), axes=[['p', 'vR*'], ['p*', 'vL*']])
            B_i = npc.tensordot(self.L_Ortho[i], X_list[i], axes=[['vr'], ['vl']])
            L_B_add = npc.tensordot(L_B_add, B_i.conj(), axes=[['p', 'vR*'], ['p*', 'vL*']])
            L_B += L_B_add
            # Update Lp_site
            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form='A').conj(), axes=[['p', 'vR*'], ['p*', 'vL*']])
        A_onsite = npc.tensordot(L_B, self.psi_GS.get_C(0), axes=[['vR', 'vR*'], ['vL', 'vR']])
        return A_onsite
    
    def Prepare_RB(self, X_list):
        """
        We prepare the right environment with B ansastz:
            _________
        ---|____B____|--- = ---B1---AL2---... + ---AR1---B2---... + ...
             | | | |           |    |               |    |
        
        where ---Bi--- = ---VLi---Xi---
                 |          |
        R_B is:
        _________
            |    |
            |    |
            |   [C]
            |    |
        ___[B]___|
        In this diagram B`s leg should be a legpipe.
        """
        R_B = self.psi_GS.get_C(0).replace_labels(['vR'], ['vL*'])
        R_B = npc.tensordot(R_B, self.psi_GS.get_B(self.L_UC-1, form='A'), axes=[['vL'], ['vR']])
        Rp_site = npc.tensordot(R_B, self.psi_GS.get_B(self.L_UC-1, form='B').conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])
        
        B_last = npc.tensordot(self.L_Ortho[self.L_UC-1], X_list[self.L_UC-1], axes=['vr', 'vl'])
        R_B = npc.tensordot(R_B, B_last.conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])

        for i in range(self.L_UC-2, -1, -1):
            B_i = npc.tensordot(self.L_Ortho[i], X_list[i], axes=['vr', 'vl'])
            
            R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form='A'), axes=[['vL'], ['vR']])
            R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form='A').conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])

            R_B_add = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form='A'), axes=[['vL'], ['vR']])
            R_B_add = npc.tensordot(R_B_add, B_i.conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])

            R_B += R_B_add

            # Update Rp_site
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form='A'), axes=[['vL'], ['vR']])
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form='B').conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])   

        return R_B    

    def Prepare_initial_LP_op(self, op_n, ops_flag):
        """
        We first prepare the left part of the following diagram:
         _________             _______ n
        |    |                    |
        |   [op]  ... x \sum_n    |     exp(-ikn)
        |____|____             ___|___
        If self.L_UC > 1, just use the large UC version of this equation
        """
        op_order = 0
        LP_op = self.psi_GS.get_B(0, form="A")
        for n in range(0, self.L_UC):
            if n == 0: 
                if ops_flag[n] == 1:
                    LP_op = npc.tensordot(LP_op, op_n[op_order], axes=[['p'], ['p*']])
                    op_order += 1
                LP_op = npc.tensordot(LP_op, self.psi_GS.get_B(n, form="A").conj(), axes=[['vL', 'p'], ['vL*', 'p*']])
            else:
                LP_op = npc.tensordot(LP_op, self.psi_GS.get_B(n, form='A'), axes=[['vR'], ['vL']])
                if ops_flag[n] == 1:
                    LP_op = npc.tensordot(LP_op, op_n[op_order], axes=[['p'], ['p*']])
                    op_order += 1
                LP_op = npc.tensordot(LP_op, self.psi_GS.get_B(n, form="A").conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
        return LP_op

    def Lorentzian_w(self, Ek, omega):
        Loren = self.eta_w/(self.eta_w**2 + ((Ek-omega).real)**2)
        return Loren / cmath.pi
    
    def Lorentzian_k(self, k_n, k):
        Loren = self.eta_k/(self.eta_k**2 + (k_n-k)**2)
        return Loren / cmath.pi
        
          

        