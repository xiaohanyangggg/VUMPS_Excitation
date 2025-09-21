import numpy as np
import cmath
import copy
import tracemalloc
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPOEnvironment, MPOTransferMatrix
from tenpy.algorithms.mps_common import *
from tenpy.linalg.krylov_based import Arnoldi, GMRES
from .Exi_Transfer_System import Exi_Transfer_System
from .H_eff_ex import Excited_OnesiteH, Excited_MultisiteH
# from ...Linalg.GMRES_NPC import GMRES
from ...System.mps_Replica import MPS_Replica
from ...System.Bond_Control_RE_RE import l_Ortho, r_Ortho

class Excited_Edge(Excited_MultisiteH):
    def __init__(self, env, psi_GS, H, k, X_edge_list, exci_site_num=1, side="first", group_sites=True, ex_charge=[0]):
        r"""
        This class applies excitation ansatz which only perturbates on one edge.
        parameters:
        X_edge_list: list of :class: `Array`
            The X tensors on the edge we consider.
        
        exci_site_num: int
            Number of the sites on the edge the excitation state covers.
        
        side : `` 'first' | 'last' ``
            The side on which we detect the excitation.
        """
        self.env = env
        self.FR = env.get_RP(psi_GS.L-1)
        self.FL = env.get_LP(0)
        self.psi_GS = psi_GS
        self.H_MPO = H
        self.IdL = H.get_IdL(0)
        self.IdR = H.get_IdR(psi_GS.L-1)
        self.L_UC = psi_GS.L
        self.exci_length = exci_site_num

        print("Exci_site_num:{}".format(exci_site_num))
        print("length of X_list:{}".format(len(X_edge_list)))
        assert exci_site_num == len(X_edge_list)

        if side == "first":
            self.side = side
            self.exci_site = 0
        elif side == "last":
            self.side = side
            self.exci_site = self.L_UC - self.exci_length
        self.k = k
        if group_sites:
            self.Delta_x = 1
        else:
            self.Delta_x = self.L_UC
        
        self.L_Ortho = []
        self.R_Ortho = []
        self.X_edge_list = X_edge_list
        self.X_list_com = []
        self.X_legs = []
        self.X_length = []
        self.X_slice = [0]
        self.E_shift = []
        for i in range(len(X_edge_list)):
            self.X_list_com.append(X_edge_list[i].combine_legs(["vl", "vR"]))
            self.X_length.append(self.X_list_com[i].shape[0])
            self.X_slice.append(self.X_slice[i] + self.X_length[i])
            self.X_legs.append(self.X_list_com[i].get_leg("(vl.vR)"))
        for j in range(self.L_UC):
            self.L_Ortho.append(l_Ortho(psi_GS, j).split_legs())
            self.R_Ortho.append(r_Ortho(psi_GS, j).split_legs())
        
        # Energy shift in the on-site term
        for j in range(self.L_UC):
            Lp_j = env.get_LP(j)
            Rp_j = env.get_RP(j)
            E0 = npc.tensordot(self.psi_GS.get_Bc(j), Lp_j, axes=[["vL"], ["vR"]])
            E0 = npc.tensordot(E0, self.H_MPO.get_W(j), axes=[["wR", "p"], ["wL", "p*"]])
            E0 = npc.tensordot(E0, Rp_j, axes=[["vR", "wR"], ["vL", "wL"]])
            E0 = npc.tensordot(E0, self.psi_GS.get_Bc(j).conj(), axes=[["vR*", "p", "vL*"], ["vL*", "p*", "vR*"]])
            self.E_shift.append(E0)

        # We initialize the L and R transfer system just with B0 at the 0th site and we will change it later on 
        B_legs = psi_GS.get_Bc(0).legs
        B_leg_labels = psi_GS._Bc[0].get_leg_labels()
        B0 = npc.Array.from_func(func = np.random.standard_normal, legcharges=B_legs, 
                                 dtype=psi_GS.get_Bc(0).dtype, qtotal=ex_charge, labels=B_leg_labels)    
        self.L_Transfer = Exi_Transfer_System(psi_GS, H, B0, k, 0, self.FL, self.FR, group_sites=group_sites, transpose=False)
        self.R_Transfer = Exi_Transfer_System(psi_GS, H, B0, k, 0, self.FL, self.FR, group_sites=group_sites, transpose=True)
        
        self.dtype = np.find_common_type([psi_GS.dtype, psi_GS.dtype, H.dtype], [])
    
    def matvec(self, X):
        """ Note that here X is the concatenation of several X`s live on one edge """
        # X should be the direct sum of the leg-combined X`s
        # Here we treat X as a vector
        # tracemalloc.start()
        X_old = []
        B_old = []
        X_new = []

        # To get X at each site of the UC from the direct sum of X`s
        # Extract the old X(B)`s to put them into the eigensolver
        for i in range(self.exci_length):
            X_old_i = X[self.X_slice[i] : self.X_slice[i+1]]    # Take the slices corresponding to the X tensor at the i site
            X_old_ii = npc.Array.from_ndarray(data_flat=X_old_i.to_ndarray(), legcharges=[self.X_legs[i]],
                                              dtype=self.dtype, qtotal=X_old_i.qtotal, labels=["(vl.vR)"])
            X_old.append(X_old_ii.split_legs())
            B_old.append(self.get_B_from_X(X_old_ii.split_legs(), i))
        
        # Get environments :
        # 1) L_B, R_B : B tensors of the ket mps are all at different unit cells from the bra's
        # 2) L_env_onsite, R_env_onsite : B tensors of the ket mps are at the same unit cell as the bra's
        # In case 2 we consider 2 lists of envs since finite length environments depend on the position
        L_B_init, R_B_init = self.Init_B_env(B_old, self.FL, self.FR)
        # print("Before Calc LB and RB.\n")
        L_B, R_B = self.get_B_env_GMRES(L_B_init, R_B_init)
        L_env_onsite, R_env_onsite = self.get_onsite_envs(B_old)
        # Get the new X from each site of the UC
        # Here the i index is out tensor B`s site
        for i in range(self.exci_site, self.exci_site + self.exci_length):
            B_i = self.Construct_H_chiral(self.FL, R_B, i_out=i, move="left")
            B_i += self.Construct_H_chiral(L_B, self.FR, i_out=i, move="right")
            ind_B = self.convert_MPS_to_B_index(i)
            B_i += self.Onsite_term(B_old[ind_B], L_env_onsite[ind_B], R_env_onsite[ind_B], i_out=i)
            
            X_i = self.proj_B_to_X(B_i, i)
            X_new.append(X_i)
            
        self.X_list = copy.deepcopy(X_new)
        X = X_new[0].combine_legs(["vl", "vR"])
        for k in range(1, self.exci_length):
            X = npc.concatenate([X, X_new[k].combine_legs(["vl", "vR"])], axis=0)
        # print("H_eff_multi has X_list:{}".format(self.X_list))
        # print("X vec is:{}".format(X))
        # current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        # print(f"Peak memory usage: {peak / 1024 /1024:.2f} MB")
        return X
    
    """TODO: Debug the `Init_B_env()` Method."""
    def Init_B_env(self, B, Lp, Rp):
        L = self.psi_GS.L
        if self.side == "first":
            # R_B
            R_B = Rp
            # For example, range(7, 1)
            for i in range(L-1, self.exci_site+self.exci_length-1, -1):
                R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="B"), axes=["vL", "vR"])
                R_B = npc.tensordot(R_B, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            Rp_site = R_B.copy(deep=True)
            # For example, range(1, -1)
            for i in range(self.exci_site+self.exci_length-1, -1, -1):
                R_B_add = npc.tensordot(B[i], Rp_site, axes=[["vR"], ["vL"]])
                R_B_add = npc.tensordot(R_B_add, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                R_B_add = npc.tensordot(R_B_add, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
                if i == self.exci_site+self.exci_length-1:
                    R_B = R_B_add
                else:
                    R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="A"), axes=[["vL"], ["vR"]])
                    R_B = npc.tensordot(R_B, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                    R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
                    R_B += R_B_add

                Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form="B"), axes=[["vL"], ["vR"]])
                Rp_site = npc.tensordot(Rp_site, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            # L_B
            L_B = Lp
            Lp_site = L_B.copy(deep=True)
            # For example, range(0, 2)
            for i in range(0, self.exci_length, +1):
                L_B_add = npc.tensordot(B[i], Lp_site, axes=[["vL"], ["vR"]])
                L_B_add = npc.tensordot(L_B_add, self.H_MPO.get_W(i), axes=[["p", "wR"], ["p*", "wL"]])
                L_B_add = npc.tensordot(L_B_add, self.psi_GS.get_B(i, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
                if i == 0:
                    L_B = L_B_add
                else:
                    L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="B"), axes=[["vR"], ["vL"]])
                    L_B = npc.tensordot(L_B, self.H_MPO.get_W(i), axes=[["p", "wR"], ["p*", "wL"]])
                    L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
                    L_B += L_B_add

                Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form="A"), axes=[["vR"], ["vL"]])
                Lp_site = npc.tensordot(Lp_site, self.H_MPO.get_W(i), axes=[["p", "wR"], ["p*", "wL"]])
                Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            # For example, range(2, 8)
            for i in range(self.exci_length, L, +1):
                L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="B"), axes=["vR", "vL"])
                L_B = npc.tensordot(L_B, self.H_MPO.get_W(i), axes=[["wR", "p"], ["wL", "p*"]])
                L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="A").conj(), axes=[["vR*", "p"], ["vL*", "p*"]])

        if self.side == "last":
            # R_B
            R_B = Rp
            Rp_site = R_B.copy(deep=True)
            for i in range(L-1, L-self.exci_length-1, -1):
                ind_B = self.convert_MPS_to_B_index(i)
                R_B_add = npc.tensordot(B[ind_B], Rp_site, axes=[["vR"], ["vL"]])
                R_B_add = npc.tensordot(R_B_add, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                R_B_add = npc.tensordot(R_B_add, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
                if i == L-1:
                    R_B = R_B_add
                else:
                    R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="A"), axes=[["vL"], ["vR"]])
                    R_B = npc.tensordot(R_B, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                    R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
                    R_B += R_B_add

                Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form="B"), axes=[["vL"], ["vR"]])
                Rp_site = npc.tensordot(Rp_site, self.H_MPO.get_W(i), axes=[["p", "wL"], ["p*", "wR"]])
                Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

            for i in range(L-self.exci_length-1, -1, -1):
                R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="A"), axes=["vL", "vR"])
                R_B = npc.tensordot(R_B, self.H_MPO.get_W(i), axes=[["wL", "p"], ["wR", "p*"]])
                R_B = npc.tensordot(R_B, self.psi_GS.get_B(i, form="B").conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
            # L_B
            L_B = Lp
            Lp_site = L_B.copy(deep=True)
            for i in range(0, L-self.exci_length, +1):
                L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="A"), axes=["vR", "vL"])
                L_B = npc.tensordot(L_B, self.H_MPO.get_W(i), axes=[["wR", "p"], ["wL", "p*"]])
                L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="A").conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
            for i in range(L-self.exci_length, L, +1):
                ind_B = self.convert_MPS_to_B_index(i)
                L_B_add = npc.tensordot(B[ind_B], Lp_site, axes=[["vL"], ["vR"]])
                L_B_add = npc.tensordot(L_B_add, self.H_MPO.get_W(i), axes=[["p", "wR"], ["p*", "wL"]])
                L_B_add = npc.tensordot(L_B_add, self.psi_GS.get_B(i, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
                if i == L-self.exci_length:
                    L_B = L_B_add
                else:
                    L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="B"), axes=[["vR"], ["vL"]])
                    L_B = npc.tensordot(L_B, self.H_MPO.get_W(i), axes=[["p", "wR"], ["p*", "wL"]])
                    L_B = npc.tensordot(L_B, self.psi_GS.get_B(i, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
                    L_B += L_B_add

                Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form="A"), axes=[["vR"], ["vL"]])
                Lp_site = npc.tensordot(Lp_site, self.H_MPO.get_W(i), axes=[["p", "wR"], ["p*", "wL"]])
                Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
        return L_B, R_B
            
    def get_onsite_envs(self, B):
        '''
        This method will give the environments on a single unit cell, as the case the system is finite.
        Returns : L_env_onsite/R_env_onsite : 1 x self.exci_length list
        '''
        L_exci = self.exci_length
        L_env_onsite = [None] * L_exci
        R_env_onsite = [None] * L_exci
        L_env_onsite[0] = self.env.get_LP(self.exci_site)
        R_env_onsite[L_exci - 1] = self.env.get_RP(self.exci_site + L_exci - 1)
        Lp_site = L_env_onsite[0].copy(deep=True)
        Rp_site = R_env_onsite[L_exci - 1].copy(deep=True)
        # Get left onsite envs
        for site in range(1, L_exci, +1):
            ind_MPS = self.convert_B_to_MPS_index(site)
            if site == 1:
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site-1], B[site-1], axes=[["vR"], ["vL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.H_MPO.get_W(ind_MPS-1), axes=[["p", "wR"], ["p*", "wL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.psi_GS.get_B(ind_MPS-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            else:
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site-1], self.psi_GS.get_B(ind_MPS-1, form="B"), axes=[["vR"], ["vL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.H_MPO.get_W(ind_MPS-1), axes=[["p", "wR"], ["p*", "wL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.psi_GS.get_B(ind_MPS-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

                L_B_add = npc.tensordot(B[site-1], Lp_site, axes=[["vL"], ["vR"]])
                L_B_add = npc.tensordot(L_B_add, self.H_MPO.get_W(ind_MPS-1), axes=[["p", "wR"], ["p*", "wL"]])
                L_B_add = npc.tensordot(L_B_add, self.psi_GS.get_B(ind_MPS-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
                
                L_env_onsite[site] += L_B_add

            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(ind_MPS-1, form="A"), axes=[["vR"], ["vL"]])
            Lp_site = npc.tensordot(Lp_site, self.H_MPO.get_W(ind_MPS-1), axes=[["p", "wR"], ["p*", "wL"]])
            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(ind_MPS-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

        # Get right onsite envs
        for site in range(L_exci-2, -1, -1):
            ind_MPS = self.convert_B_to_MPS_index(site)
            if site == L_exci-2:
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site+1], B[site+1], axes=[["vL"], ["vR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.H_MPO.get_W(ind_MPS+1), axes=[["p", "wL"], ["p*", "wR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.psi_GS.get_B(ind_MPS+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            else:
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site+1], self.psi_GS.get_B(ind_MPS+1, form="A"), axes=[["vL"], ["vR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.H_MPO.get_W(ind_MPS+1), axes=[["p", "wL"], ["p*", "wR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.psi_GS.get_B(ind_MPS+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

                R_B_add = npc.tensordot(B[site+1], Rp_site, axes=[["vR"], ["vL"]])
                R_B_add = npc.tensordot(R_B_add, self.H_MPO.get_W(ind_MPS+1), axes=[["p", "wL"], ["p*", "wR"]])
                R_B_add = npc.tensordot(R_B_add, self.psi_GS.get_B(ind_MPS+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
                
                R_env_onsite[site] += R_B_add
                
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(ind_MPS+1, form="B"), axes=[["vL"], ["vR"]])
            Rp_site = npc.tensordot(Rp_site, self.H_MPO.get_W(ind_MPS+1), axes=[["p", "wL"], ["p*", "wR"]])
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(ind_MPS+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

        return L_env_onsite, R_env_onsite

    def Onsite_term(self, B_i, L_env, R_env, i_out):
        Lp = self.env.get_LP(i_out)
        Rp = self.env.get_RP(i_out)
        '''
        |L_B|A_R|R_p|
        '''
        B_out = npc.zeros(legcharges=B_i.legs, dtype=B_i.dtype, qtotal=B_i.qtotal, labels=B_i.get_leg_labels())
        if i_out != self.exci_site:
            B_out_l = npc.tensordot(self.psi_GS.get_B(i_out, form="B"), L_env, axes=[["vL"], ["vR"]])
            B_out_l = npc.tensordot(B_out_l, self.H_MPO.get_W(i_out), axes=[["wR", "p"], ["wL", "p*"]])
            B_out_l = npc.tensordot(B_out_l, Rp, axes=[["vR", "wR"], ["vL", "wL"]]).replace_labels(["vR*", "vL*"], ["vL", "vR"])
            B_out += B_out_l
        '''
        |L_p|A_L|R_B|
        '''
        if i_out != self.exci_site + self.exci_length - 1:
            B_out_r = npc.tensordot(self.psi_GS.get_B(i_out, form="A"), Lp, axes=[["vL"], ["vR"]])
            B_out_r = npc.tensordot(B_out_r, self.H_MPO.get_W(i_out), axes=[["wR", "p"], ["wL", "p*"]])
            B_out_r = npc.tensordot(B_out_r, R_env, axes=[["vR", "wR"], ["vL", "wL"]]).replace_labels(["vR*", "vL*"], ["vL", "vR"])
            B_out += B_out_r
        '''
        |L_p|B|R_p|
        '''
        B_out_c = npc.tensordot(B_i, Lp, axes=[["vL"], ["vR"]])
        B_out_c= npc.tensordot(B_out_c, self.H_MPO.get_W(i_out), axes=[["wR", "p"], ["wL", "p*"]])
        B_out_c = npc.tensordot(B_out_c, Rp, axes=[["vR", "wR"], ["vL", "wL"]]).replace_labels(["vR*", "vL*"], ["vL", "vR"])
        B_out += B_out_c
        # <\phi(B)|H|\phi(B)> - <H>_0
        E0 = self.E_shift[i_out]
        # print("E_0 is:{}".format(E0))
        B_out -= E0 * B_i
        
        return B_out

    def convert_MPS_to_B_index(self, i):
        # Since the index in the mps UC may not match the index of X list, we need to shift it
        # self.exci_site = self.L_UC - self.exci_length
        # You can check that if i = L - self.exci_length, this method will return 0
        # If i = L - 1, this method will return self.exci_length - 1
        return i - self.exci_site
    
    def convert_B_to_MPS_index(self, i):
        return i + self.exci_site
    

