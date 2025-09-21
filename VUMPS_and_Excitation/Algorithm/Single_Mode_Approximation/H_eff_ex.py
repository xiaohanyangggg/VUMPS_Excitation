import numpy as np
import cmath
import copy
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPOEnvironment, MPOTransferMatrix
from tenpy.algorithms.mps_common import *
from tenpy.linalg.krylov_based import Arnoldi, GMRES
from .Exi_Transfer_System import Exi_Transfer_System
# from ...Linalg.GMRES_NPC import GMRES
from ...System.mps_Replica import MPS_Replica
from ...System.Bond_Control_RE_RE import l_Ortho, r_Ortho

class Excited_OnesiteH(EffectiveH):     
    r"""The effective Hamiltonian to apply excitation ansatz with a fixed momentum k.
    The effective Hamiltonian should act on the `tangent vector` B like this:
    |                                                 ____                ____               
    |    .----  B  ----.                .---- AL ----|    |              |    |---- AR ----.
    |    |      |      |                |     |      |    |              |    |     |      |
    |   Lp---W[i_ex]---Rp  +  exp(ik)  Lp---W[i_ex]--|R(B)|  +  exp(-ik) |L(B)|--W[i_ex]---Rp
    |    |      |      |                |     |      |    |              |    |     |      |  
    |    .----     ----.                .----    ----|____|              |____|----    ----.
    |        onsite                        left mover                          right mover

    We can apply this effective Hamiltonian with method `matvec`
    
    Parameters
    ----------
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    psi_GS : :class:`MPS_Replica`
        Ground state wave function ``|psi_GS>``.
    H : :class:`MPO`
        Hamiltonian of the system which is written in mpo form.
    k : float
        Momentum of the excitation particle.
    site_B : The site where we create the excited particle.

    Attributes
    ----------
    i_ex : int
        The site at which we create an elementary excitation
    env : :class:`~tenpy.networks.mpo.MPOEnvironment`
        Environment for contraction ``<psi|H|psi>``.
    psi_GS : :class:`MPS_Replica`
        Ground state wave function ``|psi_GS>``.
    H_MPO : :class:`MPO`
        Hamiltonian of the system which is written in mpo form.
    L_UC : int
        Length of the mps unit cell. 
    momentum : float
        Momentum of the elementary excitation.
    A_L : :class:`~tenpy.linalg.np_conserved.Array`
        The original tensor at the `excited` site, in left canonical form.
    A_R : :class:`~tenpy.linalg.np_conserved.Array`
        The original tensor at the `excited` site, in right canonical form
    N_L : :class:`~tenpy.linalg.np_conserved.Array`
        A_L's orthonormal columns, with A_L reshaped to (Dd x D). And thus N_L is (Dd x D(d-1)).
    N_R : :class:`~tenpy.linalg.np_conserved.Array`
        A_R's orthonormal rows, with A_L reshaped to (D x Dd). And thus N_L is (D(d-1) x Dd).
    dtype : np.dtype
        The data type of the entries.
    """
    
    def __init__(self, env, psi_GS, H, k, site_B, ex_charge=None):
        self.i_ex = site_B
        self.env = env
        self.psi_GS = psi_GS
        self.H_MPO = H
        self.IdL = H.get_IdL(0)
        self.IdR = H.get_IdR(psi_GS.L-1)
        self.L_UC = psi_GS.L
        self.k = k
        self.A_L = psi_GS.get_B(site_B, form="A")
        self.A_R = psi_GS.get_B(site_B, form="B")
        self.N_L = l_Ortho(psi_GS, site_B).split_legs()
        self.N_R = r_Ortho(psi_GS, site_B).split_legs()
        self.FR = env.get_RP(psi_GS.L-1)
        self.FL = env.get_LP(0)
        # Maybe we can randomly generate a B tensor to intialize L and R Transfer System and update Transfer.B every time we change it
        # We initialize the Transfer_System at this stage because the four different leading eigen_environment should be calculated out of the
        # procedure ofn solving H_ex
        B_legs = psi_GS.get_Bc(site_B).legs
        B_leg_labels = psi_GS._Bc[site_B].get_leg_labels()
        B0 = npc.Array.from_func(func = np.random.standard_normal, legcharges=B_legs, dtype=psi_GS.get_Bc(site_B).dtype, qtotal=ex_charge, labels=B_leg_labels)
        self.L_Transfer = Exi_Transfer_System(psi_GS, H, B0, k, site_B, self.FL, self.FR, transpose=False)
        self.R_Transfer = Exi_Transfer_System(psi_GS, H, B0, k, site_B, self.FL, self.FR, transpose=True)
        # ================================================================================================
        self.dtype = np.find_common_type([psi_GS.dtype, psi_GS.dtype, H.dtype], [])

    def matvec(self, X):
        # Apply the effective Hamiltonian acting on the `tangent tensor` X, note that an on-site term exists   
        Lp = self.env.get_LP(0)
        Rp = self.env.get_RP(self.L_UC-1)
        B = self.get_B_from_X(X)
        Label_X = X.get_leg_labels()
        Label_B = B.get_leg_labels()
        R_B = self.get_B_env_GMRES(B, Lp, Rp, move="left")
        L_B = self.get_B_env_GMRES(B, Lp, Rp, move="right")
        B_onsite = self.Onsite_term(Lp, Rp, B).itranspose(Label_B)
        B_new = self.Construct_H_chiral(Lp, R_B, move="left").itranspose(Label_B) + self.Construct_H_chiral(L_B, Rp, move="right").itranspose(Label_B) + B_onsite
        en = npc.tensordot(self.psi_GS.get_Bc(0), Lp, axes=[["vL"], ["vR"]])
        en = npc.tensordot(en, self.H_MPO.get_W(0), axes=[["wR", "p"], ["wL", "p*"]])
        en = npc.tensordot(en, Rp, axes=[["vR", "wR"], ["vL", "wL"]])
        en = npc.tensordot(en, self.psi_GS.get_Bc(0).conj(), axes=[["vR*", "p", "vL*"], ["vL*", "p*", "vR*"]])
        B_new -= en * B
        X = self.proj_B_to_X(B_new).itranspose(Label_X)
        return X
    
    def Construct_H_chiral(self, L_env, R_env, move="left"):
        # Construct the H_chiral_with the left and right, envs, MPO, and psi_GS
        theta_R = R_env.copy()
        theta_L = L_env.copy()
        if move == "left":
            up_form = "A"
        elif move == "right":
            up_form = "B"
        for i in range(self.L_UC-1, self.i_ex, -1):
            A_u = self.psi_GS.get_B(i, form=up_form)
            W = self.H_MPO.get_W(i)
            A_d = self.psi_GS.get_B(i, form="B")
            theta_R = npc.tensordot(theta_R, A_u, axes=["vL", "vR"])
            theta_R = npc.tensordot(theta_R, W, axes=[["wL", "p"], ["wR", "p*"]])
            theta_R = npc.tensordot(theta_R, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        for i in range(0, self.i_ex):
            A_u = self.psi_GS.get_B(i, form=up_form)
            W = self.H_MPO.get_W(i)
            A_d = self.psi_GS.get_B(i, form="A")
            theta_L = npc.tensordot(theta_L, A_u, axes=["vR", "vL"])
            theta_L = npc.tensordot(theta_L, W, axes=[["wR", "p"], ["wL", "p*"]])
            theta_L = npc.tensordot(theta_L, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        theta_L = npc.tensordot(theta_L, self.psi_GS.get_B(self.i_ex, form=up_form), axes=["vR", "vL"])
        theta_L = npc.tensordot(theta_L, self.H_MPO.get_W(self.i_ex), axes=[["wR", "p"], ["wL", "p*"]])
        theta = npc.tensordot(theta_L, theta_R, axes=[["vR", "wR"], ["vL", "wL"]])
        return theta.replace_labels(["vR*", "vL*"], ["vL", "vR"])

    def Onsite_term(self, L_env, R_env, B):
        theta_R = R_env.copy()
        theta_L = L_env.copy()
        for i in range(self.L_UC-1, self.i_ex, -1):
            A_u = self.psi_GS.get_B(i, form="B")
            W = self.H_MPO.get_W(i)
            A_d = self.psi_GS.get_B(i, form="B")
            theta_R = npc.tensordot(theta_R, A_u, axes=["vL", "vR"])
            theta_R = npc.tensordot(theta_R, W, axes=[["wL", "p"], ["wR", "p*"]])
            theta_R = npc.tensordot(theta_R, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        for i in range(0, self.i_ex):
            A_u = self.psi_GS.get_B(i, form="A")
            W = self.H_MPO.get_W(i)
            A_d = self.psi_GS.get_B(i, form="A")
            theta_L = npc.tensordot(theta_L, A_u, axes=["vR", "vL"])
            theta_L = npc.tensordot(theta_L, W, axes=[["wR", "p"], ["wL", "p*"]])
            theta_L = npc.tensordot(theta_L, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])

        theta_L = npc.tensordot(theta_L, B, axes=["vR", "vL"])
        theta_L = npc.tensordot(theta_L, self.H_MPO.get_W(self.i_ex), axes=[["wR", "p"], ["wL", "p*"]])
        theta = npc.tensordot(theta_L, theta_R, axes=[["vR", "wR"], ["vL", "wL"]])
        return theta.replace_labels(["vR*", "vL*"], ["vL", "vR"])

    def get_B_env_GMRES(self, B, Lp, Rp, move="left", GMRES_options={"N_min": 5,
                                                                     "N_max": 30,
                                                                     "res": 1.e-13}):
        momentum = self.k
        # Conjugate B moves to left, right environment R_B should be calculated
        if move == "left":
            self.L_Transfer.B = B
            T_B_Rp = self.L_Transfer.Apply_TB(Rp, move="left")
            GMRES_solver = GMRES(A=self.L_Transfer, x=T_B_Rp, b=T_B_Rp, options=GMRES_options)
            R_B,  Diff_Ax_b_R, toterror_R, totiter_R = GMRES_solver.run()
            R_B = cmath.exp(+1.j*momentum) * R_B
            return R_B
        
        elif move == "right":
            self.R_Transfer.B = B
            T_B_Lp = self.R_Transfer.Apply_TB(Lp, move="right")
            GMRES_solver = GMRES(A=self.R_Transfer, x=T_B_Lp, b=T_B_Lp, options=GMRES_options)
            L_B, Diff_Ax_b_L, toterror_L, totiter_L  = GMRES_solver.run()
            L_B = cmath.exp(-1.j*momentum) * L_B
            return L_B

    def get_B_env_power(self, B, Lp, Rp, move="left", tol=1.e-10):
        """ Use a power iteration to get the converged B_environment."""
        momentum = self.k
        if move == "left":
            self.L_Transfer.B = B
            T_B_Rp = self.L_Transfer.Apply_TB(Rp, move="left")
            # T_B_Rp = L_Transfer_System.project(T_B_Rp, move="left")
            R_B = T_B_Rp.copy(deep=True)
            diff = npc.norm(T_B_Rp)
            while diff > tol: # Act from left, on T_B_Rp
                # We change to Apply_T_proj to calculate charge_0 sector
                T_B_Rp = cmath.exp(+1.j*momentum) * self.L_Transfer.Apply_T_Project(T_B_Rp, T_type="LR", transpose=False)
                R_B += T_B_Rp.copy(deep=True)
                diff = npc.norm(T_B_Rp)
                # print("R_diff", diff)
            R_B = cmath.exp(+1.j*momentum) * R_B
            return R_B

        elif move == "right":
            self.R_Transfer.B = B
            T_B_Lp = self.R_Transfer.Apply_TB(Lp, move="right")
            # T_B_Lp = R_Transfer_System.project(T_B_Lp, move="right")
            L_B = T_B_Lp.copy(deep=True)
            diff = npc.norm(T_B_Lp)
            while diff > tol: # Act from right, on T_B_Lp
                # We change to Apply_T_proj to calculate charge_0 sector
                T_B_Lp = cmath.exp(-1.j*momentum) * self.R_Transfer.Apply_T_Project(T_B_Lp, T_type="RL", transpose=True)
                L_B += T_B_Lp.copy(deep=True)
                diff = npc.norm(T_B_Lp)
                # print("L_diff", diff)
            L_B = cmath.exp(-1.j*momentum) * L_B
            return L_B

    def get_B_from_X(self, X):
        r"""Parameters:
        ----------
        X: :class:`Array`
        The tangent tensor's component which is projected to the complementary space of local tensor A_L
        
        Returns:
        ----------
        B: :class:`Array`
        The local tangent tensor, whose labels are ['vL', 'p', 'vr']"""
        B = npc.tensordot(self.N_L, X, axes=['vr', 'vl'])
        return B

    def proj_B_to_X(self, B):
        X = npc.tensordot(self.N_L.conj(), B, axes=[['vL*', 'p*'], ['vL', 'p']])
        X = X.replace_label(old_label='vr*', new_label='vl')
        return X
    
    def Normalize_Lp(self, Lp, Rp):
        C = self.psi_GS.get_C(0)
        overlap = npc.tensordot(Lp, C, axes=["vR", "vL"])
        overlap = npc.tensordot(overlap, C.conj(), axes=["vR*", "vL*"])
        overlap = npc.tensordot(overlap, Rp, axes=[["vR", "wR", "vR*"], ["vL", "wL", "vL*"]])
        Lp_new = Lp / overlap
        return Lp_new

    def test_result_B(self, X):
        X_p = self.matvec(X)
        E = npc.tensordot(X_p, X.conj(), axes=[['vl', 'vR'], ['vl*', 'vR*']])
        return E
    
    def Exi_Eng(self, B):
        pass


class Excited_MultisiteH(Excited_OnesiteH):
    # We choose the charge gauge that qtotal is on the 0th tensor
    def __init__(self, env, psi_GS, H, k, X_list, eigen_shift=0, group_sites=False, ex_charge=None):
        self.env = env
        self.FR = env.get_RP(psi_GS.L-1)
        self.FL = env.get_LP(0)
        self.psi_GS = psi_GS
        self.H_MPO = H
        self.IdL = H.get_IdL(0)
        self.IdR = H.get_IdR(psi_GS.L-1)
        self.L_UC = psi_GS.L
        if group_sites:
            self.Delta_x = 1
        else:
            self.Delta_x = self.L_UC
        self.k = k
        self.L_Ortho = []
        self.R_Ortho = []
        self.X_list = X_list
        self.X_list_com = []
        self.X_legs = []
        self.X_length = []
        self.X_slice = [0]
        self.E_shift = []
        self.eigen_shift = eigen_shift
        for i in range(len(X_list)):
            self.X_list_com.append(X_list[i].combine_legs(["vl", "vR"]))
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
        """ Note that here X is the concatenation of several X`s of a unit cell """
        # X should be the direct sum of the leg-combined X`s
        # Here we treat X as a vector
        X_old = []
        B_old = []
        X_new = []

        # To get X at each site of the UC from the direct sum of X`s
        # Extract the old X(B)`s to put them into the eigensolver
        for i in range(self.L_UC):
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
        L_B, R_B = self.get_B_env_GMRES(L_B_init, R_B_init)
        L_env_onsite, R_env_onsite = self.get_onsite_envs(B_old)

        # Get the new X from each site of the UC
        # Here the i index is out tensor B`s site
        for i in range(self.L_UC):
            B_i = self.Construct_H_chiral(self.FL, R_B, i_out=i, move="left")
            B_i += self.Construct_H_chiral(L_B, self.FR, i_out=i, move="right")

            B_i += self.Onsite_term(B_old[i], L_env_onsite[i], R_env_onsite[i], i_out=i)
            
            X_i = self.proj_B_to_X(B_i, i)
            # Shift the eigenvalue with the nonupdated X`s by self.eigen_shift to avoid eigenvalues near 0
            X_new.append(X_i+self.eigen_shift*X_old[i])
            
        self.X_list = copy.deepcopy(X_new)
        X = X_new[0].combine_legs(["vl", "vR"])
        for k in range(1, self.L_UC):
            X = npc.concatenate([X, X_new[k].combine_legs(["vl", "vR"])], axis=0)
        # print("H_eff_multi has X_list:{}".format(self.X_list))
        # print("X vec is:{}".format(X))
        return X
    
    def get_B_from_X(self, X, i):
        r"""Parameters:
        ----------
        X: :class:`Array`
        The tangent tensor's component which is projected to the complementary space of local tensor A_L
        
        Returns:
        ----------
        B: :class:`Array`
        The local tangent tensor, whose labels are ['vL', 'p', 'vr']"""
        B = npc.tensordot(self.L_Ortho[i], X, axes=['vr', 'vl'])
        return B
    
    def proj_B_to_X(self, B, i):
        X = npc.tensordot(self.L_Ortho[i].conj(), B, axes=[['vL*', 'p*'], ['vL', 'p']])
        X = X.replace_label(old_label='vr*', new_label='vl')
        return X
    
    def Init_B_env(self, B, Lp, Rp):
        # The method to initialize environment for constructing multi-site unit cell H_eff
        L = self.psi_GS.L

        R_B = npc.tensordot(B[L-1], Rp, axes=[["vR"], ["vL"]])
        R_B = npc.tensordot(R_B, self.H_MPO.get_W(L-1), axes=[["p", "wL"], ["p*", "wR"]])
        R_B = npc.tensordot(R_B, self.psi_GS.get_B(L-1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
        Rp_site = Rp
        for site in range(L-2, -1, -1):
            R_B = npc.tensordot(R_B, self.psi_GS.get_B(i=site, form="A"), axes=[["vL"], ["vR"]])
            R_B = npc.tensordot(R_B, self.H_MPO.get_W(site), axes=[["p", "wL"], ["p*", "wR"]])
            R_B = npc.tensordot(R_B, self.psi_GS.get_B(i=site, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i=site+1, form="B"), axes=[["vL"], ["vR"]])
            Rp_site = npc.tensordot(Rp_site, self.H_MPO.get_W(site+1), axes=[["p", "wL"], ["p*", "wR"]])
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i=site+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

            R_B_add = npc.tensordot(B[site], Rp_site, axes=[["vR"], ["vL"]])
            R_B_add = npc.tensordot(R_B_add, self.H_MPO.get_W(site), axes=[["p", "wL"], ["p*", "wR"]])
            R_B_add = npc.tensordot(R_B_add, self.psi_GS.get_B(site, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

            R_B += R_B_add

        L_B = npc.tensordot(B[0], Lp, axes=[["vL"], ["vR"]])
        L_B = npc.tensordot(L_B, self.H_MPO.get_W(0), axes=[["p", "wR"], ["p*", "wL"]])
        L_B = npc.tensordot(L_B, self.psi_GS.get_B(0, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
        Lp_site = Lp
        for site in range(1, L, +1):
            L_B = npc.tensordot(L_B, self.psi_GS.get_B(i=site, form="B"), axes=[["vR"], ["vL"]])
            L_B = npc.tensordot(L_B, self.H_MPO.get_W(site), axes=[["p", "wR"], ["p*", "wL"]])
            L_B = npc.tensordot(L_B, self.psi_GS.get_B(i=site, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            
            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i=site-1, form="A"), axes=[["vR"], ["vL"]])
            Lp_site = npc.tensordot(Lp_site, self.H_MPO.get_W(site-1), axes=[["p", "wR"], ["p*", "wL"]])
            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i=site-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            L_B_add = npc.tensordot(B[site], Lp_site, axes=[["vL"], ["vR"]])
            L_B_add = npc.tensordot(L_B_add, self.H_MPO.get_W(site), axes=[["p", "wR"], ["p*", "wL"]])
            L_B_add = npc.tensordot(L_B_add, self.psi_GS.get_B(site, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

            L_B += L_B_add

        return L_B, R_B
            
    def get_onsite_envs(self, B):
        '''
        This method will give the environments on a single unit cell, as the case the system is finite.
        Returns : L_env_onsite/R_env_onsite : 1 x L_UC list
        '''
        L = self.L_UC
        L_env_onsite = [None] * L
        R_env_onsite = [None] * L
        L_env_onsite[0] = self.FL
        R_env_onsite[L-1] = self.FR
        Lp_site = self.FL
        Rp_site = self.FR
        for site in range(1, L, +1):
            if site == 1:
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site-1], B[site-1], axes=[["vR"], ["vL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.H_MPO.get_W(site-1), axes=[["p", "wR"], ["p*", "wL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.psi_GS.get_B(site-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
            else:
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site-1], self.psi_GS.get_B(site-1, form="B"), axes=[["vR"], ["vL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.H_MPO.get_W(site-1), axes=[["p", "wR"], ["p*", "wL"]])
                L_env_onsite[site] = npc.tensordot(L_env_onsite[site], self.psi_GS.get_B(site-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

                L_B_add = npc.tensordot(B[site-1], Lp_site, axes=[["vL"], ["vR"]])
                L_B_add = npc.tensordot(L_B_add, self.H_MPO.get_W(site-1), axes=[["p", "wR"], ["p*", "wL"]])
                L_B_add = npc.tensordot(L_B_add, self.psi_GS.get_B(site-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])
                
                L_env_onsite[site] += L_B_add

            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i=site-1, form="A"), axes=[["vR"], ["vL"]])
            Lp_site = npc.tensordot(Lp_site, self.H_MPO.get_W(site-1), axes=[["p", "wR"], ["p*", "wL"]])
            Lp_site = npc.tensordot(Lp_site, self.psi_GS.get_B(i=site-1, form="A").conj(), axes=[["p", "vR*"], ["p*", "vL*"]])

        for site in range(L-2, -1, -1):
            if site == L-2:
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site+1], B[site+1], axes=[["vL"], ["vR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.H_MPO.get_W(site+1), axes=[["p", "wL"], ["p*", "wR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.psi_GS.get_B(site+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
            else:
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site+1], self.psi_GS.get_B(site+1, form="A"), axes=[["vL"], ["vR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.H_MPO.get_W(site+1), axes=[["p", "wL"], ["p*", "wR"]])
                R_env_onsite[site] = npc.tensordot(R_env_onsite[site], self.psi_GS.get_B(site+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

                R_B_add = npc.tensordot(B[site+1], Rp_site, axes=[["vR"], ["vL"]])
                R_B_add = npc.tensordot(R_B_add, self.H_MPO.get_W(site+1), axes=[["p", "wL"], ["p*", "wR"]])
                R_B_add = npc.tensordot(R_B_add, self.psi_GS.get_B(site+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])
                
                R_env_onsite[site] += R_B_add
                
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i=site+1, form="B"), axes=[["vL"], ["vR"]])
            Rp_site = npc.tensordot(Rp_site, self.H_MPO.get_W(site+1), axes=[["p", "wL"], ["p*", "wR"]])
            Rp_site = npc.tensordot(Rp_site, self.psi_GS.get_B(i=site+1, form="B").conj(), axes=[["p", "vL*"], ["p*", "vR*"]])

        return L_env_onsite, R_env_onsite
    
    def get_B_env_GMRES(self, L_B_init, R_B_init, GMRES_options={"N_min": 10,
                                                                 "N_max": 1000,
                                                                 "res": 1.e-10}):
        # L_Transfer means the transfer matrix has a left action on the right environment
        # Calculate R_B by `L_Transfer`
        momentum = self.k
        GMRES_solver_L = GMRES(A=self.L_Transfer, x=R_B_init, b=R_B_init, options=GMRES_options)
        R_B, Diff_Ax_b_R, toterror_R, totiter_R = GMRES_solver_L.run()
        R_B = cmath.exp(+1.j*momentum*self.Delta_x) * R_B

        GMRES_solver_R = GMRES(A=self.R_Transfer, x=L_B_init, b=L_B_init, options=GMRES_options)
        L_B, Diff_Ax_b_L, toterror_L, totiter_L  = GMRES_solver_R.run()
        L_B = cmath.exp(-1.j*momentum*self.Delta_x) * L_B

        # print("Check GMRES error L: {}".format(Diff_Ax_b_L))
        # print("Check GMRES error R: {}".format(Diff_Ax_b_R))
        return L_B, R_B
    
    def get_B_env_power(self, L_B_init, R_B_init, tol=1.e-10):
        momentum = self.k
        T_R_B = R_B_init
        R_B = R_B_init
        T_L_B = L_B_init
        L_B = L_B_init
        diff = npc.norm(T_R_B)
        while diff > tol:
            T_R_B = cmath.exp(+1.j*momentum*self.Delta_x) * self.L_Transfer.Apply_T_proj_out_dominant(T_R_B, T_type="LR", transpose=False)
            R_B += T_R_B.copy(deep=True)
            diff = npc.norm(T_R_B)
            print("R_diff", diff)
        R_B = cmath.exp(+1.j*momentum*self.Delta_x) * R_B

        diff = npc.norm(T_L_B)
        while diff > tol:
            T_L_B = cmath.exp(-1.j*momentum*self.Delta_x) * self.R_Transfer.Apply_T_proj_out_dominant(T_L_B, T_type="RL", transpose=True)
            L_B += T_L_B.copy(deep=True)
            diff = npc.norm(T_L_B)
            print("L_diff", diff)
        L_B = cmath.exp(-1.j*momentum*self.Delta_x) * L_B
        return L_B, R_B

    def Construct_H_chiral(self, L_env, R_env, i_out, move="left"):
        # Construct the H_chiral_with the left and right envs, MPO, and psi_GS
        """
        move : string
            The relative position of the `down` tensor B.conj() compared with the `up` tensor B
        """
        theta_R = R_env.copy()
        theta_L = L_env.copy()
        if move == "left":
            up_form = "A"
        elif move == "right":
            up_form = "B"
        for i in range(self.L_UC-1, i_out, -1):
            A_u = self.psi_GS.get_B(i, form=up_form)
            W = self.H_MPO.get_W(i)
            A_d = self.psi_GS.get_B(i, form="B")
            theta_R = npc.tensordot(theta_R, A_u, axes=["vL", "vR"])
            theta_R = npc.tensordot(theta_R, W, axes=[["wL", "p"], ["wR", "p*"]])
            theta_R = npc.tensordot(theta_R, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        for i in range(0, i_out):
            A_u = self.psi_GS.get_B(i, form=up_form)
            W = self.H_MPO.get_W(i)
            A_d = self.psi_GS.get_B(i, form="A")
            theta_L = npc.tensordot(theta_L, A_u, axes=["vR", "vL"])
            theta_L = npc.tensordot(theta_L, W, axes=[["wR", "p"], ["wL", "p*"]])
            theta_L = npc.tensordot(theta_L, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        theta_L = npc.tensordot(theta_L, self.psi_GS.get_B(i_out, form=up_form), axes=["vR", "vL"])
        theta_L = npc.tensordot(theta_L, self.H_MPO.get_W(i_out), axes=[["wR", "p"], ["wL", "p*"]])
        theta = npc.tensordot(theta_L, theta_R, axes=[["vR", "wR"], ["vL", "wL"]])
        return theta.replace_labels(["vR*", "vL*"], ["vL", "vR"])

    def Onsite_term(self, B_i, L_env, R_env, i_out):
        # Note that if the i_out is at the boundary, there will only be 2 terms
        # If i_out = 0, |L_B|A_R|R_p| term will vanish
        # If i_out = L-1, |L_p|A_L|R_B| term will vanish
        Lp = self.env.get_LP(i_out)
        Rp = self.env.get_RP(i_out)
        '''
        |L_B|A_R|R_p|
        '''
        B_out = npc.zeros(legcharges=B_i.legs, dtype=B_i.dtype, qtotal=B_i.qtotal, labels=B_i.get_leg_labels())
        if i_out != 0:
            B_out_l = npc.tensordot(self.psi_GS.get_B(i_out, form="B"), L_env, axes=[["vL"], ["vR"]])
            B_out_l = npc.tensordot(B_out_l, self.H_MPO.get_W(i_out), axes=[["wR", "p"], ["wL", "p*"]])
            B_out_l = npc.tensordot(B_out_l, Rp, axes=[["vR", "wR"], ["vL", "wL"]]).replace_labels(["vR*", "vL*"], ["vL", "vR"])
            B_out += B_out_l
        '''
        |L_p|A_L|R_B|
        '''
        if i_out != self.L_UC-1:
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
