import numpy as np
import cmath
import functools
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPOEnvironment
from tenpy.linalg.sparse import FlatLinearOperator
from tenpy.linalg.krylov_based import Arnoldi

class Exi_Transfer_System():
    def __init__(self, psi, H, B, k, i_ex, FL, FR, group_sites=False, transpose=False):
        self.psi_GS = psi 
        self.B = B
        self.momentum = k
        self.L_UC = psi.L
        self.H_MPO = H
        self.i_ex = i_ex
        self.transpose = transpose

        if group_sites:
            self.Delta_x = 1
        else:
            self.Delta_x = self.L_UC

        self.IdL = H.get_IdL(0)
        self.IdR = H.get_IdR(psi.L-1)

        print("B qtotal:{}".format(B.qtotal))
        if len(B.qtotal) == 0:
            R_guess = None
            L_guess = None
        elif B.qtotal[0] == 0:
            R_guess = None
            L_guess = None
        elif B.qtotal[0] != 0 and B.qtotal is not None:
            R_legs = FR.legs
            L_legs = FL.legs
            R_labels = FR.get_leg_labels()
            L_labels = FL.get_leg_labels()
            R_guess = npc.Array.from_func(np.random.standard_normal, R_legs, dtype=FR.dtype, qtotal=B.qtotal, labels=R_labels)
            L_guess = npc.Array.from_func(np.random.standard_normal, L_legs, dtype=FL.dtype, qtotal=B.qtotal, labels=L_labels)
        
        
        self.T_LR_R = self.dominant_eigen(T_type="LR", transpose=False, guess=None)
        self.T_LR_L = self.dominant_eigen(T_type="LR", transpose=True, guess=None)
        self.T_RL_R = self.dominant_eigen(T_type="RL", transpose=False, guess=None)
        self.T_RL_L = self.dominant_eigen(T_type="RL", transpose=True, guess=None)
        
        # S_re = np.reciprocal(psi.get_SR(psi.L-1))
        # S_re_conj = S_re.conj()
        # print(S_re)
        # self.T_LR_R = npc.tensordot(psi.get_C(psi.L), FR, axes=["vR", "vL"])
        # self.T_LR_L = FL.scale_axis(s=S_re_conj, axis="vR*")
        # self.T_RL_R = npc.tensordot(psi.get_C(psi.L).conj(), FR, axes=["vR*", "vL*"])
        # self.T_RL_L = FL.scale_axis(s=S_re, axis="vR")

    def matvec(self, vec):
        """
        Apply `1-exp(jk*Delta_x)*QEQ` or `1-exp(-jk*Delta_x)*QEQ` to Q*vec
        But since the excitation env has qtotal 2, the overlap between it and the projector is 0
        """
        Labels = vec.get_leg_labels()
        if not self.transpose:
            T_vec = self.Apply_T_proj_out_dominant(vec, "LR", transpose=self.transpose)
            mat_vec = vec - cmath.exp(+1.j*self.momentum*self.Delta_x)*T_vec
        elif self.transpose:
            T_vec = self.Apply_T_proj_out_dominant(vec, "RL", transpose=self.transpose)
            mat_vec = vec - cmath.exp(-1.j*self.momentum*self.Delta_x)*T_vec
        # -- small test --
        """ IdL = self.H_MPO.get_IdL(0)
        wL = self.H_MPO.get_W(0).get_leg("wL")
        C0 = self.psi_GS.get_C(0)
        C0_conj = C0.conj()
        E_shift = C0.add_leg(wL, IdL, axis=1, label='wL').ireplace_labels(["vR"], ["vL*"])
        print(npc.norm(E_shift - self.Apply_T(E_shift)))
        assert npc.norm(E_shift - self.Apply_T(E_shift)) < 1e-7 """
        # ----------------
        return mat_vec.itranspose(Labels)

    def Apply_T(self, vec, T_type="LR", transpose=False):
        """
        Parameters
        ----------
        vec: :class:`Array`
             The input environment tensor
        T_type: str
             `LR` | `RL` | `LL` | `RR`
             Type of the transfermatrix
        transpose: bool
             The order of the left and right leg of T, determining which environment T applies on

        Returns
        -------
        vec: :class:`Array`
              The output environment which has been acted on
        """
        Labels = vec.get_leg_labels()
        if T_type == "LR":
            up_form = "A"
            down_form = "B"
        elif T_type == "RL":
            up_form = "B"
            down_form = "A"
        elif T_type == "LL":
            up_form = "A"
            down_form = "A"
        elif T_type == "RR":
            up_form = "B"
            down_form = "B"
        if not transpose:
            for i in range(self.L_UC-1, -1, -1):
                A_u = self.psi_GS.get_B(i, form=up_form)
                A_d = self.psi_GS.get_B(i, form=down_form)
                W = self.H_MPO.get_W(i)
                vec = npc.tensordot(vec, A_u, axes=["vL", "vR"])
                vec = npc.tensordot(vec, W, axes=[["wL", "p"], ["wR", "p*"]])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        elif transpose:
            for i in range(0, self.L_UC):
                A_u = self.psi_GS.get_B(i, form=up_form)
                A_d = self.psi_GS.get_B(i, form=down_form)
                W = self.H_MPO.get_W(i)
                vec = npc.tensordot(vec, A_u, axes=["vR", "vL"])
                vec = npc.tensordot(vec, W, axes=[["wR", "p"], ["wL", "p*"]])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        return vec.itranspose(Labels)

    def Apply_TB(self, vec, move="left"):
        if move == "left":
            down_form = "B"
        elif move == "right":
            down_form = "A"
        if not self.transpose:
            for i in range(self.L_UC-1, -1, -1):
                if i > self.i_ex:
                    A_u = self.psi_GS.get_B(i, form="B")
                if i == self.i_ex:
                    A_u = self.B
                if i < self.i_ex:
                    A_u = self.psi_GS.get_B(i, form="A")
                W = self.H_MPO.get_W(i)
                A_d = self.psi_GS.get_B(i, form=down_form)
                vec = npc.tensordot(vec, A_u, axes=["vL", "vR"])
                vec = npc.tensordot(vec, W, axes=[["wL", "p"], ["wR", "p*"]])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        elif self.transpose:
            for i in range(0, self.L_UC):
                if i < self.i_ex:
                    A_u = self.psi_GS.get_B(i, form="A")
                if i == self.i_ex:
                    A_u = self.B
                if i > self.i_ex:
                    A_u = self.psi_GS.get_B(i, form="B")
                W = self.H_MPO.get_W(i)
                A_d = self.psi_GS.get_B(i, form=down_form)
                vec = npc.tensordot(vec, A_u, axes=["vR", "vL"])
                vec = npc.tensordot(vec, W, axes=[["wR", "p"], ["wL", "p*"]])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        return vec

    def project(self, vec, move="left", T_type="LR", mpo_index="IdL"):
        """
        Project out additive energy part from vec.
        This method`s parameter may be modified since move=`left` means transpose=False and move=`right` means transpose=True
        But at this stage we just work with it
        """
        IdL = self.H_MPO.get_IdL(0)
        IdR = self.H_MPO.get_IdR(-1)
        wL = self.H_MPO.get_W(0).get_leg("wL")
        wR = self.H_MPO.get_W(self.L_UC-1).get_leg("wR")
        C0 = self.psi_GS.get_C(0)
        C0_conj = C0.conj()
        Labels = vec.get_leg_labels()

        if T_type == "LR":
            proj_L = C0_conj.replace_labels(["vL*"], ["vR"])
            proj_R = C0.replace_labels(["vR"], ["vL*"])
        elif T_type == "RL":
            proj_R = C0_conj.replace_labels(["vR*"], ["vL"])
            proj_L = C0.replace_labels(["vL"], ["vR*"])

        if mpo_index == "IdL":
            Id = IdL
        elif mpo_index == "IdR":
            Id = IdR

        if move == "left":  # In this case, we need to do projection of a right env, and transpose == False
            # `LR`, IdL
            proj_rho = proj_L.add_leg(wR, Id, axis=1, label='wR')
            E = npc.tensordot(proj_rho, vec, axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']])
            E_shift = proj_R.add_leg(wL, Id, axis=1, label='wL')
            # print("E is :{}".format(E))
            if len(E_shift.qtotal) != 0:
                if E_shift.qtotal[0] != vec.qtotal[0]:
                    return vec
            if isinstance(E, npc.Array):
                vec -= (npc.outer(E_shift, E)).itranspose(Labels)
            else:
                # print("Overlap of projector and env is : {}".format(E))
                vec -= (E_shift*E).itranspose(Labels)
                
        elif move == "right":  # In this case, we need to do projection of a left env, and transpose == True
            # `RL`, IdR
            proj_rho = proj_R.add_leg(wL, Id, axis=1, label='wL')
            E = npc.tensordot(proj_rho, vec, axes=[['vL', 'wL', 'vL*'], ['vR', 'wR', 'vR*']])
            E_shift = proj_L.add_leg(wR, Id, axis=1, label='wR')
            # print("E is :{}".format(E))
            if len(E_shift.qtotal) != 0:
                if E_shift.qtotal[0] != vec.qtotal[0]:
                    return vec
            if isinstance(E, npc.Array):
                vec -= (npc.outer(E_shift, E)).itranspose(Labels)
            else:
                # print("Overlap of projector and env is : {}".format(E))
                vec -= (E_shift*E).itranspose(Labels)
        else:
            print("The movement of B should be either left or right when we calculate environments containing B.\n")
        return vec.itranspose(Labels)
    
    # These two methods are for the projection of the dominant eigenvector of the T^R_L and T^L_R
    def Apply_T_Project(self, vec, T_type="LR", transpose=False):
        # By the way, notice the correspondence: move_left <==> LR, move_right <==> RL
        if not transpose:
            move = "left"
            mpo_index = "IdL"
        elif transpose:
            move = "right"
            mpo_index = "IdR"
        else:
            raise ValueError("Move should be either left or right\n")
        
        vec = self.Apply_T(vec, T_type, transpose)
        vec = self.project(vec, move, T_type, mpo_index)
        return vec
    
    def dominant_eigen_Arnoldi(self, T_type="LR", transpose=False, guess=None, **kwargs):
        return
    
    def dominant_eigen(self, T_type="LR", transpose=False, guess=None, **kwargs):
        """The method is constructed to calculate the left and right eigen of the modified LR and RL MPO transfer matrix."""
        IdR = self.H_MPO.IdR[0]
        IdL = self.H_MPO.IdL[0]
        wL = self.H_MPO.get_W(0).get_leg("wL")
        wR = self.H_MPO.get_W(-1).get_leg("wR")
        if not transpose:
            proj_norm = self.psi_GS.get_C(0).add_leg(wR, IdR, axis=1, label='wR')
        elif transpose:
            proj_norm = self.psi_GS.get_C(0).add_leg(wL, IdL, axis=1, label='wL')
        if not transpose:
            if guess is None:
                vR = self.psi_GS.get_B(self.L_UC-1, form="B").get_leg("vR")
                wL = self.H_MPO.get_W(0).get_leg('wL')
                eye_R = npc.diag(1., vR.conj(), dtype=self.psi_GS.dtype, labels=['vL', 'vL*'])
                guess_R = eye_R.add_leg(wL, self.IdR, axis=1, label='wL')
            else:
                guess_R = guess
            T_act = functools.partial(self.Apply_T_Project, T_type=T_type, transpose=transpose)
            flat_linop, _ = FlatLinearOperator.from_guess_with_pipe(T_act, v0_guess=guess_R, dtype=self.psi_GS.dtype)
            vals, vecs = flat_linop.eigenvectors(num_ev=1, **kwargs)
            print("Dominant eigenvalues at the charge sector: {}".format(vals))
            vec = vecs[0].split_legs()
            # Normalization
            vec = vec / npc.tensordot(proj_norm, vec, axes=[['vR', 'wR', 'vL'], ['vL', 'wL', 'vL*']])
            
        if transpose:
            if guess is None:
                vL = self.psi_GS.get_B(0, form="A").get_leg("vL")
                wR = self.H_MPO.get_W(-1).get_leg('wR')
                eye_L = npc.diag(1., vL.conj(), dtype=self.psi_GS.dtype, labels=['vR', 'vR*'])
                guess_L = eye_L.add_leg(wR, self.IdL, axis=1, label='wR')
            else:
                guess_L = guess
            T_act = functools.partial(self.Apply_T_Project, T_type=T_type, transpose=transpose)
            flat_linop, _ = FlatLinearOperator.from_guess_with_pipe(T_act, v0_guess=guess_L, dtype=self.psi_GS.dtype)
            vals, vecs = flat_linop.eigenvectors(num_ev=1, **kwargs)
            print("Dominant eigenvalues at the charge sector: {}".format(vals))
            vec = vecs[0].split_legs()
            # Normalization
            vec = vec / npc.tensordot(vec, proj_norm, axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vR']])
        # print("vals and vecs:{0}, {1}".format(vals, vecs))
        return vec
    
    def Apply_T_proj_out_dominant(self, vec, T_type="LR", transpose=False):
        Labels = vec.get_leg_labels()
        # This two dominant should be calculated and saved but not calculated every time we need it
        
        if T_type == "LR":
            R_eig_vec = self.T_LR_R
            L_eig_vec = self.T_LR_L
        elif T_type == "RL":
            R_eig_vec = self.T_RL_R
            L_eig_vec = self.T_RL_L

        vec = self.Apply_T(vec, T_type=T_type, transpose=transpose)
        
        if not transpose:
            if len(R_eig_vec.qtotal) == 0:
                overlap = npc.tensordot(L_eig_vec, vec, axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']])
                vec -= overlap * R_eig_vec.itranspose(Labels)
            elif R_eig_vec.qtotal[0] == vec.qtotal[0]:
                overlap = npc.tensordot(L_eig_vec, vec, axes=[['vR', 'wR', 'vR*'], ['vL', 'wL', 'vL*']])
                vec -= overlap * R_eig_vec.itranspose(Labels)
        elif transpose:
            if len(L_eig_vec.qtotal) == 0:
                overlap = npc.tensordot(R_eig_vec, vec, axes=[['vL', 'wL', 'vL*'], ['vR', 'wR', 'vR*']])
                vec -= overlap * L_eig_vec.itranspose(Labels)
            elif L_eig_vec.qtotal[0] == vec.qtotal[0]:
                overlap = npc.tensordot(R_eig_vec, vec, axes=[['vL', 'wL', 'vL*'], ['vR', 'wR', 'vR*']])
                vec -= overlap * L_eig_vec.itranspose(Labels)
        return vec


class Exi_Transfer_System_MPS():
    def __init__(self, psi, k, T_type="LL", group_sites=True, transpose=False) -> None:
        self.psi_GS = psi
        self.L_UC = psi.L
        self.momentum = k
        self.T_type = T_type
        self.transpose = transpose

        if group_sites:
            self.Delta_x = 1
        if not group_sites:
            self.Delta_x = psi.L

        if T_type == "LL":
            self.L_dominant = npc.diag(1., leg=psi.get_B(0, form="A").get_leg("vL").conj(), dtype=psi.dtype, labels=["vR", "vR*"])
            self.R_dominant = npc.tensordot(psi.get_C(0), psi.get_C(0).conj(), axes=[['vR'], ['vR*']])
        if T_type == "RR":
            self.L_dominant = npc.tensordot(psi.get_C(0), psi.get_C(0).conj(), axes=[['vR'], ['vR*']])
            self.R_dominant = npc.diag(1., leg=psi.get_B(self.L_UC-1, form="B").get_leg('vR').conj(), dtype=psi.dtype, labels=['vL', 'vL*'])
        if T_type == "LR":
            self.L_dominant = psi.get_C(0).conj().replace_labels(['vL*'], ['vR'])
            self.R_dominant = psi.get_C(0).replace_labels(['vR'], ['vL*'])
        if T_type == "RL":
            self.L_dominant = psi.get_C(0).replace_labels(["vL"], ["vR*"])
            self.R_dominant = psi.get_C(0).conj().replace_labels(['vR*'], ['vL'])

    def matvec(self, vec):
        Labels = vec.get_leg_labels()
        if not self.transpose:
            T_vec = self.Apply_T_proj_out_dominant(vec)
            mat_vec = vec - cmath.exp(+1.j*self.momentum*self.Delta_x)*T_vec
        elif self.transpose:
            T_vec = self.Apply_T_proj_out_dominant(vec)
            mat_vec = vec - cmath.exp(-1.j*self.momentum*self.Delta_x)*T_vec

        return mat_vec.itranspose(Labels)
    
    def Apply_T(self, vec):
        Labels = vec.get_leg_labels()
        if self.T_type == "LR":
            up_form = "A"
            down_form = "B"
        elif self.T_type == "RL":
            up_form = "B"
            down_form = "A"
        elif self.T_type == "LL":
            up_form = "A"
            down_form = "A"
        elif self.T_type == "RR":
            up_form = "B"
            down_form = "B"
        if not self.transpose:
            for i in range(self.L_UC-1, -1, -1):
                A_u = self.psi_GS.get_B(i, form=up_form)
                A_d = self.psi_GS.get_B(i, form=down_form)
                vec = npc.tensordot(vec, A_u, axes=["vL", "vR"])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vL*", "p"], ["vR*", "p*"]])
        elif self.transpose:
            for i in range(0, self.L_UC):
                A_u = self.psi_GS.get_B(i, form=up_form)
                A_d = self.psi_GS.get_B(i, form=down_form)
                vec = npc.tensordot(vec, A_u, axes=["vR", "vL"])
                vec = npc.tensordot(vec, A_d.conj(), axes=[["vR*", "p"], ["vL*", "p*"]])
        return vec.itranspose(Labels)
    
    def Apply_T_proj_out_dominant(self, vec):
        Labels = vec.get_leg_labels()
        # This two dominant should be calculated and saved but not calculated every time we need it
    
        vec = self.Apply_T(vec)
        
        if not self.transpose:
            if len(self.R_dominant.qtotal) == 0:
                overlap = npc.tensordot(self.L_dominant, vec, axes=[['vR', 'vR*'], ['vL', 'vL*']])
                vec -= overlap * self.R_dominant.itranspose(Labels)
            elif self.R_dominant.qtotal[0] == vec.qtotal[0]:
                overlap = npc.tensordot(self.L_dominant, vec, axes=[['vR', 'vR*'], ['vL', 'vL*']])
                vec -= overlap * self.R_dominant.itranspose(Labels)
        elif self.transpose:
            if len(self.L_dominant.qtotal) == 0:
                overlap = npc.tensordot(self.R_dominant, vec, axes=[['vL', 'vL*'], ['vR', 'vR*']])
                vec -= overlap * self.L_dominant.itranspose(Labels)
            elif self.L_dominant.qtotal[0] == vec.qtotal[0]:
                overlap = npc.tensordot(self.R_dominant, vec, axes=[['vL', 'vL*'], ['vR', 'vR*']])
                vec -= overlap * self.L_dominant.itranspose(Labels)
        return vec.itranspose(Labels)

    
class Exi_Transfer_System_for_Obs(Exi_Transfer_System_MPS):
    """
        Parameters
        ----------
        vec: :class:`Array`
            The input environment tensor
        B_side: :str
            `up` | `down`
            The position of B tensor in the right intial vector.
            This parameter determines the sign of momentum k in matvec() method.
    """
    def __init__(self, psi, k, T_type="LL", group_sites=True, B_side='up', transpose=False):
        super.__init__(psi, k, T_type="LL", group_sites=True, transpose=False)
        self.B_side = B_side

    def matvec(self, vec):
        if self.B_side == 'up':
            k = self.momentum
        if self.B_side == 'down':
            k = -self.momentum
            
        Labels = vec.get_leg_labels()
        if not self.transpose:
            T_vec = self.Apply_T_proj_out_dominant(vec)
            mat_vec = vec - cmath.exp(+1.j*k*self.Delta_x)*T_vec
        elif self.transpose:
            T_vec = self.Apply_T_proj_out_dominant(vec)
            mat_vec = vec - cmath.exp(-1.j*k*self.Delta_x)*T_vec

        return mat_vec.itranspose(Labels)
