import numpy as np
import warnings
import tenpy.linalg.np_conserved as npc
from ..Linalg.GMRES_NPC import GMRES
# from tenpy.linalg.lanczos import GMRES
from tenpy.linalg.sparse import NpcLinearOperatorWrapper, FlatLinearOperator
import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.WARNING)
# from GMRES_NPC import GMRES

# Get the so-called quasi-environment with the algorithm in arXiv:1701.07035
# 1-site mpo case

# These two classes are needed in GMRES solution of aenvironments
class L_Env_NPCLinearOperator:
    """
    AL and W_ii can be lists of tensors(TODO) to make multi-sites algorithm possible
    """
    def __init__(self, AL, W_ii, l0, r0, project=False):
        self.AL = AL
        self.W_ii = W_ii
        self.project = project
        self.l0 = l0
        self.r0 = r0

    def matvec(self, v):
        delta_v = npc.tensordot(v, self.AL, axes=['vR', 'vL'])
        delta_v = npc.tensordot(delta_v, self.W_ii, axes=[['p'], ['p*']])
        delta_v = npc.tensordot(delta_v, self.AL.conj(), axes=[['vR*', 'p'] ,['vL*', 'p*']])
        v_labels = delta_v.get_leg_labels()
        delta_v.itranspose(v_labels)
        if self.project == False:
            v = v - delta_v
            return v
        elif self.project == True:
            delta_v_p = self.l0 * npc.tensordot(self.r0, v, axes=[['vL', 'vL*'], ['vR', 'vR*']])
            delta_v_p.itranspose(v_labels)
            v = v - delta_v + delta_v_p
            return v


class R_Env_NPCLinearOperator:
    def __init__(self, AR, W_ii, l0, r0, project=False):
        self.AR = AR
        self.W_ii = W_ii
        self.project = project
        self.l0 = l0
        self.r0 = r0

    def matvec(self, v):
        delta_v = npc.tensordot(v, self.AR, axes=['vL', 'vR'])
        delta_v = npc.tensordot(delta_v, self.W_ii, axes=[['p'], ['p*']])
        delta_v = npc.tensordot(delta_v, self.AR.conj(), axes=[['vL*', 'p'] ,['vR*', 'p*']])
        v_labels = delta_v.get_leg_labels()
        delta_v.itranspose(v_labels)
        
        if self.project == False:
            v = v - delta_v
            return v
        elif self.project == True:
            delta_v_p = self.r0 * npc.tensordot(self.l0, v, axes=[['vR', 'vR*'], ['vL', 'vL*']])
            delta_v_p.itranspose(v_labels)
            v = v - delta_v + delta_v_p
            return v


# This two classes are needed for the |1)(R| and |L)(1| projection operators
class L_TransferMatrix:
    # Transfer matrix with left action, which means we apply the TM from left
    def __init__(self, Au, Ad):
        self.Au = Au
        self.Ad = Ad

    def matvec(self, v):
        v = v.split_legs()
        v = npc.tensordot(v, self.Au, axes=[['vL'], ['vR']])
        v = npc.tensordot(v, self.Ad.conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])
        return v.combine_legs([0, 1])

class R_TransferMatrix:
    # Transfer matrix with right action, which means we apply the TM from right
    def __init__(self, Au, Ad):
        self.Au = Au
        self.Ad = Ad

    def matvec(self, v):
        v = v.split_legs()
        v = npc.tensordot(v, self.Au, axes=[['vR'], ['vL']])
        v = npc.tensordot(v, self.Ad.conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
        return v.combine_legs([1, 0])


# These two methods are needed to apply transfer matrix with specific MPO entries
def l_transfer(LP, AL, W_ij):
    LP_T = npc.tensordot(LP, AL, axes=['vR', 'vL'])
    LP_T = npc.tensordot(LP_T, W_ij, axes=[['p'], ['p*']])
    LP_T = npc.tensordot(LP_T, AL.conj(), axes=[['vR*', 'p'] ,['vL*', 'p*']])
    return LP_T

def r_transfer(RP, AR, W_ij):
    T_RP = npc.tensordot(AR, RP, axes=['vR', 'vL'])
    T_RP = npc.tensordot(W_ij, T_RP, axes=[['p*'], ['p']])
    T_RP = npc.tensordot(AR.conj(), T_RP, axes=[['vR*', 'p*'] ,['vL*', 'p']])
    return T_RP


def l_env_w(psi, H_MPO, LP_W_init):
    W = H_MPO.get_W(0)
    A_L = psi.get_B(0, form='A')

    chi_W = W.get_leg('wL').ind_len
    LP_W = LP_W_init.zeros_like()
    L_Slice_Labels = LP_W[0, :, :].get_leg_labels()

    LP_0 = npc.diag(s=1., leg=A_L.get_leg('vL'), dtype=H_MPO.dtype, labels=['vR*', 'vR'])
    RP_0 = npc.diag(s=1., leg=A_L.get_leg('vL'), dtype=H_MPO.dtype, labels=['vL', 'vL*'])
    RP_0_com = RP_0.copy().combine_legs([0, 1])
    Transfer = L_TransferMatrix(A_L, A_L) # Since we want the right eigenvector, TM should act from left side.
    Transfer_OP = FlatLinearOperator(Transfer.matvec, leg=RP_0_com.get_leg(0), dtype=W.dtype, charge_sector=0, vec_label='(vL.vL*)')
    vals0, R0 = Transfer_OP.eigenvectors(num_ev=1, max_tol=1.e-20, which='LM', hermitian=False)
    
    # Check:
    if vals0[0] - 1. > 1.e-7:
        logger.warning("Dominant eigenvalue of TM should be 1.0, but we get:{}".format(vals0[0]))
    RP_0 = R0[0]
    RP_0 = RP_0.split_legs()
    LP_0 = LP_0.transpose(L_Slice_Labels)
    for i in range(0, chi_W):
        # print("i:{}\n".format(i))
        # Every time we need to re-calculate YL_i
        YL_i = LP_0.zeros_like()
        W_i = W.take_slice(indices=[i], axes=['wR'])
        if i == 0:
            YL_i = LP_0
            LP_W[i, :, :] = YL_i
        else:
            for j in range(0, i):
                YL_i += l_transfer(LP_W[j, :, :], A_L, W_i.take_slice(indices=[j], axes=['wL'])).transpose(L_Slice_Labels)
            if i == chi_W - 1:
                Env_Linear_OP = L_Env_NPCLinearOperator(A_L, W_ii = W_i.take_slice(indices=[i], axes=['wL']), l0=LP_0, r0=RP_0, project=True)
                YL_i -= LP_0 * npc.tensordot(RP_0, YL_i, axes=[['vL', 'vL*'], ['vR', 'vR*']])
            else:
                Env_Linear_OP = L_Env_NPCLinearOperator(A_L, W_ii = W_i.take_slice(indices=[i], axes=['wL']), l0=LP_0, r0=RP_0, project=False)
            GMRES_solver = GMRES(Env_Linear_OP, YL_i, options={'E_tol': 1.e-15, 'P_tol': 1.e-20, 'N_min': 5, 'N_max': 50, 'reortho': True})
            # print("YL_i is:{}".format(YL_i.to_ndarray()))
            x = GMRES_solver.run()
            # print("X is:{}".format(x.to_ndarray()))
            # print("=============================================")
            LP_W[i, :, :] = x.split_legs()
    return LP_W            


def r_env_w(psi, H_MPO, RP_W_init):
    W = H_MPO.get_W(0)
    # print("W has dtype:{}".format(W.dtype))
    A_R = psi.get_B(0, form='B')

    chi_W = W.get_leg('wR').ind_len
    RP_W = RP_W_init.zeros_like()
    R_Slice_Labels = RP_W[0, :, :].get_leg_labels()

    RP_0 = npc.diag(s=1., leg=A_R.get_leg('vR').conj(), dtype=H_MPO.dtype, labels=['vL', 'vL*'])
    LP_0 = npc.diag(s=1., leg=A_R.get_leg('vR').conj(), dtype=H_MPO.dtype, labels=['vR*', 'vR'])
    LP_0_com = LP_0.copy().combine_legs([0, 1])
    Transfer = R_TransferMatrix(A_R, A_R)
    Transfer_OP = FlatLinearOperator(Transfer.matvec, leg=LP_0_com.conj().get_leg(0), dtype=W.dtype, charge_sector=0, vec_label='(vR*.vR)')
    vals0, L0 = Transfer_OP.eigenvectors(num_ev=1, max_tol=1.e-20, which='LM', hermitian=False)
    # Check:
    if vals0[0] - 1. > 1.e-7:
        logger.warning("Dominant eigenvalue of TM should be 1.0, but we get:{}".format(vals0[0]))
    LP_0 = L0[0]
    LP_0 = LP_0.split_legs()
    RP_0 = RP_0.transpose(R_Slice_Labels)
    for i in range(chi_W-1, -1, -1):
        print("i is:{}".format(i))
        YR_i = RP_0.zeros_like()
        W_i = W.take_slice(indices=[i], axes=['wL'])
        if i == chi_W - 1:
            YR_i = RP_0
            RP_W[i, :, :] = YR_i
        else:
            for j in range(chi_W-1, i, -1):
                YR_i += r_transfer(RP_W[j, :, :], A_R, W_i.take_slice(indices=[j], axes=['wR'])).transpose(R_Slice_Labels)
            if i == 0:
                Env_Linear_OP = R_Env_NPCLinearOperator(A_R, W_ii = W_i.take_slice(indices=[i], axes=['wR']), l0=LP_0, r0=RP_0, project=True)
                YR_i -= RP_0 * npc.tensordot(LP_0, YR_i, axes=[['vR', 'vR*'], ['vL', 'vL*']])
            else:
                Env_Linear_OP = R_Env_NPCLinearOperator(A_R, W_ii = W_i.take_slice(indices=[i], axes=['wR']), l0=LP_0, r0=RP_0, project=False)
            print("YR_i is:{}".format(YR_i.to_ndarray()))
            GMRES_solver = GMRES(Env_Linear_OP, YR_i, options={'E_tol': 1.e-15, 'P_tol': 1.e-20,'N_min': 5, 'N_max': 50, 'reortho': True})
            x = GMRES_solver.run()
            print("X is:{}".format(x.to_ndarray()))
            RP_W[i, :, :] = x.split_legs()
            print("=============================================")
    return RP_W

        
