import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mps_common import *
from tenpy.networks.mpo import *
from tenpy.linalg.krylov_based import LanczosGroundState
# from MPO_Env import *

# Global functions for bond dimension control
# Delta is set to 2 in this bond control algorithm since d of local Hilbert space is 3
def l_Ortho(psi, site):
    AL = psi.get_B(i=site, form='A')
    AL = AL.combine_legs(['vL', 'p'])
    NL = npc.orthogonal_columns(AL, new_label='vr') # (Dd x D(d-1))
    # print("NL's first leg is : {}".format(NL.get_leg(0)))
    # NL: ['(vL.p)', 'vr'] (Dd x D(d-1))
    # print("NL's leg:\n", NL)
    return NL

def r_Ortho(psi, site):
    AR = psi.get_B(i=site, form='B')
    AR = AR.combine_legs(['p', 'vR']).transpose(['(p.vR)', 'vL'])
    NR = npc.orthogonal_columns(AR, new_label='vl') # (Dd x D(d-1))
    NR = NR.transpose(['vl', '(p.vR)'])
    # NR: ['vl', '(p.vR)'] (D(d-1) x Dd)
    # print("NR's leg:\n", NR)
    return NR

def Calc_U_VH(psi, H, site):
    """ ====[AL(i)]--[C[i+1]]--[AR[i+1]]==== """
    """ init_lp = CalcLP(H, psi)
    init_rp = CalcRP(H, psi) """
    env = MPOEnvironment(psi, H, psi)
    Heff_A2C = TwoSiteH(env, i0=site, combine=False)
    A_L = psi.get_B(i=site, form='A').replace_label(old_label='p', new_label='p0')
    A_R = psi.get_B(i=psi._to_valid_index(site+1), form='B').replace_label(old_label='p', new_label='p1')
    C = psi._C[psi._to_valid_index(site+1)]
    # C = C[psi._to_valid_index(site+1)]
    A2C = npc.tensordot(A_L, C, axes=('vR', 'vL'))
    A2C = npc.tensordot(A2C, A_R, axes=('vR', 'vL'))
    A2C_prime = Heff_A2C.matvec(theta=A2C)
    A2C_prime = A2C_prime.combine_legs([('vL', 'p0'), ('p1', 'vR')]) 
    # print("A2C_Prime before project:", A2C_prime)
    NL = l_Ortho(psi, site)
    NR = r_Ortho(psi, psi._to_valid_index(site+1))
    #print("NL's qtotal is:{}".format(NL.qtotal))
    #print("NL is:{}".format(NL))
    #print("NL's leg's conjugate:{}".format(NL.conj().get_leg("vr*")))
    #print("NR's qtotal is:{}".format(NR.qtotal))
    #print("NR is:{}".format(NR))
    # print("Are the two legs contractible?", NL.get_leg("vr").test_contractible(NR.get_leg("vl")))
    A2C_prime = npc.tensordot(NL.conj(), A2C_prime, axes=('(vL*.p*)', '(vL.p0)'))
    A2C_prime = npc.tensordot(A2C_prime, NR.conj(), axes=('(p1.vR)', '(p*.vR*)'))
    # print("A2C_Prime after project:", A2C_prime)
    # print("The bond is:{}".format(psi._to_valid_index(site+1)))
    # print("NL:{}\n".format(NL))
    # print("NR:{}\n".format(NR))    
    # print("A2CPrime:{}\n".format(A2C_prime))
    """ if A2C_prime._data == []:
        U = npc.diag(s=1.+0.j, leg=NL.get_leg('vr').conj(), labels=['vr*', 'vrr'])    
        VH = npc.zeros(legcharges=[U.get_leg('vrr').conj(), NR.get_leg('vl').conj()], qtotal=A2C_prime.qtotal, labels=['vll', 'vl*'])
    else: """
    U, S, VH = npc.svd(A2C_prime, cutoff=None, inner_labels=['vrr', 'vll']) # Dim of U and VH are respectively (D(d-1) x Delta*d) and (Delat*d x D(d-1))
    # Here if cutoff is not set, Delta=1 since we set d=2
    # A cutoff is needed here.
    # U: ['vr*', 'vrr'] (D(d-1) x Delta*D), VH: ['vll', 'vl*'] (Delta*D x D(d-1)) 
    # U for expanding AL at site `site`, while VH for expanding AR at site `site+1`
    return U, VH, S, NL, NR

def Bond_Exp(psi, H):
    L = psi.L
    NLU = []
    VHNR = []
    S = []
    LegL = []
    LegR = []
    p_leg = psi.get_B(0).get_leg('p')
    # 1) Calculate orthonormal columns and orthonormal rows of AL and AR at each site, 
    #    save them and the legs which will be attached to the expanded tensors. 
    for i in range(L):
        U1, VH2, S_i, NL1, NR2 = Calc_U_VH(psi, H, i)
        NLU.append(npc.tensordot(NL1.split_legs(), U1, axes=('vr', 'vr*')))
        VHNR.append(npc.tensordot(VH2, NR2.split_legs(), axes=('vl*', 'vl')))
        S.append(S_i)
        LegL.append(U1.get_leg('vrr'))  # added right legcharge sector for i site
        LegR.append(VH2.get_leg('vll')) # added left legcharge sector for i+1 site

    # 2) Attach the NLU, VHNR and the zeros blocks to the original tensor AL, AR and C,
    #    Notice that the legcharges should be set correctly using LegL and LegR, 
    #    and conj() should be used to ensure contractibility.  
    for j in range(L):
        AL_Tilde = psi.get_B(j, form='A').transpose(['vL', 'p', 'vR'])
        AR_Tilde = psi.get_B(j, form='B').transpose(['vL', 'p', 'vR'])
        C_Tilde = psi._C[j]
        if AL_Tilde.get_leg_index('p') != 1:
            print("Please check the order of A's legs!\n")
        # Expand AL
        AL_Tilde = npc.concatenate([AL_Tilde, NLU[j]], axis=2, copy=True)
        zeros_L = npc.zeros([LegL[psi._to_valid_index(j-1)].conj(), p_leg, AL_Tilde.get_leg('vR')], qtotal=AL_Tilde.qtotal)
        _, AL_Tilde = npc.concatenate([AL_Tilde, zeros_L], axis=0, copy=True).sort_legcharge()
        # print("After expansion, AL:", AL_Tilde)
        # Expand AR
        AR_Tilde = npc.concatenate([AR_Tilde, VHNR[psi._to_valid_index(j-1)]], axis=0, copy=True)
        zeros_R = npc.zeros([AR_Tilde.get_leg('vL'), p_leg, LegR[j].conj()], qtotal=AR_Tilde.qtotal)
        _, AR_Tilde = npc.concatenate([AR_Tilde, zeros_R], axis=2, copy=True).sort_legcharge()
        # print("After expansion, AR:", AR_Tilde)
        # Expand C
        zeros_down = npc.zeros([LegL[psi._to_valid_index(j-1)].conj(), C_Tilde.get_leg('vR')], qtotal=C_Tilde.qtotal)
        zeros_right = npc.zeros([C_Tilde.get_leg('vL'), LegR[psi._to_valid_index(j-1)].conj()], qtotal=C_Tilde.qtotal)
        S_rd = npc.diag(s=S[psi._to_valid_index(j-1)], leg=LegL[psi._to_valid_index(j-1)].conj(), labels=['vll', 'vrr'])
        zeros_right = npc.concatenate([zeros_right, S_rd], axis=0, copy=True)
        C_Tilde = npc.concatenate([C_Tilde, zeros_down], axis=0, copy=True)
        _, C_Tilde = npc.concatenate([C_Tilde, zeros_right], axis=1, copy=True).sort_legcharge()
        
        # print("perm_C is just:{}".format(perm_C))
        # Expand _S since initialization of Environment needs it
        """ if C_Tilde.shape[0] != C_Tilde.shape[1]:
            raise ValueError('C must be square matrix after expansion!') """
        S_Tilde = np.concatenate((psi.get_SL(j), S[psi._to_valid_index(j-1)]), axis=0)

        Ac_Tilde = npc.tensordot(C_Tilde, AR_Tilde, axes=('vR', 'vL'))
        
        # Set new AL, AR, Ac and C
        psi.set_B(j, AL_Tilde, form='A')
        psi.set_B(j, AR_Tilde, form='B')
        psi.set_C(j, C_Tilde)
        psi.set_SL(j, S_Tilde)
        psi.set_Bc(j, Ac_Tilde)
    print("After bond expansion:\n")
    """ for k in range(L):
        print(psi._B[k]) """
    return psi
        
