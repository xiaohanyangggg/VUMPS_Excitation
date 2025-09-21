import numpy as np
import cmath
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPSEnvironment
from tenpy.networks.mpo import MPOEnvironment
from tenpy.linalg.krylov_based import GMRES
from VUMPS_and_Excitation.Algorithm.Single_Mode_Approximation.Exi_Transfer_System import Exi_Transfer_System_for_Obs, Exi_Transfer_System_MPS

def Local_1_site_obsevable(psi, site, op_str):
    """      ___    
        ----|_Ac|----
        |    _|_    |
        |   |_op|   |
        |    _|_    |        
        |___|Ac*|___| 
            i-site       """
    """ In the canonical form representation, LP = (1| and RP = |1) """
    site = psi._to_valid_index(site)
    Ac = psi.get_B(site, form='Th')
    p_site = psi.sites[site]
    op = p_site.get_op(op_str)
    Obs = npc.tensordot(Ac, op, axes=[['p'], ['p*']])
    Obs = npc.tensordot(Obs, Ac.conj(), axes=[['p', 'vL', 'vR'], ['p*', 'vL*', 'vR*']])
    return Obs

def Connected_Cor(psi, ops1, ops2, site_i, site_j):
    """ Calculate <\psi|op1_i, op2_j|\psi> - <\psi|op1_i|\psi><\psi|op2_j|\psi> """
    """ which is the correlation without the disconnected part """
    """
    Parameters
    ----------
    ops1: str
        Name of the operator1
    ops2: str
        Name of the operator2
    site_i: int
        Site of operator1
    site_j: int
        Site of operator2
    """
    OiOj = psi.correlation_function(ops1, ops2, sites1=[site_i], sites2=[site_j])[0][0]
    Oi = Local_1_site_obsevable(psi, site_i, ops1)
    Oj = Local_1_site_obsevable(psi, site_j, ops2)
    Cor_connected = OiOj - Oi * Oj
    return Cor_connected

def Calc_Q_L(psi, site):
    """ Calculate <Q_L> of the MPS left to site """
    S = psi.get_SL(site)
    print(S)
    Leg = psi.get_B(site, form='B').get_leg('vL')
    Q_L = 0.0
    for i in range(len(S)):
        print(S[i])
        U1_charge = Leg.get_charge(i)
        Q_L += (abs(S[i])**2) * U1_charge

    return Q_L 

class Observable_Exci_State:
    def __init__(self, psi_GS, Bs, momentum, site, op_str, i_op, group_sites, Kry_Para) -> None:
        self.psi_GS = psi_GS
        self.L_UC = psi_GS.L
        self.B_list = Bs
        self.site = site
        self.op_str = op_str
        self.i_op = i_op
        self.op = site.get_op(op_str)
        self.Kry_Para = Kry_Para
        self.k = momentum
        self.group_sites = group_sites

    def Local_Obs_Momentum_MPS(self):
        # Type 1
        R_BB = self.Generate_RBB()
        R_uB = self.Generate_uB()
        R_dB = self.Generate_dB()
        Transfer_LR = Exi_Transfer_System_for_Obs(psi=self.psi_GS, k=self.k, T_type="LR", group_sites=self.group_sites, B_side='up', transpose=False)
        Transfer_RL = Exi_Transfer_System_for_Obs(psi=self.psi_GS, k=self.k, T_type="RL", group_sites=self.group_sites, B_side='down', transpose=False)
        Sum_Tn_R_uB, _, _, _ = GMRES(A=Transfer_LR, x=R_uB, b=R_uB).run()
        Sum_Tn_R_dB, _, _, _ = GMRES(A=Transfer_RL, x=R_dB, b=R_dB).run()
        dB_R_uB = self.Apply_T_dB(Sum_Tn_R_uB, up_form='A') * cmath.exp(1.j*self.k)
        uB_R_dB = self.Apply_T_uB(Sum_Tn_R_dB, down_form='A') * cmath.exp(-1.j*self.k)
        R_BB += dB_R_uB
        R_BB += uB_R_dB
        Transfer_LL = Exi_Transfer_System_MPS(psi=self.psi_GS, k=0.0, T_type="LL", group_sites=self.group_sites, transpose=False)
        Sum_Tn_R_BB, _, _, _ = GMRES(A=Transfer_LL, x=R_BB, b=R_BB)
        for i in range(self.L_UC-1, -1, -1):
            Sum_Tn_R_BB = npc.tensordot(Sum_Tn_R_BB, self.psi_GS.get_B(i, form='A'), axes=[['vL'], ['vR']])
            if i == self.i_op:
                Sum_Tn_R_BB = npc.tensordot(Sum_Tn_R_BB, self.op, axes=[['p'], ['p*']])
            if i == 0:
                Sum_Tn_R_BB = npc.tensordot(Sum_Tn_R_BB, self.psi_GS.get_B(i, form='A').conj(), axes=[['vL', 'p', 'vL*'], ['vL*', 'p*', 'vR*']])        
            else:
                Sum_Tn_R_BB = npc.tensordot(Sum_Tn_R_BB, self.psi_GS.get_B(i, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
        Type1 = Sum_Tn_R_BB

        # Type 2
        """
        1.
         ----AL---        ----AL----                  ----B----
        |    |                |                           |    |
        |   |O|    \sum_n (   |   )^n exp(1.j*(n+1))      |    |
        |    |                |                           |    |
         ----B----        ----AR----                  ----AR---
         &

        2.
         ----B----        ----AR----                  ----AR---
        |    |                |                           |    |
        |   |O|    \sum_n (   |   )^n exp(-1.j*(n+1))     |    |
        |    |                |                           |    |
         ----AL---        ----AL----                  ----B----
        """
        Type2_1 = 0
        Type2_1_RB = Sum_Tn_R_uB * cmath.exp(1.j*self.k)
        for down_j in range(self.L_UC-1, self.i_op-1, -1):
            Type2_1_j = Type2_1_RB
            for k in range(self.L_UC-1, self.i_op-1, -1):
                Type2_1_j = npc.tensordot(Type2_1_j, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                if k == self.i_op:
                    Type2_1_j = npc.tensordot(Type2_1_j, self.op, axes=[['p'], ['p*']])
                    down_axes = [['vL', 'p', 'vL*'], ['vL*', 'p*', 'vR*']]
                else:
                    down_axes = [['p', 'vL*'], ['p*', 'vR*']]
                if k == down_j:
                    down_B = self.B_list[down_j]
                else:
                    down_B = self.psi_GS.get_B(k, form='B')
                Type2_1_j = npc.tensordot(Type2_1_j, down_B.conj(), axes=down_axes)
            Type2_1 += Type2_1_j
        
        Type2_2 = 0
        Type2_2_RB = Sum_Tn_R_dB * cmath.exp(-1.j*self.k)
        for up_i in range(self.L_UC-1, self.i_op-1, -1):
            Type2_2_i = Type2_2_RB
            for k in range(self.L_UC-1, self.i_op-1, -1):
                Type2_2_i = npc.tensordot(Type2_2_i, self.psi_GS.get_B(k, form='A').conj(), axes=[['vL*'], ['vR*']])
                if k == self.i_op:
                    Type2_2_i = npc.tensordot(Type2_2_i, self.op, axes=[['p*'], ['p']])
                    up_axes = [['vL', 'p*', 'vL*'], ['vR', 'p', 'vL']]
                else:
                    up_axes = [['vL', 'p*'], ['vR', 'p']]
                if k == up_i:
                    up_B = self.B_list[up_i]
                else:
                    up_B = self.psi_GS.get_B(k, form='B')
                    Type2_2_i = npc.tensordot(Type2_2_i, up_B, axes=up_axes)
            Type2_2 += Type2_2_i
        Type_2 = Type2_1 + Type2_2
        # Type 3
        """
         ----B----
        |    |    |  
        |   |O|   |  
        |    |    |
         ----B----
        """

        # Type 4
        """
         ----B----           ---AR---     ----AR----
        |    |                  |             |     |
        |    |      \sum_n  (   |     )^n    |O|    |
        |    |                  |             |     |
         ----B----           ---AR---     ----AR----
        """



    def Generate_RBB(self):

        R_BB = npc.tensordot(self.psi_GS.get_B(0, form='A'), self.psi_GS.get_B(0, form='A').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        L = len(self.B_list)

        for up_i in range(0, L):
            for down_j in range(0, up_i):
                R_BB_ij = npc.tensordot(self.B_list[up_i], self.psi_GS.get_B(up_i, form='B').conj(), axes=[['p', 'vR'], ['p*', 'vR*']])
                for k in range(up_i-1, down_j, -1):
                    R_BB_ij = npc.tensordot(R_BB_ij, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                    R_BB_ij = npc.tensordot(R_BB_ij, self.psi_GS.get_B(k, form='B').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
                R_BB_ij = npc.tensordot(R_BB_ij, self.psi_GS.get_B(down_j, form='A'), axes=[['vL'], ['vR']])
                R_BB_ij = npc.tensordot(R_BB_ij, self.B_list[down_j].conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
                for k in range(down_j-1, -1, -1):
                    R_BB_ij = npc.tensordot(R_BB_ij, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                    R_BB_ij = npc.tensordot(R_BB_ij, self.psi_GS.get_B(k, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
                R_BB += R_BB_ij

            R_BB_ii = npc.tensordot(self.B_list[up_i], self.B_list[up_i].conj(), axes=[['p', 'vR'], ['p*', 'vR*']])
            for k in range(up_i-1, -1, -1):
                R_BB_ii = npc.tensordot(R_BB_ij, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                R_BB_ii = npc.tensordot(R_BB_ij, self.psi_GS.get_B(k, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            R_BB += R_BB_ii

            for down_j in range(up_i+1, L):
                R_BB_ji = npc.tensordot(self.ps_GS.get_B(down_j, form='B'), self.B_list[down_j].conj(), axes=[['p', 'vR'], ['p*', 'vR*']])
                for k in range(down_j-1, up_i, -1):
                    R_BB_ji = npc.tensordot(R_BB_ji, self.psi_GS.get_B(k, form='B'), axes=[['vL'], ['vR']])
                    R_BB_ji = npc.tensordot(R_BB_ji, self.psi_GS.get_B(k, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
                R_BB_ji = npc.tensordot(R_BB_ji, self.B_list[up_i], axes=[['vL'], ['vR']])
                R_BB_ji = npc.tensordot(R_BB_ji, self.psi_GS.get_B(up_i, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
                for k in range(up_i-1, -1, -1):
                    R_BB_ji = npc.tensordot(R_BB_ji, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                    R_BB_ji = npc.tensordot(R_BB_ji, self.psi_GS.get_B(k, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
                R_BB += R_BB_ji
        return R_BB
    
    def Generate_uB(self): # `u` means that the B tensor at upper sites
        
        R_uB = npc.tensordot(self.psi_GS.get_B(0, form='Th'), self.psi_GS.get_B(0, form='B').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        L = len(self.B_list)

        for up_i in range(0, L):
            R_uB_i = npc.tensordot(self.B_list[up_i], self.psi_GS.get_B(up_i, form='B').conj(), axes=[['p', 'vR'], ['p*', 'vR*']])
            for k in range(up_i-1, -1, -1):
                R_uB_i = npc.tensordot(R_uB_i, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                R_uB_i = npc.tensordot(R_uB_i, self.psi_GS.get_B(k, form='B').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            R_uB += R_uB_i

        return R_uB
    
    def Generate_dB(self): # `d` means that the B tensor at down sites
        
        R_dB = npc.tensordot(self.psi_GS.get_B(0, form='B'), self.psi_GS.get_B(0, form='Th').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        L = len(self.B_list)

        for down_j in range(0, L):
            R_dB_j = npc.tensordot(self.psi_GS.get_B(down_j, form='B'), self.B_list[down_j].conj(), axes=[['p', 'vR'], ['p*', 'vR*']])
            for k in range(down_j-1, -1, -1):
                R_dB_j = npc.tensordot(R_dB_j, self.psi_GS.get_B(k, form='B'), axes=[['vL'], ['vR']])
                R_dB_j = npc.tensordot(R_dB_j, self.psi_GS.get_B(k, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            R_dB += R_dB_j

        return R_dB
    
    def Apply_T_dB(self, R_vec, up_form):
        if up_form == 'A':
            TuB_R_vec = npc.tensordot(self.psi_GS.get_B(0, form='A'), self.psi_GS.get_B(0, form='A').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        if up_form == 'B':
            TuB_R_vec = npc.tensordot(self.psi_GS.get_B(0, form='B'), self.psi_GS.get_B(0, form='Th').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        L = len(self.B_list)
        for down_j in range(0, L):
            T_Vec_i = R_vec.copy(deep=True)
            for k in range(L-1, down_j, -1):
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form=up_form), axes=[['vL'], ['vR']])
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form='B').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(down_j, form=up_form), ax=[['vL'], ['vR']])
            T_Vec_i = npc.tensordot(T_Vec_i, self.B_list[down_j].conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            for k in range(down_j-1, -1, -1):
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form=up_form), axes=[['vL'], ['vR']])
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form='A').conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            TuB_R_vec += T_Vec_i
        return TuB_R_vec

    def Apply_T_uB(self, R_vec, down_form):
        if down_form == 'A':
            TuB_R_vec = npc.tensordot(self.psi_GS.get_B(0, form='A'), self.psi_GS.get_B(0, form='A').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        if down_form == 'B':
            TuB_R_vec = npc.tensordot(self.psi_GS.get_B(0, form='Th'), self.psi_GS.get_B(0, form='B').conj(), axes=[['p', 'vR'], ['p*', 'vR*']]).zeros_like()
        L = len(self.B_list)
        for up_i in range(0, L):
            T_Vec_i = R_vec.copy(deep=True)
            for k in range(L-1, up_i, -1):
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form='B'), axes=[['vL'], ['vR']])
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form=down_form).conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            T_Vec_i = npc.tensordot(T_Vec_i, self.B_list[up_i], ax=[['vL'], ['vR']])
            T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(up_i, form=down_form).conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            for k in range(up_i-1, -1, -1):
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form='A'), axes=[['vL'], ['vR']])
                T_Vec_i = npc.tensordot(T_Vec_i, self.psi_GS.get_B(k, form=down_form).conj(), axes=[['p', 'vL*'], ['p*', 'vR*']])
            TuB_R_vec += T_Vec_i
        return TuB_R_vec

    