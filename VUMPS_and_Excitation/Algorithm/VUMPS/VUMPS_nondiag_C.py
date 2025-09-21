import numpy as np
import logging
import tenpy.linalg.np_conserved as npc
import matplotlib.pyplot as plt
from tenpy.networks.mpo import MPO, MPOEnvironment, MPOTransferMatrix
from tenpy.models.spins import *
from tenpy.algorithms.mps_common import *
from tenpy.networks.site import BosonSite
from tenpy.networks.mps import *
from tenpy.models.hofstadter import HofstadterBosons
from tenpy.linalg.charges import *
from tenpy.linalg.krylov_based import Arnoldi, LanczosGroundState
from tenpy.linalg.sparse import FlatLinearOperator
from tenpy.tools.math import entropy
from ...System.mps_Replica import MPS_Replica
from ...System.Bond_Control_RE_RE import Bond_Exp, l_Ortho, r_Ortho
from ...Environment.quasi_environment import l_env_w, r_env_w
# from Bond_Control import Bond_Expansion
# from MPO_Env import *
import sys

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.WARNING)

def Contract_Tranfer_Matrix(theta, psi, H, i, up='A', down='A', ActFrom='left'):
    A_u = psi.get_B(i, form=up)
    A_d = psi.get_B(i, form=down)
    W = H.get_W(i)
    if ActFrom == 'left':
        theta = npc.tensordot(theta, A_u, axes=['vL', 'vR'])
        theta = npc.tensordot(theta, W, axes=[['wL', 'p'], ['wR', 'p*']])
        theta = npc.tensordot(theta, A_d.conj(), axes=[['vL*', 'p'], ['vR*', 'p*']])
    if ActFrom == 'right':
        theta = npc.tensordot(theta, A_u, axes=['vR', 'vL'])
        theta = npc.tensordot(theta, W, axes=[['wR', 'p'], ['wL', 'p*']])
        theta = npc.tensordot(theta, A_d.conj(), axes=[['vR*', 'p'], ['vL*', 'p*']])
    return theta

def Cal_Galerkin_L(psi, H_eff_Ac, loc):
    Ac_p = H_eff_Ac[loc].matvec(psi._Bc[loc].replace_label('p', 'p0')).replace_label('p0', 'p')
    Labels = Ac_p.get_leg_labels()
    AL_C_p = npc.tensordot(Ac_p, psi.get_B(loc, form='A').conj(), axes=[['vL', 'p'], ['vL*', 'p*']])
    AL_C_p = npc.tensordot(psi.get_B(loc, form='A'), AL_C_p, axes=['vR', 'vR*']).itranspose(Labels)
    return npc.norm(Ac_p - AL_C_p)

def Cal_Galerkin_L_alt(psi, H_eff_Ac, loc):
    # print("Ac_bef:{}".format(psi._Bc[loc].combine_legs(['vL', 'p']).to_ndarray()))
    Ac_p = H_eff_Ac[loc].matvec(psi._Bc[loc].replace_label('p', 'p0')).replace_label('p0', 'p')
    # print("|Ac_p|:{}".format(npc.norm(Ac_p)))
    Ac_p = - Ac_p / npc.norm(Ac_p)
    
    # print("Ac_aft:{}".format(Ac_p.to_ndarray()))
    N_l = l_Ortho(psi, loc).split_legs()
    # print("Ac_prime is:{}".format(Ac_p.to_ndarray()))
    # print("Ac is:{}".format(psi.get_Bc(loc).to_ndarray()))
    # print("A_c_prime - A_c = {}".format((Ac_p - psi.get_Bc(loc)).to_ndarray()))
    # print("||A_c_prime - A_c|| = {}".format(npc.norm(Ac_p - psi.get_Bc(loc))))
    # print("Nl is:{}".format(N_l.to_ndarray()))
    # print("Check Orho:{}".format(npc.norm(npc.tensordot(psi.get_Bc(loc), N_l.conj(), axes=[['vL', 'p'], ['vL*', 'p*']]))))
    # print("Check Orho_prime:{}".format(npc.tensordot(Ac_p, N_l.conj(), axes=[['vL', 'p'], ['vL*', 'p*']]).to_ndarray()))
    return npc.norm(npc.tensordot(Ac_p, N_l.conj(), axes=[['vL', 'p'], ['vL*', 'p*']]))

def Cal_Galerkin_L_alt_alt(psi, H_eff_Acc, loc):
    Ac_p = H_eff_Acc[loc].matvec(npc.tensordot(psi.get_B(loc, form='A'), psi.get_C(loc+1), axes=['vR', 'vL']).replace_label('p', 'p0')).replace_label('p0', 'p')
    N_l = l_Ortho(psi, loc).split_legs()
    return npc.norm(npc.tensordot(Ac_p, N_l.conj(), axes=[['vL', 'p'], ['vL*', 'p*']]))
    
def Cal_Galerkin_R(psi, H_eff_Ac, loc):
    Ac_p = H_eff_Ac[loc].matvec(psi._Bc[loc].replace_label('p', 'p0')).replace_label('p0', 'p')
    Labels = Ac_p.get_leg_labels()
    C_AR_p = npc.tensordot(Ac_p, psi.get_B(loc, form='B').conj(), axes=[['vR', 'p'], ['vR*', 'p*']])
    C_AR_p = npc.tensordot(psi.get_B(loc, form='B'), C_AR_p, axes=['vL', 'vL*']).itranspose(Labels)
    return npc.norm(Ac_p - C_AR_p)

def Cal_Galerkin_R_alt(psi, H_eff_Ac, loc):
    Ac_p = H_eff_Ac[loc].matvec(psi._Bc[loc].replace_label('p', 'p0')).replace_label('p0', 'p')
    Ac_p = - Ac_p / npc.norm(Ac_p)
    N_r = r_Ortho(psi, loc).split_legs()
    return npc.norm(npc.tensordot(Ac_p, N_r.conj(), axes=[['p', 'vR'], ['p*', 'vR*']]))

def update_env(psi, H, old_env_data):
    """ LP_slice = old_env_data['init_LP'][0, :, :]
    RP_slice = old_env_data['init_RP'][-1, :, :]
    print("Left environment has labels:{}".format(old_env_data['init_LP'].get_leg_labels()))
    print("Right environment has labels:{}".format(old_env_data['init_RP'].get_leg_labels()))
    print("Fix left environment`s mpo leg 0 : {}".format(LP_slice.to_ndarray()))
    print("Fix right environment`s mpo leg chi : {}".format(RP_slice.to_ndarray())) """
    boundary_env_data = MPOTransferMatrix.find_init_LP_RP(H, psi, calc_E=False, guess_init_env_data=old_env_data, cutoff=1.e-20, max_tol=1.e-15)
    env_new = MPOEnvironment(psi, H, psi, **boundary_env_data)
    R = npc.tensordot(psi.get_C(0), psi.get_C(0).conj(), axes=['vR', 'vR*'])
    print("Check zero left energy:{}".format(npc.tensordot(old_env_data['init_LP'][-1, :, :], R, axes=[['vR', 'vR*'], ['vL', 'vL*']])))
    return env_new

def update_env_GMRES(psi, H, old_env_data):
    LP_W_init = old_env_data['init_LP']
    LP_W_new = l_env_w(psi, H, LP_W_init)
    RP_W_init = old_env_data['init_RP']
    RP_W_new = r_env_w(psi, H, RP_W_init) 
    new_env_data = {
        'init_LP': LP_W_new,
        'init_RP': RP_W_new
    }
    env_new = MPOEnvironment(psi, H, psi, **new_env_data)
    return env_new

def update_env_Arnoldi(psi, H, old_env_data, Krylov_Para):
    MPO_TM_AR = MPOTransferMatrix(H, psi, transpose=False) # right to left
    MPO_TM_AL = MPOTransferMatrix(H, psi, transpose=True)  # left to right
    C = psi.get_C(0)
    wL = H.get_W(0).get_leg('wL')
    wR = wL.conj()
    rho_R = npc.tensordot(C, C.conj(), axes=['vL', 'vL*']) # ['vR', 'vR*']
    rho_L = npc.tensordot(C.conj(), C, axes=['vR*', 'vR']) # ['vL*', 'vL']
    proj_rho_R = rho_R.add_leg(wR, MPO_TM_AR.IdL, axis=1, label='wR')
    proj_rho_L = rho_L.add_leg(wL, MPO_TM_AL.IdR, axis=1, label='wL')
    MPO_TM_AR._proj_rho = proj_rho_R
    MPO_TM_AL._proj_rho = proj_rho_L

    P_tol = Krylov_Para["P_tol"]
    N_min = Krylov_Para["N_min"]
    N_max = Krylov_Para["N_max"]
    Arnoldi_Options = {'P_tol': P_tol,
                       'N_min': N_min,
                       'N_max': N_max,
                       'which': 'LM',
                       'num_ev': 1}
    Arnoldi_AR = Arnoldi(H=MPO_TM_AR, psi0=old_env_data["init_RP"], options=Arnoldi_Options)
    Arnoldi_AL = Arnoldi(H=MPO_TM_AL, psi0=old_env_data["init_LP"], options=Arnoldi_Options)

    _, RP_Ws, _ = Arnoldi_AR.run()
    _, LP_Ws, _ = Arnoldi_AL.run()

    RP_W_new = RP_Ws[0]
    RP_W_norm = npc.tensordot(MPO_TM_AR._proj_norm, RP_W_new, axes=[['vL*', 'wL*', 'vL'], ['vL', 'wL', 'vL*']])/MPO_TM_AR._chi0
    RP_W_new = RP_W_new / RP_W_norm

    LP_W_new = LP_Ws[0]
    LP_W_norm = npc.tensordot(MPO_TM_AL._proj_norm, LP_W_new, axes=[['vR*', 'wR*', 'vR'], ['vR', 'wR', 'vR*']])/MPO_TM_AL._chi0
    LP_W_new = LP_W_new / LP_W_norm

    new_env_data = {
        'init_LP': LP_W_new,
        'init_RP': RP_W_new
    }
    env_new = MPOEnvironment(psi, H, psi, **new_env_data)
    return env_new

def Calc_entropy():
    pass

def Calc_En_Variance(H, psi, Env):
    H_A2C = TwoSiteH(env=Env, i0=0)
    A_L = psi.get_B(0, form='A').replace_label(old_label='p', new_label='p0')
    A_R = psi.get_B(1, form='B').replace_label(old_label='p', new_label='p1')
    C = psi.get_C(1)
    A_2C = npc.tensordot(A_L, C, axes=['vR', 'vL'])
    A_2C = npc.tensordot(A_2C, A_R, axes=['vR', 'vL'])
    A_2CP = H_A2C.matvec(A_2C)
    NL = l_Ortho(psi, 0).split_legs()
    NR = r_Ortho(psi, 1).split_legs()
    A_2CP = npc.tensordot(A_2CP, NL.conj(), axes=[['vL', 'p0'], ['vL*', 'p*']])
    A_2CP = npc.tensordot(A_2CP, NR.conj(), axes=[['vR', 'p1'], ['vR*', 'p*']])
    A_2CP_c = A_2CP.conj()
    return npc.tensordot(A_2CP, A_2CP_c, axes=[['vr*', 'vl*'], ['vr', 'vl']])

def run_parallel_uMPS(H, psi, N_2Site=10, method='polar', tol=1e-12, max_iter_num=300, Krylov_Para={'E_tol':1.e-15, 
                                                                                                    'P_tol': 1.e-15, 
                                                                                                    'reortho':True, 
                                                                                                    'N_min':200, 
                                                                                                    'N_max':300, 
                                                                                                    'cutoff':1.e-15}):
    n_iter = 0
    Energy = []
    err_B = []
    delta_E = []
    mag = []
    L = psi.L
    eps_prec = 1.
    tol_s = tol/100
    Left_U = None
    Right_U = None

    Env = MPOEnvironment(psi, H, psi)
    Heff_Ac = [OneSiteH(Env, i0=i) for i in range(0, L)]
    Heff_C = [ZeroSiteH(Env, i0=i) for i in range(0, L)]

    while eps_prec >= tol and n_iter < max_iter_num:
        print("The {} VUMPS step.".format(n_iter))
        # Update environment
        eps_prec_list = []
        
        # Convergence is driven by this code section
        """ Solve for minimal energy """
        for i in range(0, L):
            """ Gs_Ac = Arnoldi(Heff_Ac[i], psi._Bc[i].ireplace_label('p', 'p0'), options={'which':'SR', 'num_ev':1})
            Gs_CL = Arnoldi(Heff_C[i], psi._C[i], options={'which':'SR', 'num_ev':1})  """
            Gs_Ac = LanczosGroundState(Heff_Ac[i], psi._Bc[i].ireplace_label('p', 'p0'), options=Krylov_Para)
            Gs_CL = LanczosGroundState(Heff_C[i], psi._C[i], options=Krylov_Para)
            E0_Ac, Ac_Til, N_Ac = Gs_Ac.run()
            E0_CL, CL_Til, N_CL = Gs_CL.run()
            psi.set_Bc(i, Ac_Til.ireplace_label('p0', 'p'))
            psi.set_C(i, CL_Til)
            # print("Updated Bc is: {}".format(psi._Bc[i]))
                    
        """ Calculate AL and AR from new Ac and C """
        for i in range(0, L):
            """ ====[A_L(i-1)]----[C(i)]----[A_R(i)]==== """
            """ Calculate A_L and A_R from updated A_c and C, then calculate error. """
            if method == "svd":
                U_Al, S_Al, VH_Al = npc.svd(npc.tensordot(psi._Bc[psi._to_valid_index(i-1)], psi._C[i].conj(), axes=('vR', 'vR*')).combine_legs(['vL', 'p']), inner_labels=['vR', 'vR*'])
                A_L = npc.tensordot(U_Al, VH_Al, axes=('vR', 'vR*')).split_legs().ireplace_label('vL*', 'vR')
                U_Ar, S_Ar, VH_Ar = npc.svd(npc.tensordot(psi._C[i].conj(), psi._Bc[i], axes=('vL*', 'vL')).combine_legs(['p', 'vR']), inner_labels=['vL*', 'vL'])
                A_R = npc.tensordot(U_Ar, VH_Ar, axes=('vL*', 'vL')).split_legs().ireplace_label('vR*', 'vL')

            elif method == "polar":
                W_C, S_C, VH_C = npc.svd(psi._C[i], 
                                         cutoff=None,
                                         inner_labels=['vR', 'vR*'])
                U_C = npc.tensordot(W_C, VH_C, axes=[['vR'], ['vR*']]) # ['vL', 'vR']
                # print("U_C is:{}".format(U_C.to_ndarray()))
                W_Al, S_Al, VH_Al = npc.svd(psi._Bc[psi._to_valid_index(i-1)].combine_legs(['vL', 'p'], qconj=[+1]), 
                                            cutoff=None,
                                            inner_labels=['vR', 'vR*'])
                # print("VH_Al is:{}".format(VH_Al.to_ndarray()))
                # print("Singular of C : {}".format(S_C))
                # print("Singular of A_c : {}".format(S_Al))

                U_Al = npc.tensordot(W_Al, VH_Al, axes=[['vR'], ['vR*']]) # ['(vL.p)', 'vR']
                A_L = npc.tensordot(U_Al, U_C.conj(), axes=[['vR'], ['vR*']]).ireplace_label('vL*', 'vR')
                A_L = A_L.split_legs()

                W_Ar, S_Ar, VH_Ar = npc.svd(psi._Bc[i].combine_legs(['p', 'vR'], qconj=[+1]),
                                            cutoff=None,
                                            inner_labels=['vL*', 'vL'])
                U_Ar = npc.tensordot(W_Ar, VH_Ar, axes=[['vL*'], ['vL']]) # ['vL', '(vR.p)']
                A_R = npc.tensordot(U_C.conj(), U_Ar, axes=[['vL*'], ['vL']]).ireplace_label('vR*', 'vL')
                A_R = A_R.split_legs()

            elif method == "QR":
                Q_Ac_l, R_Ac_l = npc.qr(psi.get_Bc(i-1).combine_legs(['vL', 'p']), mode='reduced', inner_labels=['vR', 'vL'], pos_diag_R=True, qtotal_Q=psi.get_Bc(i).qtotal)
                Q_C_l, R_C_l = npc.qr(psi.get_C(i), mode='reduced', inner_labels=['vR', 'vL'], pos_diag_R=True, qtotal_Q=psi.get_C(i+1).qtotal)
                A_L = npc.tensordot(Q_Ac_l, Q_C_l.conj(), axes=['vR', 'vR*']).split_legs()
                A_L = A_L.replace_label('vL*', 'vR')
                err_l = npc.norm(R_Ac_l - R_C_l)
                print("Err_l in QR:{}".format(err_l))

                Q_Ac_r, R_Ac_r = npc.qr(psi.get_Bc(i).combine_legs(['p', 'vR']).transpose(['(p.vR)', 'vL']), mode='reduced', inner_labels=['vL', 'vR'], pos_diag_R=True, qtotal_Q=psi.get_Bc(i).qtotal)
                Q_C_r, R_C_r = npc.qr(psi.get_C(i).transpose(['vR', 'vL']), mode='reduced', inner_labels=['vL', 'vR'], pos_diag_R=True, qtotal_Q=psi.get_C(i+1).qtotal)
                A_R = npc.tensordot(Q_Ac_r, Q_C_r.conj(), axes=['vL', 'vL*']).transpose(['vR*', '(p.vR)']).split_legs()
                A_R = A_R.replace_label('vR*', 'vL')
                err_r = npc.norm(R_Ac_r - R_C_r)
                print("Err_r in QR:{}".format(err_r))

            psi.set_B(i-1, A_L, form='A')
            psi.set_B(i, A_R, form='B')

        # ========================================================
        # Env.clear()
        # Update the environment for the calculation of error functions
        # and for the next loop of variational update
        old_env_data = Env.get_initialization_data()
        logger.debug("old_env_data is:{}".format(old_env_data))
        init_LP = old_env_data['init_LP']
        init_RP = old_env_data['init_RP']
        if Left_U is not None:
            init_LP = npc.tensordot(init_LP, Left_U, axes=[['vR'], ['vL']])
            init_LP = npc.tensordot(init_LP, Left_U.conj(), axes=[['vR*'], ['vL*']])
            old_env_data['init_LP'] = init_LP
        if Right_U is not None:
            init_RP = npc.tensordot(init_RP, Right_U, axes=[['vL'], ['vR']])
            init_RP = npc.tensordot(init_RP, Right_U.conj(), axes=[['vL*'], ['vR*']])
            old_env_data['init_RP'] = init_RP
        Env.clear()
        Env = update_env_Arnoldi(psi, H, old_env_data, Krylov_Para)

        """ for i in range(0, L):
            Ac = npc.tensordot(psi.get_B(i, form='A'), psi.get_C(i+1), axes=['vR', 'vL'])
            psi.set_Bc(i, Ac) """

        Heff_Ac = [OneSiteH(Env, i0=i) for i in range(0, L)]
        Heff_C = [ZeroSiteH(Env, i0=i) for i in range(0, L)]

        # After updating A_L and A_R
        # Calc errors
        print("================================================")
        
        for i in range(0, L):
            err_L = Cal_Galerkin_L_alt(psi, Heff_Ac, i)
            print("Err_L is:{}".format(err_L))
            err_R = Cal_Galerkin_R_alt(psi, Heff_Ac, i)
            print("Err_R is:{}".format(err_R))
            
            eps_prec_list.append(max(err_L, err_R))
        
        eps_prec = np.amax(np.array(eps_prec_list))
        En_var = Calc_En_Variance(H, psi, Env)
        print("Error during iteration:{}\n".format(eps_prec))
        print("Energy Variance:{}".format(En_var))
        print("================================================")
        n_iter += 1
        err_B.append(eps_prec)
        delta_E.append(En_var)
        # psi._Bc = [npc.tensordot(psi._C[i], psi.get_B(i, form='B'), axes=('vR', 'vL')) for i in range(0, L)]
        Energy.append(H.expectation_value(psi))
        # mag.append(psi.expectation_value('N'))
        Ac = psi._Bc
        C = psi._C  # No need to return
        # print("\n")
        
    return psi, C, Ac, Env, n_iter, Energy, err_B, delta_E, mag


# psi, C, Ac, n_iteration, Energy, err_B, mag = run_parallel_uMPS(H_Mod, psi)

""" if __name__ == "__main__":
    for j in range(Exp_Num):
        psi, C, Ac, n_iteration, Energy, err_B, mag = run_parallel_uMPS(H_Mod, psi)
        print("Energy is:{}".format(H_Mod.expectation_value(psi)))
        print("The iterations finally converge! Now the shape of local tensor is:{}\n".format(Ac[0].shape))
        density = np.real(psi.expectation_value("N"))
        print('Particle density = ', density)
        print('Tot_particale number:{}'.format(sum(density)))
        N_iter += n_iteration
        E_plt.extend(Energy)
        Err_plt.extend(err_B)
        if (j+1) < Exp_Num:
            psi = Bond_Exp(psi, H_Mod)
        # for i in range(psi.L):
            # print("At Site:", i, "\n Left canonical AL:", psi._BL[i], "\n Right canonical AR:", psi._B[i], "\n Mixed canonical form Ac:", psi._Bc[i], "\n C[i]:", psi._C[i])

    Fig = plt.figure()
    ax1 = Fig.add_subplot(111)
    steps = np.linspace(start=1, stop=N_iter, num=N_iter)
    ax1.set_title("Energy")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("E")
    ax1.grid()
    ax1.plot(steps, E_plt, '-vr')
    plt.show() """

