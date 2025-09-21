import sys
import math
import pickle
import gzip
import numpy as np
import tenpy.linalg.np_conserved as npc

from tenpy.networks.mpo import MPOEnvironment
from tenpy.linalg.krylov_based import Arnoldi
from loguru import logger
from VUMPS_and_Excitation.Algorithm.Single_Mode_Approximation.H_eff_ex import Excited_MultisiteH
from VUMPS_and_Excitation.System.Bond_Control_RE_RE import l_Ortho
from VUMPS_and_Excitation.Algorithm.VUMPS.VUMPS_diag_C import update_env_Arnoldi

""" Please enter the path you save the data and the file name of VUMPS and iDMRG data """
""" ======================================= """
DataPath = 'Your_Data_Path'
VUMPSFileName = 'The_Name_VUMPS'
IDMRGFileName = 'The_Name_iDMRG'
""" ======================================= """

Bond_Dim = 200
Excitation_Sector = 1
Tot_k_num = 100
num_energy_levels = 30
Ly = 10
Eigen_Shift = 0.0 # The eigenshift we add to avoid small eigenvalues approching 0
# The input momentum should be multiplied by 2$\pi$
momentum_idx = 1.0 * int(sys.argv[1])
momentum = momentum_idx * math.pi * (2.0/Tot_k_num)
logger.info("Momentum index is:{}".format(momentum_idx))
logger.info("Momentum is:{}".format(momentum))

P_tol_env = 1.e-9
P_tol_exci = 1.e-9

with gzip.open(DataPath+VUMPSFileName, "rb") as input_f:
    VUMPS_data = pickle.load(input_f)
input_f.close()

psi_GS = VUMPS_data['psi']
L = psi_GS.L

with gzip.open(DataPath+IDMRGFileName, "rb") as input_f_dmrg:
    dmrg_data = pickle.load(input_f_dmrg)
input_f_dmrg.close()

B_Hof = dmrg_data["iDMRG_Engine"].model
B_Hof_MPO = B_Hof.calc_H_MPO()

# Initialize of a X at U1 sector 1
X0_list = []
data_type = B_Hof.dtype
for i in range(0, L):
    NL_i = l_Ortho(psi=psi_GS, site=i).split_legs()
    v_leg_L = psi_GS.get_Bc(i).get_leg("vL")
    v_leg_R = psi_GS.get_Bc(i).get_leg("vR")
    p_leg = psi_GS.get_Bc(i).get_leg("p")
    # Ensure that B0_init has true q_total
    qtot_i = psi_GS.get_Bc(i).qtotal[0]
    B0_init = npc.Array.from_func(func=np.random.standard_normal, legcharges=[v_leg_L, p_leg, v_leg_R]
                                , dtype=data_type, qtotal=[qtot_i + Excitation_Sector], labels=['vL', 'p', 'vR'])
    logger.info("The qtotal of the {} excitation tensor B:{}", i, B0_init.qtotal[0])
    X0_i = npc.tensordot(B0_init, NL_i.conj(), axes=[['vL', 'p'], ['vL*', 'p*']])
    X0_i = X0_i.replace_label(old_label="vr*", new_label="vl")
    logger.info("The intial excitation tensor at site {}: {}", i, X0_i)
    logger.info("The qtotal of the {} excitation tensor X:{}", i, X0_i.qtotal[0])
    X0_list.append(X0_i)
X0 = X0_list[0].combine_legs(["vl", "vR"])
for i in range(1, L):
    X0 = npc.concatenate([X0, X0_list[i].combine_legs(["vl", "vR"])], axis=0)

env_old = MPOEnvironment(psi_GS, B_Hof_MPO, psi_GS)
old_env_data = env_old.get_initialization_data()
Exi_Lanczos_Para = {'E_tol':1.e-9, 
                    'P_tol':P_tol_env, 
                    'reortho':True, 
                    'hermitian':True,
                    'N_min':2, 
                    'N_max':10000,
                    'num_ev':num_energy_levels,
                    'which':'SR',
                    'cutoff':1.e-12}
# Here We just take `N_min`, `N_max`, `P_tol`, `E_tol` from this Para_dict. So don't worry.
Env = update_env_Arnoldi(psi_GS, B_Hof_MPO, old_env_data, Exi_Lanczos_Para)

H_ex = Excited_MultisiteH(env=Env, psi_GS=psi_GS, H=B_Hof_MPO, k=momentum, X_list=X0_list, eigen_shift=Eigen_Shift, group_sites=True, ex_charge=[Excitation_Sector])

Exi_Lanczos_Para['P_tol'] = P_tol_exci

Exi_Solver = Arnoldi(H=H_ex, psi0=X0, options=Exi_Lanczos_Para)
E0s, psis, N_Kry = Exi_Solver.run()
E0s -= Eigen_Shift # We add eigen_shift to the eigenvalues in Excited_MultisiteH, so we need to substract it here.
Excitation_data = {'Momentum': momentum,
                   'E0s': E0s,
                   'psis': psis,
                   'E_k': E0s,
                   'H_ex_engine': H_ex,
                   'N_Krylov': N_Kry}
logger.info("Lanczos Result:{}", E0s)
with gzip.open(DataPath+"Hofstadter_Ly_"+str(Ly)+"_Bond_Dim_"+str(Bond_Dim)+"_k_"+str(momentum_idx)+".pklz", "wb") as output_H_ex:
    pickle.dump(Excitation_data, output_H_ex)
output_H_ex.close()