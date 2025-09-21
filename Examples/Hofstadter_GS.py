# %%
import numpy as np
import math
import pickle
import gzip
import tenpy.linalg.np_conserved as npc

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from VUMPS_and_Excitation.Algorithm.VUMPS.VUMPS_diag_C import run_parallel_uMPS
from VUMPS_and_Excitation.System.mps_Replica import MPS_Replica
from VUMPS_and_Excitation.Models.Hof_Hamiltonian_MPO import HofstadterBoson_Monomial_Potential

from tenpy.tools.params import asConfig
from loguru import logger

import logging

logging.basicConfig(level=logging.INFO)

""" Please enter the path you save the data """
""" ======================================= """
DataPath = 'Your_Data_Path'
""" ======================================= """

single_particle = False
Bond_Dim = 200
# Exp_Num = 5
# `Filling` has no effect on the model`s MPO but it shift the definition of `dN`
# `Landau_y` means that if `phi` is 1/q, we choose a magnetic unit cell (1,q)
filling_num = '1_over_2'
Lx = 1
Ly = 10
V = 0.0
Jx = 1.
Jy = 1.

Mod_para = {    "bc_MPS": "infinite",
                "bc_x": "periodic",
                "bc_y": "periodic",
                "conserve": "N",
                "filling": (1, Ly),
                "phi": (1, 4),
                "gauge": "landau_y",
                "Jy": Jx,
                "Jx": Jy,
                "U": 0.,
                "V": V,
                "mu": 0.,
                "Lx": Lx,
                "Ly": Ly,
                "Nmax": 1,
                "order": "default",
                "phi_ext": 0.,
                "potential_type": "Harmonic"
            }

# Initialization of Model MPO
B_Hof = HofstadterBoson_Monomial_Potential(Mod_para)
B_Hof.dtype = np.complex128
L = B_Hof.lat.N_sites
B_Hof_MPO = B_Hof.calc_H_MPO()

data_type = B_Hof.dtype
for i in range(0, B_Hof_MPO.L):
    B_Hof_MPO._W[i] = (1. + 0.j) * B_Hof_MPO._W[i]
B_Hof_MPO.dtype = data_type

# Initialization of MPS from iDMRG 
p_leg = B_Hof_MPO.get_W(0).get_leg("p")
chinfo = p_leg.chinfo

p_site = B_Hof.lat.site(0)
L = B_Hof.lat.N_sites

product_state = []
if single_particle:
    for i in range(L):
        product_state.append(0)
elif not single_particle:
    for i in range(L):
        if i%Mod_para['filling'][1] == 0:
            product_state.append(1)
        else:
            product_state.append(0)

psi = MPS.from_product_state(B_Hof.lat.mps_sites(), product_state, bc=B_Hof.lat.bc_MPS)
for i in range(0, L):
    print(psi.get_B(i).qtotal)
initial_density = np.real(psi.expectation_value("N"))
print('Initial particle density = ', initial_density)

dmrg_params = {'mixer': True,
               'mixer_params': {
                    'amplitude': 5.e-5,
                    'decay': 1.1,
                    'disable_after': 100
                }, 
                'trunc_params': {
                    'svd_min': 1.e-10,
                    'chi_max': Bond_Dim
                },
                'max_E_err': 1.e-11,
                'max_S_err': 1.e-10,
                'min_sweeps': 10,
                'max_sweeps': 2000,
                'lanczos_params':{
                    'reortho': False, 
                    'N_min':5, 
                    'N_max':1000
                }
            }

options = asConfig(dmrg_params, 'DMRG')
active_sites = options.get('active_sites', 2)

logger.info("==================== Start the DMRG engine ====================")

if active_sites == 1:
    engine = dmrg.SingleSiteDMRGEngine(psi, B_Hof, options)
elif active_sites == 2:
    engine = dmrg.TwoSiteDMRGEngine(psi, B_Hof, options)
else:
    raise ValueError("For DMRG, can only use 1 or 2 active sites, not {}".format(active_sites))
E, _ = engine.run()

data =  {   
            'dmrg_params' : dmrg_params,
            'psi' : psi,
            'E': E,
            'iDMRG_Engine': engine
        }

filename = DataPath+'Hofstadter_Ly_'+str(Ly)+'_Vtrap_'+str(V)+'_Bond_Dim_'+str(Bond_Dim)+'iDMRG2.pklz'
with gzip.open(filename, 'wb') as f_idmrg:
    pickle.dump(data, f_idmrg)
f_idmrg.close()

print("IDMRG has ended.")

psi_init = MPS_Replica.from_MPS(psi)

Krylov_para = {'reortho':True, 'N_min':5, 'N_max':1000, 'cutoff':1.e-12}

logger.info("==================== Start the VUMPS engine ====================")

u_psi_GS, C, Ac, Env, n_iteration, Energy, err_B, delta_E, mag = run_parallel_uMPS(B_Hof_MPO, psi_init, method='polar', tol=1e-8, max_iter_num=2500, checkpoint_num=1000, Krylov_Para=Krylov_para)
vumps_data = {"psi": u_psi_GS,
              "E": Energy,
              "delta_E": delta_E[-1],
              "tangent_Norm": err_B,
              "Env": Env}

with gzip.open(DataPath+"Hofstadter_Ly_"+str(Ly)+"_Bond_Dim_"+str(Bond_Dim)+"VUMPS_Vtrap_"+str(V)+".pklz", "wb") as output_VUMPS:
    pickle.dump(vumps_data, output_VUMPS)
output_VUMPS.close()

logger.info("Now the iDMRG and VUMPS procedures have finished.")
    
