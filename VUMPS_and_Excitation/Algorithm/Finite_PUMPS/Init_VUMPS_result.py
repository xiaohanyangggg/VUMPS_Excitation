import numpy as np
import tenpy.linalg.np_conserved as npc

from tenpy.tools.params import asConfig
from tenpy.algorithms import dmrg

from VUMPS_and_Excitation.Algorithm.VUMPS.VUMPS_diag_C import run_parallel_uMPS
from VUMPS_and_Excitation.System.mps_Replica import MPS_Replica

import logging

# logging.basicConfig(level=logging.INFO)

def Init_VUMPS(psi_init, H_Mod, dmrg_params, VUMPS_params, use_DMRG=True):
    """
    H_mod : :class:tenpy.models.MPOModel
    """
    options = asConfig(dmrg_params, 'DMRG')
    active_sites = options.get('active_sites', 2)

    if use_DMRG:
        if active_sites == 1:
            engine = dmrg.SingleSiteDMRGEngine(psi_init, H_Mod, options)
        elif active_sites == 2:
            engine = dmrg.TwoSiteDMRGEngine(psi_init, H_Mod, options)
        else:
            raise ValueError("For DMRG, can only use 1 or 2 active sites, not {}".format(active_sites))
        E, psi_idmrg = engine.run()
    elif not use_DMRG:
        psi_idmrg = psi_init
    H_Mod_MPO = H_Mod.calc_H_MPO()
    psi_idmrg = MPS_Replica.from_MPS(psi_idmrg)
    u_psi_GS, C, Ac, Env, n_iteration, Energy, err_B, delta_E, mag = run_parallel_uMPS(H_Mod_MPO, 
                                                                                       psi_idmrg, 
                                                                                       method='polar', 
                                                                                       tol=1.e-9, 
                                                                                       max_iter_num=2500, 
                                                                                       Krylov_Para=VUMPS_params)
    return Energy[-1], u_psi_GS