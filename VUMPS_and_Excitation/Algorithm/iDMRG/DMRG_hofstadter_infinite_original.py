import numpy as np
import pickle
import argparse
import gzip
from tenpy.networks.mps import MPS
from tenpy.models.hofstadter import HofstadterBosons
from tenpy.algorithms import dmrg

parser = argparse.ArgumentParser()
parser.add_argument('-Lx', help='use like this: -Lx=1', type=int, default=1)
parser.add_argument('-Ly', help='use like this: -Ly=8', type=int, default=8)
parser.add_argument('-Jx', help='use like this: -Jx=1.', type=float, default=1.)
parser.add_argument('-Jy', help='use like this: -Jy=1.', type=float, default=1.)
parser.add_argument('-chi', help='use like this: -chi=100', type=int, default=1000)
args = parser.parse_args()
Lx = args.Lx
Ly = args.Ly
Jx = args.Jx
Jy = args.Jy
chi = args.chi

def DMRG_HofstadterBosons(Lx, Ly, Jx, Jy, chi):
    model_params = dict(Lx = 1,
                        Ly = 8,
                        Nmax = 1,
                        filling = (1,8),
                        Jx = 1.0,
                        Jy = 1.0,
                        U = 0.,
                        conserve = 'N',
                        phi = (1,4),
                        phi_ext = 0.,
                        gauge = 'landau_y',
                        bc_MPS ='infinite',
                        bc_y = 'open',
                        order ='default',)
                        
    M = HofstadterBosons(model_params)
    
    L = M.lat.N_sites
    
    product_state = []
    for i in range(L):
        if i%model_params['filling'][1] == 0:
            product_state.append(1)
        else:
            product_state.append(0)

    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    for i in range(0, L):
        print(psi.get_B(i).qtotal)
    initial_density = np.real(psi.expectation_value("N"))
    print('Initial particle density = ', initial_density)
    
    dmrg_params = {
        'mixer': True, 
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    ground_state_density = psi.expectation_value("N")
    print('ground_state_density = ', ground_state_density)
    print('S = ', psi.entanglement_entropy())
    print('Calculate compute_K:')
    u, w, q, o, trunc_err = psi.compute_K(M.lat)
    print('trunc_err = ', trunc_err)
    print('save data:')
        
    data = {}
    data['M'] = M
    data['psi'] = psi
    data['chi'] = psi.chi
    data['E'] = E
    data['ground_state_density'] = ground_state_density
    data['S'] = psi.entanglement_entropy()
    data['u'] = u
    data['w'] = w
    data['q'] = q
    data['o'] = o
    data['trunc_err'] = trunc_err
    data['spectrum'] = psi.entanglement_spectrum()
    return psi, data


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    data = DMRG_HofstadterBosons(Lx, Ly, Jx, Jy, chi)
    filename = 'hofstadter_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_chi_'+str(chi)+'_infinite.pklz'
    f1 = gzip.open(filename, 'wb')
    pickle.dump(data,f1)
    f1.close()
