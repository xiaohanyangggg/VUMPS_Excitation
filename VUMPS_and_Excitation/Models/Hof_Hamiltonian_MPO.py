import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models.model import CouplingModel, MPOModel, NearestNeighborModel, CouplingMPOModel
from tenpy.models.hofstadter import gauge_hopping
from tenpy.models.lattice import Square
from tenpy.networks.site import BosonSite

__all__ = ["Harmonic_Potential", "HofstadterBoson_Harmonic_Potential"]

def Harmonic_Potential(model_params):
    Lx = model_params.get('Lx', 1)
    if model_params.get('bc_x', 'periodic') == 'periodic'and model_params.get('bc_MPS', 'infinite') == 'finite':
        Lx = 1
    Ly = model_params.get('Ly', 8)
    V = model_params.get('V', 0.01)
    y_0 = model_params.get('y0', (Ly - 1)/2.0)
    V_xy = []
    for x in range(0, Lx):
        V_y = []
        for y in range(0, Ly):
            V_y.append(V*((y-y_0)**2))
        V_xy.append(V_y)
    return np.array(V_xy)

class HofstadterBoson_Harmonic_Potential(CouplingMPOModel):

    default_lattice = Square
    force_default_lattice = True

    def init_sites(self, model_params):
        Nmax = model_params.get('Nmax', 3)
        conserve = model_params.get('conserve', 'N')
        filling = model_params.get('filling', (1, 8))
        filling = filling[0] / filling[1]
        site = BosonSite(Nmax=Nmax, conserve=conserve, filling=filling)
        return site
    
    def init_terms(self, model_params):
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        phi_ext = model_params.get('phi_ext', 0.)
        mu = np.asarray(model_params.get('mu', 0.))
        U = np.asarray(model_params.get('U', 0))
        if model_params.get('gauge', default='No gauge given!') != 'landau_y':
            raise ValueError("This class only works for y gauge case.")
        hop_x, hop_y = gauge_hopping(model_params)
        if Ly % hop_x.shape[1] != 0:
            delta_L = Ly % hop_x.shape[1]
            hop_x_repeated = np.tile(hop_x, (1, Ly // hop_x.shape[1]))
            hop_x_resi = hop_x[:, :delta_L]
            result_hop_x = np.concatenate((hop_x_repeated, hop_x_resi), axis=1)
            hop_x = result_hop_x
            print("Check the extended hop_x: {}".format(hop_x))
        # 6) add terms of the Hamiltonian
        self.add_onsite(U / 2, 0, 'NN')
        self.add_onsite(-U / 2 - mu, 0, 'N')
        # 7) Add the harmonic potential
        Harm_V = Harmonic_Potential(model_params)
        print("Check harmonic potential:{}, {}".format(Harm_V, type(Harm_V)))
        print("Lattice has Ls:{}".format(self.lat.Ls))
        self.add_onsite(Harm_V, 0, 'N')
        dx = np.array([1, 0])
        self.add_coupling(hop_x, 0, 'Bd', 0, 'B', dx)
        self.add_coupling(np.conj(hop_x), 0, 'Bd', 0, 'B', -dx)  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Bd', 0, 'B', -dy)  # h.c.
        
class HofstadterBoson_Monomial_Potential(CouplingMPOModel):

    default_lattice = Square
    force_default_lattice = True

    def init_sites(self, model_params):
        Nmax = model_params.get('Nmax', 3)
        conserve = model_params.get('conserve', 'N')
        filling = model_params.get('filling', (1, 8))
        filling = filling[0] / filling[1]
        site = BosonSite(Nmax=Nmax, conserve=conserve, filling=filling)
        return site
    
    def init_terms(self, model_params):
        Lx = self.lat.shape[0]
        Ly = self.lat.shape[1]
        phi_ext = model_params.get('phi_ext', 0.)
        mu = np.asarray(model_params.get('mu', 0.))
        U = np.asarray(model_params.get('U', 0))
        if model_params.get('gauge', default='No gauge given!') != 'landau_y':
            raise ValueError("This class only focuses on y gauge case.")
        hop_x, hop_y = gauge_hopping(model_params)
        if Ly % hop_x.shape[1] != 0:
            delta_L = Ly % hop_x.shape[1]
            hop_x_repeated = np.tile(hop_x, (1, Ly // hop_x.shape[1]))
            hop_x_resi = hop_x[:, :delta_L]
            result_hop_x = np.concatenate((hop_x_repeated, hop_x_resi), axis=1)
            hop_x = result_hop_x
            print("Check the extended hop_x: {}".format(hop_x))
        # 6) add terms of the Hamiltonian
        self.add_onsite(U / 2, 0, 'NN')
        self.add_onsite(-U / 2 - mu, 0, 'N')
        
        # 7) Add the monomial potential
        Harm_V = self.Monomial_Potential(model_params)
        print("The monomial potential type is:{}".format(model_params.get('potential_type', 'Harmonic')))
        print("Check hthe monomial potential:{}, {}".format(Harm_V, type(Harm_V)))
        print("Lattice has Ls:{}".format(self.lat.Ls))
        self.add_onsite(Harm_V, 0, 'N')
        
        dx = np.array([1, 0])
        self.add_coupling(hop_x, 0, 'Bd', 0, 'B', dx)
        self.add_coupling(np.conj(hop_x), 0, 'Bd', 0, 'B', -dx)  # h.c.
        dy = np.array([0, 1])
        hop_y = self.coupling_strength_add_ext_flux(hop_y, dy, [0, phi_ext])
        self.add_coupling(hop_y, 0, 'Bd', 0, 'B', dy)
        self.add_coupling(np.conj(hop_y), 0, 'Bd', 0, 'B', -dy)  # h.c.
        
    def Monomial_Potential(self, model_params):
        Lx = model_params.get('Lx', 1)
        if model_params.get('bc_x', 'periodic') == 'periodic'and model_params.get('bc_MPS', 'infinite') == 'finite':
            Lx = 1
        Ly = model_params.get('Ly', 8)
        V = model_params.get('V', 0.01)
        y_0 = model_params.get('y0', (Ly - 1)/2.0)
        potential_type = model_params.get('potential_type', 'Harmonic')
        V_xy = []
        for x in range(0, Lx):
            V_y = []
            for y in range(0, Ly):
                if potential_type == 'Linear':
                    V_y.append(V*abs(y-y_0))
                elif potential_type == 'Harmonic':
                    V_y.append(V*((y-y_0)**2))
                elif potential_type == 'Cubic':
                    V_y.append(V*(abs(y-y_0)**3))
                elif potential_type == 'Quartic':
                    V_y.append(V*((y-y_0)**4))
                elif potential_type == 'Quintic':
                    V_y.append(V*(abs(y-y_0)**5))
                elif potential_type == 'Exp':
                    V_y.append(V*(np.exp(abs(y-y_0))-1))
            V_xy.append(V_y)
        return np.array(V_xy)