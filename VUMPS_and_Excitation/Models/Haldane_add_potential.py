import numpy as np

from tenpy.models import CouplingMPOModel
from tenpy.models.lattice import Honeycomb
from tenpy.networks.site import FermionSite


class HaldaneHoney_Harmonic(CouplingMPOModel):
    default_lattice = Honeycomb
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        site = FermionSite(conserve=conserve)
        return site

    def init_terms(self, model_params):
        t1 = np.asarray(model_params.get('t1', -1.))
        t2_default = np.sqrt(129) / 36 * t1 * np.exp(1j * np.arccos(3 * np.sqrt(3 / 43)))
        t2 = np.asarray(model_params.get('t2', t2_default))
        V = np.asarray(model_params.get('V', 0))
        mu = np.asarray(model_params.get('mu', 0.))
        phi_ext = model_params.get('phi_ext', 0.)
        potential_type = model_params.get('potential_type', 'Harmonic')
        
        self.add_onsite(mu, 0, 'N', category='mu N')
        self.add_onsite(-mu, 1, 'N', category='mu N')

        # V_harm = self.Harmonic_Potential(model_params)
        if potential_type == 'Harmonic':
            V_harm0, V_harm1 = self.Harmonic_Potential_2(model_params)
            self.add_onsite(V_harm0, 0, 'N', category='V_harm')
            self.add_onsite(V_harm1, 1, 'N', category='V_harm')
        elif potential_type == 'Linear':
            V_Linear0, V_Linear1 = self.Linear_Potential(model_params)
            self.add_onsite(V_Linear0, 0, 'N', category='V_linear')
            self.add_onsite(V_Linear1, 1, 'N', category='V_linear')
            print("Linear potential: {0}, {1}".format(V_Linear0, V_Linear1))

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            t1_phi = self.coupling_strength_add_ext_flux(t1, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(t1_phi, u1, 'Cd', u2, 'C', dx, category='t1 Cd_i C_j', plus_hc=True)
            self.add_coupling(V, u1, 'N', u2, 'N', dx, category='V N_i N_j')

        for u1, u2, dx in [(0, 0, np.array([-1, 1])), (0, 0, np.array([1, 0])),
                           (0, 0, np.array([0, -1])), (1, 1, np.array([0, 1])),
                           (1, 1, np.array([1, -1])), (1, 1, np.array([-1, 0]))]:
            t2_phi = self.coupling_strength_add_ext_flux(t2, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(t2_phi, u1, 'Cd', u2, 'C', dx, category='t2 Cd_i C_j', plus_hc=True)

    def Harmonic_Potential(self, model_params):
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 8)
        V_trap = model_params.get('V_trap', 0.01)
        y_0 = (Ly - 1)/2.0
        V_xy = []
        V_y = []
        for x in range(0, Lx):
            for y in range(0, Ly):
                V_y.append(V_trap*((y-y_0)**2))
            V_xy.append(V_y)
        return np.array(V_xy)
    
    def Harmonic_Potential_2(self, model_params):
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 8)
        V_trap = model_params.get('V_trap', 0.01)
        y_0 = (Ly - 1)/2.0
        V_xy_0 = []
        V_y_0 = []
        V_xy_1 = []
        V_y_1 = []
        y_shift = 1.0 / 6.0
        for x in range(0, Lx):
            for y in range(0, Ly):
                V_y_0.append(V_trap*((y-y_shift-y_0)**2))
                V_y_1.append(V_trap*((y+y_shift-y_0)**2))
            V_xy_0.append(V_y_0)
            V_xy_1.append(V_y_1)
        return V_xy_0, V_xy_1
    
    def Linear_Potential(self, model_params):
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 8)
        V_trap = model_params.get('V_trap', 0.01)
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 8)
        V_trap = model_params.get('V_trap', 0.01)
        y_0 = (Ly - 1)/2.0
        V_xy_0 = []
        V_y_0 = []
        V_xy_1 = []
        V_y_1 = []
        y_shift = 1.0 / 6.0
        for x in range(0, Lx):
            for y in range(0, Ly):
                V_y_0.append(V_trap*abs(y-y_shift-y_0))
                V_y_1.append(V_trap*abs(y+y_shift-y_0))
            V_xy_0.append(V_y_0)
            V_xy_1.append(V_y_1)
        return V_xy_0, V_xy_1