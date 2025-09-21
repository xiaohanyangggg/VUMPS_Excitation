import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

import tenpy.linalg.np_conserved as npc
from loguru import logger
from VUMPS_and_Excitation.Observables.Spec_Func_Edge import Spectral_function


def Packing_Exci_data(exci_file_path, save_file_path, file_name, k_range, Bond_Dim, Ly, num_E_mesh):
    # Import the data of excitaztion states and their corresponding energy
    # We also store the X_list and Es_list for further convenience
    # This code block should be commented out at the next run  
    X_list = []
    Es_list = []
    for k_n in k_range:
        with gzip.open(exci_file_path+"/"+ file_name[0]+ "{:.1f}".format(k_n) + file_name[1], "rb") as k_point_file:
            k_point_data = pickle.load(k_point_file)
            
            X_list_k = k_point_data["psis"]
            print("Momentum at {} is {}".format(k_n, k_point_data['Momentum']))
            X_list_k_of_Xlist = []
            if len(X_list_k) < num_E_mesh:
                raise ValueError("Number of energy levels is:{0}, but E mesh number is {1}. Number of E mesh should be equal to or smaller than the number of energy levels at a k m point.\n".format(len(X_list_k), num_E_mesh))
            for j in range(num_E_mesh):
                X_list_k_of_Xlist.append([X_list_k[j]])
            Es_list_k = k_point_data["E0s"][0 : num_E_mesh]
            del X_list_k

            X_list.append(X_list_k_of_Xlist)
            Es_list.append(Es_list_k)
    X_E_data = {
                    "X_list": X_list,
                    "Es_list": Es_list
                }
    with gzip.open(save_file_path + "/Charge_1_Organized_ExciData_BD" + str(Bond_Dim) + "_Ly_"+str(Ly)+"_"+str(num_E_mesh)+"levels.pklz", "wb") as save_X_E_file:
        pickle.dump(X_E_data, save_X_E_file)
    save_X_E_file.close()
    return X_E_data

if __name__ == "__main__":
    """ 1) Import Excitation data and organize them into the form for Spectral_function """
    
    DataPath = "Your_Data_Path"
    VUMPSFileName = 'The_Name_VUMPS'
    # The file name is a list with two elements, the first one is the prefix, the second one is the suffix
    file_name = ["Hofstadter_Ly_10_Bond_Dim_200_k_", ".pklz"]
    
    k_range = np.linspace(start=-25.0, stop=25.0, num=51, endpoint=True)
    Bond_Dim = 2000
    Ly = 10
    num_E_mesh = 20
    k_space = np.linspace(-25.0*np.pi/50, 25.0*np.pi/50, num=51, endpoint=True)
    X_E_data = Packing_Exci_data(exci_file_path=DataPath,
                                 save_file_path=DataPath,
                                 file_name=file_name,
                                 k_range=k_range,
                                 Bond_Dim=Bond_Dim,
                                 Ly=Ly,
                                 num_E_mesh=num_E_mesh)
    
    X_list = X_E_data["X_list"]
    Es_list = X_E_data["Es_list"]
    
    charge_sector = 0
    
    """ 2) Import necessary data of the ground state"""
    with gzip.open(DataPath+VUMPSFileName, "rb") as input_GS_f:
        VUMPS_data = pickle.load(input_GS_f)
    input_GS_f.close()
    psi_GS = VUMPS_data["psi"]
    del VUMPS_data
    
    # Cut the concatenated X tensors
    with gzip.open(DataPath+"/"+ file_name[0]+ "{:.1f}".format(k_range[0]) + file_name[1], "rb") as H_exci_file_0:
        H_exci_data_0 = pickle.load(H_exci_file_0)
    H_exci_file_0.close()
    H_eff = H_exci_data_0['H_ex_engine']
    X_slice = H_eff.X_slice
    X_legs = H_eff.X_legs
    L_UC = H_eff.L_UC

    X_list_cut = []
    
    for k_n in range(0, len(k_range)):
        X_list_k = []
        for w_n in range(num_E_mesh):
            X_list_k_w = []
            for site in range(L_UC):
                X_k_w_site = X_list[int(k_n)][w_n][0][X_slice[site] : X_slice[site+1]]
                print("X_k_site has type:{}".format(X_k_w_site))
                X_kwsite_from_ndarray = npc.Array.from_ndarray(data_flat=X_k_w_site.to_ndarray(), legcharges=[X_legs[site]],
                                                    dtype=psi_GS.dtype, qtotal=X_k_w_site.qtotal, labels=["(vl.vR)"])
                X_list_k_w.append(X_kwsite_from_ndarray.split_legs())
            X_list_k.append(X_list_k_w)
        X_list_cut.append(X_list_k)
    del H_exci_data_0
    
    """ 3) Calculate Spectral function """
    Kry_Para = {
                    "N_min":2,
                    "N_max":1000,
                    "res":1.e-9
                }
    overlap = []
    E_k = []
    for y in range(Ly):
        if charge_sector == 1:
            Ops = ['Bd']
        elif charge_sector == 0:
            Ops = ['N']
        Ops_indx = [y]

        Spectral_Solver = Spectral_function(psi=psi_GS, 
                                            X_lists=X_list_cut, 
                                            Eks=Es_list, 
                                            ops_str=Ops, 
                                            ops_indices=Ops_indx, 
                                            eta_w=0.03,
                                            eta_k=0.03,
                                            k_space=k_space,
                                            Kry_Para=Kry_Para)

        Spectrum_data = {
                            "Spectral_Solver": Spectral_Solver
                        }

        with gzip.open(DataPath + "/B_Hof_charge_"+str(charge_sector)+"_Ly_"+str(Ly)+"_spectral_data_solver_Dim"+str(Bond_Dim)+"_site"+str(y)+".pklz", "wb") as save_spectrum_file:
            pickle.dump(Spectrum_data, save_spectrum_file)
        save_spectrum_file.close()
        overlap.append(Spectrum_data['Spectral_Solver'].overlap_matrix)
        E_k.append(Spectrum_data['Spectral_Solver'].Eks)

