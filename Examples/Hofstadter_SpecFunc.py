import gzip
import pickle
import math
import string
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

plt.rcParams['mathtext.fontset'] = 'cm'
# Import spectral intensity data
Bond_Dim = 200
k_shift = False
k_brodening = False
eta_w = 0.01
eta_k = 0
Ly = 10
sites_range = range(0, 10)

V_trap = 0.0

nrows, ncols = 2, int(np.ceil(len(sites_range)/2))
w_lowerbound = -2.65
w_upperbound = -2.1
w_shift = 0.0 # w -> w + w_shift
k_leftbound = -math.pi/2
k_rightbound = math.pi/2
num_of_w = 51
delta_w = (w_upperbound - w_lowerbound)/(num_of_w - 1)
alpha_energy = 0.2
k_center_half = False

if k_center_half:
    delta_k = 0
elif not k_center_half:
    delta_k = 2*np.pi/100

charge_sector = 1

Plot_energy_levels = True

DataPath = "Your_Data_Path"

with gzip.open(DataPath+"/B_Hof_charge_"+str(charge_sector)+"_Ly_"+str(Ly)+"_spectral_data_solver_Dim"+str(Bond_Dim)+"_site"+str(0)+".pklz", "rb") as Packaged_E_k_file:
    E_ks_data = pickle.load(Packaged_E_k_file)
Packaged_E_k_file.close()
E_ks = E_ks_data["Spectral_Solver"].Eks
num_Plot_level = 10

fig, Axes = plt.subplots(nrows, ncols, figsize=(ncols*6+12, 14), constrained_layout=False)
print("Axes:{}".format(Axes))
# Axes = [[axes[0]], [axes[1]]]
fig.subplots_adjust(wspace=0.11, hspace=0.070, left=0.1, right=0.95, bottom=0.065, top=0.98)

alphabet_list = list(string.ascii_lowercase)

cmap_type = 'magma'

line_color = 'aliceblue'

text_color = 'white'
# text_color = 'black'
cmap = plt.get_cmap(cmap_type)

print("*"*30)
print("Package data for plotting.")
print("*"*30)

A_for_Plot_i_B = []
for site_B in sites_range:
    print("Row y:{}".format(site_B))
    
    with gzip.open(DataPath+"/B_Hof_charge_"+str(charge_sector)+"_Ly_"+str(Ly)+"_spectral_data_solver_Dim"+str(Bond_Dim)+"_site"+str(site_B)+".pklz", "rb") as B_Hof_intensity_file:
        B_Hof_intensity_data = pickle.load(B_Hof_intensity_file)
    B_Hof_intensity_file.close()

    Spectral_Solver = B_Hof_intensity_data["Spectral_Solver"]
    Spec_Intensity = Spectral_Solver.overlap_matrix
    num_level = Spectral_Solver.num_exci_level
    num_exci_k = Spectral_Solver.num_exci_k
    print("Total number of k points we calculated:{}".format(num_exci_k))
    # Set eta_k = 0 since we just consider energy broadening
    Spectral_Solver.eta_k = eta_k
    Spectral_Solver.eta_w = eta_w

    # If we don't need brodening in k space, the k_plot is just the indices list. So we only need a list of integers in this case
    k_range = np.linspace(start=0, stop=num_exci_k-1, num=num_exci_k, endpoint=True)
    k_space = np.linspace(start=-np.pi, stop=np.pi, num=num_exci_k, endpoint=True)
    # Here k_range need not to be shifted since we do it in Spectral_Solver
    w_space = np.linspace(start=w_lowerbound, stop=w_upperbound, num=num_of_w, endpoint=True)
    A = []
    if k_brodening:
        k_plot = k_space
    elif not k_brodening:
        k_plot = k_range
        
    for k_n in k_plot:
        A_k = []
        for w in w_space:
            A_k_w = Spectral_Solver.Calculate_A_k_w(k_n, w)
            A_k.append(A_k_w)
        A.append(A_k)

    A_Transpose = [[row[i] for row in A] for i in range(len(A[0]))]
    A_for_Plot = A_Transpose[::-1]
    A_for_Plot_i_B.append(A_for_Plot)

A_for_Plot_i_B_array = np.array(A_for_Plot_i_B)
v_max = A_for_Plot_i_B_array.max()
v_min = A_for_Plot_i_B_array.min()
print("v_max: {}".format(v_max))
print("v_min: {}".format(v_min))

print("*"*30)
print("Plot the intensity.")
print("*"*30)

# ax_index is the order of the axes
for ax_index in range(len(sites_range)):
    if ax_index < ncols:
        A_index = ax_index
    elif ax_index >= ncols:
        A_index = ncols-ax_index-1
    print("ax_idx:{}".format(ax_index))
    site_B = sites_range[A_index]
    print("Row y:{}".format(site_B))
    row = math.floor(ax_index / ncols)
    column =  (ax_index) % ncols
    print(row, column)
    ax = Axes[row][column]
    alph_index = alphabet_list[column]
    extent = [k_leftbound-delta_k/2, k_rightbound+delta_k/2, w_lowerbound+w_shift-delta_w/2, w_upperbound+w_shift+delta_w/2]
    im = ax.imshow(A_for_Plot_i_B[A_index], cmap=cmap, aspect='auto', vmin=v_min, vmax=v_max, extent=extent)
    ax.set_xlim(extent[0]+delta_k/2, extent[1]-delta_k/2)
    ax.set_ylim(extent[2]+delta_w/2, extent[3]-delta_w/2)
    # Control the ticks:
    if column == 0:
        ax.set_ylabel(r'$\omega+\mu$', fontdict={'fontsize': 70, 'family': 'serif', 'fontstyle': 'italic'})
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_yticklabels(ax.get_yticks(), fontsize=35)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    else:
        ax.tick_params(axis="y", which="both", left=True, labelleft=False)
    if row == nrows - 1 or (len(sites_range)%2==1 and column == ncols-1):
        ax.set_xticks(np.linspace(-math.pi/2, math.pi/2, 3), ['-1/2', '0', '1/2'], fontsize=32)
        ax.set_xlabel(r'$k_x/\pi$', fontdict={'fontsize': 65, 'family': 'serif', 'fontstyle': 'italic'})
    else:
        ax.set_xticks(np.linspace(-math.pi/2, math.pi/2, 3), ['-1/2', '0', '1/2'])

        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)  
    print(row, column)    
    ax.text(0.03, 0.97, "("+str(alph_index)+str(row)+")"+r"$n={}$".format(site_B+1), color=text_color, fontfamily='serif', fontsize=48, transform=ax.transAxes, verticalalignment="top")
    
    #Plot the energy levels:
    if Plot_energy_levels:
        for n in range(0, num_Plot_level):
            E_n = [E_ks[k_n][n] + w_shift for k_n in range(num_exci_k)]
            ax.plot(np.linspace(start=extent[0]+delta_k/2, stop=extent[1]-delta_k/2, num=num_exci_k, endpoint=True), E_n, c=line_color, lw=2.5, ls='-', alpha=alpha_energy)
if len(sites_range)%2==1:
    fig.delaxes(Axes[-1, -1])
cbar = fig.colorbar(im, ax=Axes, orientation='vertical', fraction=0.06, pad=0.015)
cbar.ax.tick_params(labelsize=30)
# cbar.set_label("Spectral Intensity of Bosonic Hofstadter")
cbar.set_label(r'$A(k_x,\omega)$', fontdict={'fontsize': 45, 'family': 'serif'})

# Save the figure
plt.savefig(DataPath+'/Ly_'+str(Ly)+'_Charge'+str(charge_sector)+'_Vtrap'+str(V_trap)+'_BD'+str(Bond_Dim)+'_etaw'+str(eta_w)+'_eta_k'+str(eta_k)+str(cmap_type)+'_site_'+str(sites_range[0])+'_to_'+str(sites_range[-1])+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.savefig(DataPath+'/Ly_'+str(Ly)+'_Charge'+str(charge_sector)+'_Vtrap'+str(V_trap)+'_BD'+str(Bond_Dim)+'_etaw'+str(eta_w)+'_eta_k'+str(eta_k)+str(cmap_type)+'_site_'+str(sites_range[0])+'_to_'+str(sites_range[-1])+'.svg', bbox_inches='tight', pad_inches=0.1)
plt.close()

