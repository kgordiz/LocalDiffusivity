#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from IPython.core.display import display, HTML
from IPython.display import display, HTML
display(HTML("<style>.container { width: 98% !important; }</style>"))


# In[ ]:


import numpy as np
import os
import multiprocessing
num_processes = multiprocessing.cpu_count()  # Adjust this to the desired number of processes
import scipy.stats
import matplotlib
matplotlib.use('Agg')

# In[ ]:


def perform_PBC(L, dx_direct):
    L0 = np.sqrt(np.sum(np.power(L[0,:], 2)))
    L1 = np.sqrt(np.sum(np.power(L[1,:], 2)))
    L2 = np.sqrt(np.sum(np.power(L[2,:], 2)))
    L0hat = L[0,:]/L0
    L1hat = L[1,:]/L1
    L2hat = L[2,:]/L2
    #dx_cartesian  = dx_direct @ L # Conversion from direct to cartesian
    dx_cartesian  = dx_direct
    Natoms = dx_cartesian.shape[0]
    for i in range(Natoms):
        # correct along the 1st vector
        d0 = np.dot(dx_cartesian[i, :], L0hat)
        if (d0 >= L0/2):
            dx_cartesian[i,:] -= L[0,:]
        elif (d0 < -L0/2):
            dx_cartesian[i,:] += L[0,:]
        # correct along the 2nd vector
        d1 = np.dot(dx_cartesian[i,:],L1hat)
        if (d1 >= L1/2):
            dx_cartesian[i,:] -= L[1,:]
        elif (d1 < -L1/2):
            dx_cartesian[i,:] += L[1,:]
        # correct along the 3rd vector
        d2 = np.dot(dx_cartesian[i,:],L2hat)
        if (d2 >= L2/2):
            dx_cartesian[i,:] -= L[2,:]
        elif (d2 < -L2/2):
            dx_cartesian[i,:] += L[2,:]
        
    return dx_cartesian


# In[ ]:


def get_equil_info_for_specific_element(poscar_equil_path, desired_element):
    filename = poscar_equil_path
    L = np.zeros((3, 3))
    with open(filename, "r") as f:
        lines = f.readlines()
    
    # Read the cell vectors
    counter = 0
    for i in range(2,5,1):
        templ = lines[i].split() 
        L[counter, :] = [float(templ[0]), float(templ[1]), float(templ[2])]
        counter += 1
    templ = lines[6].split() # for Nspecies
    Number_of_species = len(templ)
    Nspecies = np.zeros(Number_of_species, dtype=int)   

    for i in range(Number_of_species):
        Nspecies[i] = int(templ[i])
    
    templ = lines[5].split() # for elements included 
    countelement = 0
    for i in range(Number_of_species):
        if templ[i] == desired_element:
            desired_element_sequence = countelement
            break
        countelement += 1
    
    Ndesiredelement = Nspecies[desired_element_sequence]
    x0 = np.zeros((Ndesiredelement, 3))
    N1 = 0
    for i in range(desired_element_sequence):
        N1 += Nspecies[i]
    N1 += 8
    #N2 = np.sum(Nspecies[0:-1])
    counter = 0
    #for i in range(N1+N2, N1+N2+Noxygen):
    for i in range(N1, N1+Ndesiredelement):
        templ = lines[i].split()
        x0[counter, :] = [float(templ[0]), float(templ[1]), float(templ[2])]
        counter += 1
    return x0, L


# In[ ]:


def get_coords_vs_time(xyz_path, Ndesiredelement):
    filename = xyz_path
    with open(filename, "r") as f:
        lines = f.readlines()
    
    #templ = lines[6].split() # for Nspecies
    #Number_of_species = len(templ)
    #Nspecies = np.zeros(Number_of_species, dtype=int)
    #for i in range(Number_of_species):
    #    Nspecies[i] = int(templ[i])

    #templ = lines[5].split() # for elements included 
    #countelement = 0
    #for i in range(Number_of_species):
    #    if templ[i] == desired_element:
    #        desired_element_sequence = countelement
    #        break
    #    countelement += 1
    
    #Ndesiredelement = Nspecies[desired_element_sequence]
    
    #N1 = 7
    N2 = 2 # Rejecting reading the first and second lines in each block in the xyz file
    Nheaders = 0 # No header in the file that does not repeat in the xyz file
    Nsnapshots = int((len(lines)-Nheaders)/(Ndesiredelement+N2))
    #Nsnapshots = 100 ################################################################### CHANGE THIS LATER
    x_total = np.zeros((Nsnapshots*Ndesiredelement, 3))
    counter = 0
    for i in range(Nsnapshots):
        N1 = Nheaders + i*(Ndesiredelement+N2)
        for j in range(N1+N2, N1+N2+Ndesiredelement):
            #print(j)
            #print(lines[j])
            templ = lines[j].split()
            
            x_total[counter, :] = [float(templ[1]), float(templ[2]), float(templ[3])] # Let's not read the atomtype
            counter += 1

    return x_total, Nsnapshots


# In[ ]:


def get_x_info(xdatcar_path, desired_element):
    filename = xdatcar_path
    with open(filename, "r") as f:
        lines = f.readlines()
    
    templ = lines[6].split() # for Nspecies
    Number_of_species = len(templ)
    Nspecies = np.zeros(Number_of_species, dtype=int)
    for i in range(Number_of_species):
        Nspecies[i] = int(templ[i])

    templ = lines[5].split() # for elements included 
    countelement = 0
    for i in range(Number_of_species):
        if templ[i] == desired_element:
            desired_element_sequence = countelement
            break
        countelement += 1
    
    Ndesiredelement = Nspecies[desired_element_sequence]
    
    #N1 = 7
    N2 = np.sum(Nspecies[0:desired_element_sequence]) + 1 # Rejecting reading other species and the "Direct configuration" line
    Nheaders = 7
    Natoms = np.sum(Nspecies)
    Nsnapshots = int((len(lines)-Nheaders)/(Natoms+1))
    x_total = np.zeros((Nsnapshots*Ndesiredelement, 3))
    counter = 0
    for i in range(Nsnapshots):
        N1 = Nheaders + i*(Natoms+1)
        for i in range(N1+N2, N1+N2+Ndesiredelement):
            templ = lines[i].split()
            x_total[counter, :] = [float(templ[0]), float(templ[1]), float(templ[2])]
            counter += 1

    return x_total, Ndesiredelement


# In[ ]:


# Read coordinates -- equilibirum -- i.e., sites

NLi = 4
NLa = 4
NTi = 8
NOx = 24

# Input filenames
poscar_equil_path = "./POSCAR"
#xdatcar_path = "./total_XDATCAR_800"
element_for_equil = 'Li'

# read equilibirum POSCAR file
x_sites_direct, L = get_equil_info_for_specific_element(poscar_equil_path, element_for_equil) # x0 is the equilibrium position of oxygen atoms
Nsites = x_sites_direct.shape[0]

xO_equil, L = get_equil_info_for_specific_element(poscar_equil_path, 'O') # Let's get equilibrium O positions too!

# read the generated XDATCAR file
#convert_traj_to_xdatcar(traj_filename)
#x_total, Nox = get_x_info(xdatcar_path, element)


# In[ ]:


######## Related to XDATCAR generation from xyz file

# Read coordinates -- vs. time

xyz_path_La = f"./file_La.xyz"
xyz_path_Li = f"./file_Li.xyz"
xyz_path_Ti = f"./file_Ti.xyz"
xyz_path_O = f"./file_O.xyz"

coords_La, Nsnapshots = get_coords_vs_time(xyz_path_La, NLa)
coords_Li, Nsnapshots = get_coords_vs_time(xyz_path_Li, NLi)
coords_Ti, Nsnapshots = get_coords_vs_time(xyz_path_Ti, NTi)
coords_O, Nsnapshots = get_coords_vs_time(xyz_path_O, NOx)


# In[ ]:


######## Related to XDATCAR generation from xyz file

coords_La_direct = coords_La @ np.linalg.inv(L)
coords_Li_direct = coords_Li @ np.linalg.inv(L)
coords_Ti_direct = coords_Ti @ np.linalg.inv(L)
coords_O_direct = coords_O @ np.linalg.inv(L)

precision = 6  # Set the desired precision

filename_xdatcar = f"./XDATCAR_generated"
#print(filename_out)
with open(filename_xdatcar, 'w') as file:
    file.write("LAMMPS to XDATCAR\n")
    file.write("1.0\n")
    file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(L[0][0], precision, L[0][1], precision, L[0][2], precision))
    file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(L[1][0], precision, L[1][1], precision, L[1][2], precision))
    file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(L[2][0], precision, L[2][1], precision, L[2][2], precision))
    file.write("    {}    {}    {}    {}\n".format('La', 'Li', 'Ti', 'O'))
    file.write("    {}    {}    {}    {}\n".format(NLa, NLi, NTi, NOx))
    for nframe in range(Nsnapshots):
    #for nframe in range(int(Nsnapshots/10000)):
        file.write("Direct configuration=     {}\n".format(nframe))
        for i in range(NLa):
            file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(coords_La_direct[nframe*NLa + i][0], precision, coords_La_direct[nframe*NLa + i][1], precision, coords_La_direct[nframe*NLa + i][2], precision))
        for i in range(NLi):
            file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(coords_Li_direct[nframe*NLi + i][0], precision, coords_Li_direct[nframe*NLi + i][1], precision, coords_Li_direct[nframe*NLi + i][2], precision))
        for i in range(NTi):
            file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(coords_Ti_direct[nframe*NTi + i][0], precision, coords_Ti_direct[nframe*NTi + i][1], precision, coords_Ti_direct[nframe*NTi + i][2], precision))
        for i in range(NOx):
            file.write("    {:.{}f}    {:.{}f}    {:.{}f}\n".format(coords_O_direct[nframe*NOx + i][0], precision, coords_O_direct[nframe*NOx + i][1], precision, coords_O_direct[nframe*NOx + i][2], precision))


# In[ ]:


### Input params

#poscar_equil_path = "./POSCAR_all_Li_for_equil"
xdatcar_path = "./XDATCAR_generated"
element = 'Li'
dt = 1 # fs # timestep between coordinate outputs in MD


# In[ ]:


# read equilibirum POSCAR file
#x_sites, L = get_equil_info_for_specific_element(poscar_equil_path, element) # x0 is the equilibrium position of oxygen atoms
#Nsites = x_sites.shape[0]

# read the XDATCAR file
x_total, NLi = get_x_info(xdatcar_path, element)


# In[ ]:


# Wish downselecting (?)
# x will be the array you need to use for further calculations
Neverythis = 1
Nsnapshots = int(x_total.shape[0]/NLi)
downselected_snapshots = np.arange(0, Nsnapshots, Neverythis)
Nsnaps = len(downselected_snapshots)
print("Number of snapshots that will be used is: %d" %Nsnaps)
x = np.zeros((len(downselected_snapshots)*NLi, 3))
counter = 0
for i in downselected_snapshots:
    idx1total = i*NLi
    idx2total = (i+1)*NLi
    idx1 = counter*NLi
    idx2 = (counter+1)*NLi
    x[idx1:idx2, :] = x_total[idx1total:idx2total, :]
    counter += 1


# In[ ]:


Num_4O_BNs_all = 24
indx_4O_BNs_all = np.empty([Num_4O_BNs_all, 4], dtype=int)

indx_4O_BNs_all [0, :] = [19, 2, 5, 18]
indx_4O_BNs_all [1, :] = [14, 13, 3, 2]
indx_4O_BNs_all [2, :] = [12, 9, 8, 13]
indx_4O_BNs_all [3, :] = [21, 18, 7, 9]
indx_4O_BNs_all [4, :] = [7, 5, 3, 8]
indx_4O_BNs_all [5, :] = [14, 19, 21, 12]
indx_4O_BNs_all [6, :] = [16, 6, 12, 19]# Lx
indx_4O_BNs_all [7, :] = [10, 1, 8, 5]#
indx_4O_BNs_all [8, :] = [1, 6, 13, 2]#
indx_4O_BNs_all [9, :] = [10, 16, 9, 18]#
indx_4O_BNs_all [10, :] = [23, 4, 14, 21]# Ly
indx_4O_BNs_all [11, :] = [11, 15, 3, 7]#
indx_4O_BNs_all [12, :] = [23, 11, 2, 18]#
indx_4O_BNs_all [13, :] = [4, 15, 13, 9]#
indx_4O_BNs_all [14, :] = [17, 24, 12, 8]# Lz
indx_4O_BNs_all [15, :] = [22, 20, 19, 5]#
indx_4O_BNs_all [16, :] = [20, 24, 21, 7]#
indx_4O_BNs_all [17, :] = [22, 17, 14, 3]#
indx_4O_BNs_all [18, :] = [4, 6, 16, 23]# LxLy
indx_4O_BNs_all [19, :] = [15, 1, 10, 11]# 
indx_4O_BNs_all [20, :] = [22, 1, 6, 17]# LxLz
indx_4O_BNs_all [21, :] = [20, 10, 16, 24]#
indx_4O_BNs_all [22, :] = [24, 4, 15, 17]# LyLz
indx_4O_BNs_all [23, :] = [20, 23, 11, 22]#

indx_4O_BNs_all -= 1


# In[ ]:


# We need to calculate which site is connected to which site and through which BN
# Let's just use POSCAR_equil for this purpose -- you need to grab the O coordinates as well.

#BNO4_indx_matrix = np.zeros((Nsites, Nsites, 4), dtype=int)
BNO4_ids_matrix = np.zeros((Nsites, Nsites, 2), dtype=int) # Storing the two BN #s in between i and j sites.
NN12_O_to_site = np.zeros((Nsites, 12), dtype=int)

x_sites  = x_sites_direct @ L # Conversion from direct to cartesian

xO_equil_cartesian = xO_equil @ L # Conversion from direct to cartesian

# First, let's determine the 12 NN O atoms to each site
for i in range(Nsites):
    dist_with_O_atoms = np.zeros(NOx)
    dx_O_atoms = np.zeros((NOx, 3))
    for j in range(NOx):
        #dx_O_atoms[j] = x_sites[i] - coords_O[j]  # You need the O coordinates!!!! Use the first napshot from XDATCAR
        dx_O_atoms[j] = x_sites[i] - xO_equil_cartesian[j]  
    dx_O_atoms_PBC_corrected = perform_PBC(L, dx_O_atoms)
    for j in range(NOx):
        dist_with_O_atoms[j] = np.sqrt(np.sum(np.power(dx_O_atoms_PBC_corrected[j], 2)))
    
    sorted_dists_ids = np.argsort(dist_with_O_atoms)
    NN12_O_to_site[i] = sorted_dists_ids[0: 12]
    #sorted_dists = np.sort(dist_with_O_atoms)
    #print(NN12_O_to_site[0: 12])

Num_4O_BNs = 0
# Then whichever sites have 4 O ids that are the same: they are adjacent sites and the O ids are the 4O BN ids!
for i in range(Nsites):
    for j in range(Nsites):
        if i != j:
            # Find the common elements between the two arrays
            common_elements = np.intersect1d(NN12_O_to_site[i], NN12_O_to_site[j])
            #print(len(common_elements))
            if len(common_elements) == 8:
                third_elem = 0 # third indx in the BNO4_ids_matrix array
                for nbn in range(Num_4O_BNs_all):
                    light = 0
                    for nbn_idx in range(4):
                        if np.isin(indx_4O_BNs_all[nbn, nbn_idx], common_elements):
                            light += 1
                    #print(indx_4O_BNs_all[nbn, :], common_elements, light)
                    if light  == 4:
                        #print('k', nbn)
                        BNO4_ids_matrix[i, j, third_elem] = nbn
                        Num_4O_BNs += 1
                        third_elem += 1
                    if third_elem == 2:
                        break
            elif len(common_elements) > 8:
                print(i, j, len(common_elements))
                #print("Oh Sh*t!!!!!")


# In[ ]:


# Let's simplify the meaningful BNs!! according to the pathways!
BN_pathway_desired = np.empty((0, 3), dtype=int)
for i in range(Nsites):
    for j in range(Nsites):
        if j > i:
            if (BNO4_ids_matrix[i, j] != 0).any():
                print(i, j, BNO4_ids_matrix[i, j])
                BN_pathway_desired = np.vstack([BN_pathway_desired, [i, j, BNO4_ids_matrix[i, j, 0]]])
                BN_pathway_desired = np.vstack([BN_pathway_desired, [i, j, BNO4_ids_matrix[i, j, 1]]])
print(BN_pathway_desired)
print(BN_pathway_desired.shape)


# In[ ]:


######## Related to site info

def split_into_chunks(m_range, num_processes):
    chunk_size, remainder = divmod(len(m_range), num_processes)
    m_range = list(m_range)  # Convert the range to a list
    chunks = [m_range[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_processes)]    
    return chunks

def distance_all_call(chunk, Nsites, NLi, x_sites, x, L):
    dx = np.zeros((Nsites, 3))
    dx_cartesian = np.zeros((Nsites, 3))
    dist = np.zeros(Nsites)
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    
    distance_all_snaps_temp = np.zeros((Nsnapshots3 * NLi, Nsites))

    x_chunk = x[chunk[0]*NLi:(chunk[-1]+1)*NLi:, :]
    for n in range(Nsnapshots3):
        for i in range(NLi):
            for j in range(Nsites):
                dx[j, :] = x_chunk[n*NLi+i, :] - x_sites[j, :]
            dx  = dx @ L # Conversion from direct to cartesian
            dx_cartesian = perform_PBC(L, dx) # this includes PBC correction, too.
            for j in range(Nsites):
                dist[j] = np.sqrt(np.sum(np.power(dx_cartesian[j, :], 2)))
            distance_all_snaps_temp[n*NLi + i, :] = dist

    return distance_all_snaps_temp

def one_over_distance2_all_call(chunk, NLi, distance_all_snaps): ############# Correction needed: Nsites is used in the function but has not been passed to it!
    #dx = np.zeros((Nsites, 3))
    #dx_cartesian = np.zeros((Nsites, 3))
    #dist = np.zeros(Nsites)
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    
    one_over_distance2_all_snaps_temp = np.zeros((Nsnapshots3 * NLi, Nsites))

    distance_all_snaps_chunk = distance_all_snaps[chunk[0]*NLi:(chunk[-1]+1)*NLi:, :]
    for n in range(Nsnapshots3):
        for i in range(NLi):
            zero_indices = np.where(distance_all_snaps_chunk[n*NLi + i, :] == 0)
            if zero_indices[0].size > 0:
                one_over_distance2_all_snaps[n*NLi + i, zero_indices[0]] = 1
            else:
                usethissum = np.sum(np.power(distance_all_snaps_chunk[n*NLi + i, :], -2))
                for j in range(Nsites):
                    one_over_distance2_all_snaps_temp[n*NLi + i, j] = 1./np.square(distance_all_snaps_chunk[n*NLi + i, j]) / usethissum

    return one_over_distance2_all_snaps_temp

# Calculating all the distances to all the sites
distance_all_snaps = np.zeros((Nsnaps * NLi, Nsites))
m_range = range(Nsnaps)
chunks = split_into_chunks(m_range, num_processes)

with multiprocessing.Pool(num_processes) as pool:
    results = pool.starmap(distance_all_call, [(chunk, Nsites, NLi, x_sites_direct, x, L) for chunk in chunks])

distance_all_snaps = np.concatenate(results, axis=0)

# Making the normalized 1/d^2 distance array
one_over_distance2_all_snaps = np.zeros((Nsnaps * NLi, Nsites))
m_range = range(Nsnaps)
chunks = split_into_chunks(m_range, num_processes)

with multiprocessing.Pool(num_processes) as pool:
    results = pool.starmap(one_over_distance2_all_call, [(chunk, NLi, distance_all_snaps) for chunk in chunks])

one_over_distance2_all_snaps = np.concatenate(results, axis=0)

# Making the normalized 1/d^2 distance array
topWhat = 3
normalized_top_what_largest = np.zeros((Nsnaps * NLi, topWhat))
indices_top_what_largest = np.zeros((Nsnaps * NLi, topWhat), dtype=int)

for n in range(Nsnaps):
    if n%5000 == 0:
        print(n)
    for i in range(NLi):
        arr = one_over_distance2_all_snaps[n*NLi + i, :]
        arr_sorted = np.sort(arr)
        top_what_largest = arr_sorted[-topWhat:]
        indices_top_what_largest[n*NLi + i, :] = np.argsort(arr)[-topWhat:]
        normalized_top_what_largest[n*NLi + i, :] = top_what_largest / np.sum(top_what_largest)

# Write the position of hopping atoms into file
x_cartesian = x @ L # Conversion from direct to cartesian # If we are reading XDATCAR with direct coordinates, the conversion should happen!
#x_cartesian = x
filename = "./r_hopping_atom_cartesian" + ".txt"
f_hopping_atom_r = open(filename, "w")
for nt in range(Nsnaps):
    for i in range(NLi):
        f_hopping_atom_r.write("%d %f %f %f\n" %(2, x_cartesian[nt*NLi + i, 0], x_cartesian[nt*NLi + i, 1], x_cartesian[nt*NLi + i, 2]))
f_hopping_atom_r.close()

# Write the topwhat largest numbers into file (value in one file and index in another)
# values
filename = "./site_info_continuous_spectrum_topwhat_values" + ".txt"
f_hopping_atom_v = open(filename, "w")
for n in range(Nsnaps):
    for i in range(NLi):    
        for j in range(topWhat):
            f_hopping_atom_v.write("%f " %(normalized_top_what_largest[n*NLi + i, j]))
        f_hopping_atom_v.write("\n")
f_hopping_atom_v.close()

# indices
filename = "./site_info_continuous_spectrum_topwhat_indices" + ".txt"
f_hopping_atom_i = open(filename, "w")
for n in range(Nsnaps):
    for i in range(NLi):    
        for j in range(topWhat):
            f_hopping_atom_i.write("%d " %(indices_top_what_largest[n*NLi + i, j]))
        f_hopping_atom_i.write("\n")
f_hopping_atom_i.close()


# In[ ]:


aa = np.loadtxt('site_info_continuous_spectrum_topwhat_indices.txt')
print(aa.shape)

import numpy as np
import matplotlib.pyplot as plt

# Assuming aa is your NumPy array
# For demonstration, let's create a sample array
#aa = np.random.randn(1000)  # Generating 1000 random values

# Creating a histogram
plt.hist(aa.flatten(), bins=30, edgecolor='black')  # Adjust the number of bins as needed
#plt.hist(aa, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Histogram of aa')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.savefig('Figure_site_ocuupancy_histogram.png')


# In[ ]:


# Finding the correct bonding/connections for angles!

xO_equil_cartesian = xO_equil @ L # Conversion from direct to cartesian

BN_angle_ids = np.zeros((Num_4O_BNs_all, 4, 2), dtype=int) # These ids are only with respect to the indx_4O_BNs_all array!

for nBN in range(Num_4O_BNs_all):
    for nangle in range(4):
        dist_with_others = np.zeros(4)
        dx_O_atoms = np.zeros((4, 3))
        for nangle_other in range(4):
            dx_O_atoms[nangle_other] = xO_equil_cartesian[indx_4O_BNs_all[nBN, nangle_other]] - xO_equil_cartesian[indx_4O_BNs_all[nBN, nangle]] 
        dx_O_atoms_PBC_corrected = perform_PBC(L, dx_O_atoms)
        for nangle_other in range(4):
            dist_with_others[nangle_other] = np.sqrt(np.sum(np.power(dx_O_atoms_PBC_corrected[nangle_other], 2)))
        #print(dist_with_others)
        sorted_dists_ids = np.argsort(dist_with_others)
        BN_angle_ids[nBN, nangle, :] = sorted_dists_ids[1: 3]
        #print(dist_with_others[sorted_dists_ids[1: 3]])


# In[ ]:


# Finding the angles in a BN for timestep zero!

import numpy as np

def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (norm_vector1 * norm_vector2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

BN_angles = np.zeros((Num_4O_BNs_all, 4))

for nBN in range(Num_4O_BNs_all):
    for nangle in range(4):
        dx_O_atoms = np.zeros((2, 3))
        for nangle_other in range(2):
            dx_O_atoms[nangle_other] = xO_equil_cartesian[indx_4O_BNs_all[nBN, BN_angle_ids[nBN, nangle, nangle_other]]] - xO_equil_cartesian[indx_4O_BNs_all[nBN, nangle]] 
        dx_O_atoms_PBC_corrected = perform_PBC(L, dx_O_atoms)
        vector1 = np.array(dx_O_atoms_PBC_corrected[0])
        vector2 = np.array(dx_O_atoms_PBC_corrected[1])
        angle_degrees = angle_between_vectors(vector1, vector2)
        BN_angles[nBN, nangle] = angle_degrees
        #print(nBN, nangle, angle_degrees)
        


# In[ ]:


# Finding the angles in a BN for all timesteps!

def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_angle = dot_product / (norm_vector1 * norm_vector2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def split_into_chunks(m_range, num_processes):
    chunk_size, remainder = divmod(len(m_range), num_processes)
    m_range = list(m_range)  # Convert the range to a list
    chunks = [m_range[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_processes)]    
    return chunks

def angle_call(chunk, coords_O, Num_4O_BNs_all, NOx, indx_4O_BNs_all):
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    coords_O_chunk = coords_O[chunk[0]*NOx:(chunk[-1]+1)*NOx:, :]
    BN_angles_time_temp = np.zeros((Nsnapshots3, Num_4O_BNs_all, 4))
    for n in range(Nsnapshots3):
        #if n%1000 == 0:
        #    print(n)
        for nBN in range(Num_4O_BNs_all):
            for nangle in range(4):
                dx_O_atoms = np.zeros((2, 3))
                for nangle_other in range(2):
                    #dx_O_atoms[nangle_other] = xO_equil_cartesian[indx_4O_BNs_all[nBN, BN_angle_ids[nBN, nangle, nangle_other]]] - xO_equil_cartesian[indx_4O_BNs_all[nBN, nangle]]
                    dx_O_atoms[nangle_other] = coords_O_chunk[n * NOx + indx_4O_BNs_all[nBN, BN_angle_ids[nBN, nangle, nangle_other]]] - coords_O_chunk[n * NOx + indx_4O_BNs_all[nBN, nangle]] 
                dx_O_atoms_PBC_corrected = perform_PBC(L, dx_O_atoms)
                vector1 = np.array(dx_O_atoms_PBC_corrected[0])
                vector2 = np.array(dx_O_atoms_PBC_corrected[1])
                angle_degrees = angle_between_vectors(vector1, vector2)
                BN_angles_time_temp[n, nBN, nangle] = angle_degrees
                #print(nBN, nangle, angle_degrees)
    return BN_angles_time_temp   

'''def distance_all_call(chunk, Nsites, NLi, x_sites, x, L):
    dx = np.zeros((Nsites, 3))
    dx_cartesian = np.zeros((Nsites, 3))
    dist = np.zeros(Nsites)
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    
    distance_all_snaps_temp = np.zeros((Nsnapshots3 * NLi, Nsites))

    x_chunk = x[chunk[0]*NLi:(chunk[-1]+1)*NLi:, :]
    for n in range(Nsnapshots3):
        for i in range(NLi):
            for j in range(Nsites):
                dx[j, :] = x_chunk[n*NLi+i, :] - x_sites[j, :]
            dx  = dx @ L # Conversion from direct to cartesian
            dx_cartesian = perform_PBC(L, dx) # this includes PBC correction, too.
            for j in range(Nsites):
                dist[j] = np.sqrt(np.sum(np.power(dx_cartesian[j, :], 2)))
            distance_all_snaps_temp[n*NLi + i, :] = dist

    return distance_all_snaps_temp'''

# Calculating all the distances to all the sites
#distance_all_snaps = np.zeros((Nsnaps * NLi, Nsites))
#BN_angles_time = np.zeros((Nsnaps, Num_4O_BNs_all, 4))
m_range = range(Nsnaps)
chunks = split_into_chunks(m_range, num_processes)

with multiprocessing.Pool(num_processes) as pool:
    results = pool.starmap(angle_call, [(chunk, coords_O, Num_4O_BNs_all, NOx, indx_4O_BNs_all) for chunk in chunks])

BN_angles_time = np.concatenate(results, axis=0)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming area_effective_all is defined and contains your data

plt.figure(figsize=(24, 10))
countit = 0
colors = plt.cm.viridis(np.linspace(0, 1, BN_angles_time.shape[1]))  # Generate colors based on number of i values

for i in range(BN_angles_time.shape[1]):
    color = colors[i]  # Get color based on i value
    for j in range(BN_angles_time.shape[2]):
        x_values = [int(countit)] * BN_angles_time.shape[0]
        y_values = BN_angles_time[:, i, j]
        plt.scatter(x_values, y_values, alpha=0.5, color=color)  # Set color for scatter plot

        # Calculate mean and standard deviation
        mean = np.mean(y_values)
        std_dev = np.std(y_values)

        # Plot error bars with larger size and black color
        plt.errorbar(countit, mean, yerr=std_dev, fmt='o', color='black', markersize=8)
        
        countit += 1

plt.title('Distribution of bottleneck areas')
plt.xlabel('Bottleneck #')
plt.ylabel('Bottleneck area ($\\AA^2$)')
plt.show()
plt.savefig('Figure_BN_num_angle_all.png')


# In[ ]:


# Let's downselect only for the BNs that matter for Li hop

BN_angles_time_that_matters = np.zeros((Nsnaps, BN_pathway_desired.shape[0], 4))

countit = 0
for bn_matters in range(BN_pathway_desired.shape[0]):
    BN_angles_time_that_matters[:, countit, :] = BN_angles_time[:, bn_matters, :]
    countit += 1

plt.figure(figsize=(24, 10))
countit = 0
#colors = plt.cm.viridis_r(np.linspace(0, 1, BN_angles_time_that_matters.shape[1]))  # Generate colors based on number of i values
colors_column = np.linspace(0, 1, BN_angles_time_that_matters.shape[1])  # Generate colors based on number of i values
#colors = np.zeros((BN_angles_time_that_matters.shape[1], 3))
colors = np.ones((BN_angles_time_that_matters.shape[1], 3))*.5
colors[:, 2] = colors_column

for i in range(BN_angles_time_that_matters.shape[1]):
    color = colors[i]  # Get color based on i value
    for j in range(BN_angles_time_that_matters.shape[2]):
        x_values = [int(countit)] * BN_angles_time_that_matters.shape[0]
        y_values = BN_angles_time_that_matters[:, i, j]
        plt.scatter(x_values, y_values, alpha=0.2, color=color)  # Set color for scatter plot

        # Calculate mean and standard deviation
        mean = np.mean(y_values)
        std_dev = np.std(y_values)

        # Plot error bars with larger size and black color
        #plt.errorbar(countit, mean, yerr=std_dev, fmt='o', color='black', markersize=8)
        plt.errorbar(countit, mean, yerr=std_dev, fmt='o', color='black', markersize=14, linewidth=4)  # Set line width for error bars
        
        countit += 1

plt.title('Distribution of bottleneck areas')
plt.xlabel('Bottleneck #')
plt.ylabel('Bottleneck area ($\\AA^2$)')
plt.show()
plt.savefig('Figure_BN_num_angle_the_ones_that_matter.png')


# In[ ]:


## FFT Corr calculation function

def cross_correlation_fft_zero_padding_normalized(var1, var2):
    # Make sure both variables are of the same length
    if len(var1) != len(var2):
        raise ValueError("Input variables must have the same length")

    # Calculate the FFT of the variables with zero-padding
    fft_size = 2 * len(var1)
    fft_var1 = np.fft.fft(var1, fft_size)
    fft_var2 = np.fft.fft(var2, fft_size)

    # Calculate the complex conjugate of the second variable
    conj_fft_var2 = np.conj(fft_var2)

    # Calculate the cross-correlation in the frequency domain
    cross_corr_freq = fft_var1 * conj_fft_var2

    # Perform inverse FFT to get back to the time domain
    cross_corr_time = np.fft.ifft(cross_corr_freq)

    # Normalize the result
    #cross_corr_time /= fft_size  # Normalization #1 (FFTW) -- not needed for this fft from numpy actually 
    cross_corr_time /= np.concatenate([np.arange(len(var1), 0, -1), np.arange(1, len(var1) + 1)]) # Normalization #2 (Allen-Tidesly)
    
    return cross_corr_time.real


# In[ ]:


# Get the atomic velocities

#dt = 1 # fs # timestep between coordinate outputs # After the downselection

vel = np.zeros(((Nsnaps-1) * NLi, 3))
dx = np.zeros(((Nsnaps-1) * NLi, 3))

for ns in range(Nsnaps-1):
    for i in range(NLi):
        dx[ns * NLi + i, :] = x[ns * NLi + i, :] - x[(ns+1) * NLi + i, :]
        
dx  = dx @ L # Conversion from direct to cartesian
dx_cartesian = perform_PBC(L, dx)
vel = dx_cartesian / dt


# In[ ]:


# Diffusion analysis before projection on different local environments
# Correlation for all atoms and all xyz direcitons

cross_corr_final = np.zeros((vel[0::NLi, :].shape[0]))

for nLi in range(NLi):
    for alpha in range(3):
        vel_under_consideration = vel[nLi::NLi, alpha]
        #vel_under_consideration = vel[nLi::NLi, alpha] - np.mean(vel[nLi::NLi, alpha])
        # Calculate cross-correlation using FFT with zero-padding

        var1 = vel_under_consideration
        var2 = vel_under_consideration

        cross_corr_result = cross_correlation_fft_zero_padding_normalized(var1, var2)
        #print(cross_corr_result.shape)
        #print(len(vel_under_consideration))
        cross_corr_final += cross_corr_result[0: len(vel_under_consideration)]
        
cross_corr_final /= (NLi * 3 * 10) # 10 is for conversion from A^2/fs to cm^2/s
print(cross_corr_final.shape)

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

# Generate x-axis values (time * 0.001)
x_values = np.arange(len(cross_corr_final)) * dt # Let's keep the values on the x-axis as fs

# Calculate the cumulative integral using cumtrapz
cumulative_integral_values = cumtrapz(cross_corr_final, x=x_values, initial=0)
cumulative_integral_values_2 = cumtrapz(cumulative_integral_values, x=x_values, initial=0)

print(cumulative_integral_values_2.shape)

begin_data = 100
end_data = 20000

# Plot double integ autocorr
plt.figure(figsize=(8, 6), facecolor='w')
#plt.plot(x / 1000, y, 'b', linewidth=1.5)

plt.plot(x_values / 1000, cumulative_integral_values_2, color='b')

x_for_plot = x_values
y_for_plot = cumulative_integral_values_2

# Perform linear fit
y_for_fit = y_for_plot[begin_data:end_data]
x_for_fit = x_for_plot[begin_data:end_data]
p = np.polyfit(x_for_fit, y_for_fit, 1)

# Plot the linear fit
plt.plot(x_for_fit / 1000, np.polyval(p, x_for_fit), 'r--', linewidth=1.5)

# Set plot properties
plt.xlabel('$\Delta t$(ps)')
plt.ylabel('MSD (from double integ autocorr) ($\\AA^2$)')
plt.title('Mean Squared Displacement vs. Time')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.axis('square')
#plt.show()
plt.savefig('Figure_MSD_before_projection.png')

# Display the slope in cm^2/s
D_in_cm2_per_s = p[0] * 0.5 # the 0.5 coeff is for double integral with double counting! (slide 13 in the pdf by Zabaras) # NO need for 0.1 is to convert from A^2/fs to cm^2/s (applied previously) 
print("D (cm^2/s):", D_in_cm2_per_s)

# Project vel on different local environments - serial

Env_cont_val = np.loadtxt('./site_info_continuous_spectrum_topwhat_values.txt')
Env_cont_indx = np.loadtxt('./site_info_continuous_spectrum_topwhat_indices.txt', dtype=int)

vel_env_prjctd = np.zeros((Nsnaps-1, Nsites, Nsites, 2, NLi, 3))

#sum_for_normalization = 0

for nLi in range(NLi): # Loop over Li atoms
    #print(nLi)
    for alpha in range(3): # Loop over x, y, and z
        vel_under_consideration = vel[nLi::NLi, alpha]
        Nsnaps_vel = vel_under_consideration.shape[0] # This should be Nsnaps-1
        #if Nsnaps_vel != Nsnaps-1:
        #    print('Sh*t')
        for nt in range(Nsnaps_vel):
            sum_for_normalization = 0
            indx1 = nt * NLi + nLi
            indx2 = (nt+1) * NLi + nLi 
            for envi in range(topWhat):
                for envj in range(topWhat):
                    idx_i_env = Env_cont_indx[indx1][envi]
                    idx_j_env = Env_cont_indx[indx2][envj]
                    # BN_pathway_desired
                    #if idx_i_env != idx_j_env:
                    result1 = np.any((BN_pathway_desired[:, 0:2] == idx_i_env), axis=1)
                    result2 = np.any((BN_pathway_desired[:, 0:2] == idx_j_env), axis=1)
                    result3 = result1 & result2
                    if idx_i_env != idx_j_env and result3.any():
                        #if idx_i_env > 2 or idx_j_env > 2:
                        #    print(idx_i_env, idx_j_env)
                        sum_for_normalization += Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]
        
            for envi in range(topWhat):
                for envj in range(topWhat):
                    idx_i_env = Env_cont_indx[indx1][envi]
                    idx_j_env = Env_cont_indx[indx2][envj]
                    result1 = np.any((BN_pathway_desired[:, 0:2] == idx_i_env), axis=1)
                    result2 = np.any((BN_pathway_desired[:, 0:2] == idx_j_env), axis=1)
                    result3 = result1 & result2
                    if idx_i_env != idx_j_env and result3.any():
                        # the two BNs in b/w these sites are here:
                        BN0 = BNO4_ids_matrix[idx_i_env, idx_j_env, 0]
                        BN1 = BNO4_ids_matrix[idx_i_env, idx_j_env, 1]
                        ids_O_0 = indx_4O_BNs_all[BN0]
                        ids_O_1 = indx_4O_BNs_all[BN1]
                        if (len(ids_O_0) != 4 or len(ids_O_1) != 4):
                            print('Sh*t')
                        # BN0
                        dist0_array = np.zeros((len(ids_O_0), 3))
                        for indxO in range(len(ids_O_0)):
                            dist0_array[indxO, :] = x[NLi*nt + nLi, :] - xO_equil[ids_O_0[indxO], :]
                        dist0_array  = dist0_array @ L # Conversion from direct to cartesian
                        dist0_array_PBC_corrected = perform_PBC(L, dist0_array)
                        dist0 = 0
                        for indxO in range(len(ids_O_0)):
                            dist0 += np.sqrt(np.sum(np.power(dist0_array_PBC_corrected[indxO, :], 2)))
                        # BN1
                        dist1_array = np.zeros((len(ids_O_1), 3))
                        for indxO in range(len(ids_O_1)):
                            dist1_array[indxO, :] = x[NLi*nt + nLi, :] - xO_equil[ids_O_1[indxO], :]
                        dist1_array  = dist1_array @ L # Conversion from direct to cartesian
                        dist1_array_PBC_corrected = perform_PBC(L, dist1_array)
                        dist1 = 0
                        for indxO in range(len(ids_O_1)):
                            dist1 += np.sqrt(np.sum(np.power(dist1_array_PBC_corrected[indxO, :], 2)))    
                        # Let's comapre dist0 and dist0: whichever we are closer to, we take it as the BN nearby...
                        if dist1 > dist0:
                            BN_chosen = 0
                        else:
                            BN_chosen = 1
                        #if idx_i_env != idx_j_env and sum_for_normalization != 0:
                        #vel_env_prjctd[nt][idx_i_env][idx_j_env][BN_chosen][nLi][alpha] += vel_under_consideration[nt] * (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) / sum_for_normalization
                        vel_env_prjctd[nt, idx_i_env, idx_j_env, BN_chosen, nLi, alpha] += vel_under_consideration[nt] * (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) / sum_for_normalization
                    #else:    ## Think about this -- probably not needed (?) as sum_for_normalization will always have a non-zero value?!!!
                    #    vel_env_prjctd[idx_i_env][idx_j_env][nLi][alpha][nt] += vel_under_consideration[nt] * (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) / sum_for_normalization

# In[ ]:


######## Project vel on different local environments - parallel

Env_cont_val = np.loadtxt('./site_info_continuous_spectrum_topwhat_values.txt')
Env_cont_indx = np.loadtxt('./site_info_continuous_spectrum_topwhat_indices.txt', dtype=int)

def split_into_chunks(m_range, num_processes):
    chunk_size, remainder = divmod(len(m_range), num_processes)
    m_range = list(m_range)  # Convert the range to a list
    chunks = [m_range[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_processes)]    
    return chunks

def vel_env_prjctd_call(chunk, Nsnaps, NLi, Nsites, vel, Env_cont_val, Env_cont_indx, topWhat, BNO4_ids_matrix, indx_4O_BNs_all, x, xO_equil):

    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    vel_env_prjctd_temp = np.zeros((Nsnapshots3, Nsites, Nsites, 2, NLi, 3))
    
    x_chunk = x[chunk[0]*NLi:(chunk[-1]+1)*NLi:, :]
    vel_chunk = vel[chunk[0]*NLi:(chunk[-1]+1)*NLi:, :]
    #print(x_chunk.shape)
    
    for nLi in range(NLi): # Loop over Li atoms
        #print(nLi)
        for alpha in range(3): # Loop over x, y, and z
            #vel_under_consideration = vel[nLi::NLi, alpha]
            vel_under_consideration = vel_chunk[nLi::NLi, alpha]
            #Nsnaps_vel = vel_under_consideration.shape[0] # This should be Nsnaps-1
            #if Nsnaps_vel != Nsnaps-1:
            #    print('Sh*t')
            for nt in range(Nsnapshots3):
                sum_for_normalization = 0
                #indx1 = nt * NLi + nLi
                #indx2 = (nt+1) * NLi + nLi 
                thistime = chunk[nt]
                indx1 = thistime * NLi + nLi
                indx2 = (thistime + 1) * NLi + nLi 
                for envi in range(topWhat):
                    for envj in range(topWhat):
                        idx_i_env = Env_cont_indx[indx1][envi]
                        idx_j_env = Env_cont_indx[indx2][envj]
                        # BN_pathway_desired
                        #if idx_i_env != idx_j_env:
                        result1 = np.any((BN_pathway_desired[:, 0:2] == idx_i_env), axis=1)
                        result2 = np.any((BN_pathway_desired[:, 0:2] == idx_j_env), axis=1)
                        result3 = result1 & result2
                        if idx_i_env != idx_j_env and result3.any():
                            #if idx_i_env > 2 or idx_j_env > 2:
                            #    print(idx_i_env, idx_j_env)
                            sum_for_normalization += Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]

                for envi in range(topWhat):
                    for envj in range(topWhat):
                        idx_i_env = Env_cont_indx[indx1][envi]
                        idx_j_env = Env_cont_indx[indx2][envj]
                        result1 = np.any((BN_pathway_desired[:, 0:2] == idx_i_env), axis=1)
                        result2 = np.any((BN_pathway_desired[:, 0:2] == idx_j_env), axis=1)
                        result3 = result1 & result2
                        if idx_i_env != idx_j_env and result3.any():
                            # the two BNs in b/w these sites are here:
                            BN0 = BNO4_ids_matrix[idx_i_env, idx_j_env, 0]
                            BN1 = BNO4_ids_matrix[idx_i_env, idx_j_env, 1]
                            ids_O_0 = indx_4O_BNs_all[BN0]
                            ids_O_1 = indx_4O_BNs_all[BN1]
                            if (len(ids_O_0) != 4 or len(ids_O_1) != 4):
                                print('Sh*t')
                            # BN0
                            dist0_array = np.zeros((len(ids_O_0), 3))
                            for indxO in range(len(ids_O_0)):
                                dist0_array[indxO, :] = x_chunk[NLi*nt + nLi, :] - xO_equil[ids_O_0[indxO], :]
                            dist0_array  = dist0_array @ L # Conversion from direct to cartesian
                            dist0_array_PBC_corrected = perform_PBC(L, dist0_array)
                            dist0 = 0
                            for indxO in range(len(ids_O_0)):
                                dist0 += np.sqrt(np.sum(np.power(dist0_array_PBC_corrected[indxO, :], 2)))
                            # BN1
                            dist1_array = np.zeros((len(ids_O_1), 3))
                            for indxO in range(len(ids_O_1)):
                                dist1_array[indxO, :] = x_chunk[NLi*nt + nLi, :] - xO_equil[ids_O_1[indxO], :]
                            dist1_array  = dist1_array @ L # Conversion from direct to cartesian
                            dist1_array_PBC_corrected = perform_PBC(L, dist1_array)
                            dist1 = 0
                            for indxO in range(len(ids_O_1)):
                                dist1 += np.sqrt(np.sum(np.power(dist1_array_PBC_corrected[indxO, :], 2)))    
                            # Let's comapre dist0 and dist0: whichever we are closer to, we take it as the BN nearby...
                            if dist1 > dist0:
                                BN_chosen = 0
                            else:
                                BN_chosen = 1
                            #if idx_i_env != idx_j_env and sum_for_normalization != 0:
                            #vel_env_prjctd[nt][idx_i_env][idx_j_env][BN_chosen][nLi][alpha] += vel_under_consideration[nt] * (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) / sum_for_normalization
                            vel_env_prjctd_temp[nt, idx_i_env, idx_j_env, BN_chosen, nLi, alpha] += vel_under_consideration[nt] * (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) / sum_for_normalization
                        #else:    ## Think about this -- probably not needed (?) as sum_for_normalization will always have a non-zero value?!!!
                        #    vel_env_prjctd[idx_i_env][idx_j_env][nLi][alpha][nt] += vel_under_consideration[nt] * (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) / sum_for_normalization

    return vel_env_prjctd_temp

vel_env_prjctd = np.zeros((Nsnaps-1, Nsites, Nsites, 2, NLi, 3))

m_range = range(Nsnaps-1)
chunks = split_into_chunks(m_range, num_processes)

with multiprocessing.Pool(num_processes) as pool:
    results = pool.starmap(vel_env_prjctd_call, [(chunk, Nsnaps, NLi, Nsites, vel, Env_cont_val, Env_cont_indx, topWhat, BNO4_ids_matrix, indx_4O_BNs_all, x, xO_equil) for chunk in chunks])

vel_env_prjctd = np.concatenate(results, axis=0)


# In[ ]:


print(vel_env_prjctd.shape)


# In[ ]:


# Diffusion analysis after projection on different local environments

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

# Set a larger figure size
plt.figure(figsize=(10, 10))

# Create a 2x1 subplot
plt.subplot(2, 1, 1)
        
cross_corr_final_totalsum = np.zeros((vel_env_prjctd[:, 0, 0, 0, 0, 0].shape[0]))
Dij = np.zeros((Nsites, Nsites, 2)) # For this specific case, each site is connected to the other site through two BNs!

# Set a colormap and normalize based on the loop index
colormap = plt.cm.get_cmap('viridis')  # You can choose any colormap
#colormap = plt.colormaps['viridis']
norm = plt.Normalize(0, Nsites)  # For color coding based on Nenv = Nsites

#begin_data = 100
#end_data = 10000

for envi in range(Nsites):
    print(envi)
    for envj in range(Nsites):
        for BNindx in range(2):
            cross_corr_final = np.zeros((vel_env_prjctd[:, 0, 0, 0, 0, 0].shape[0]))
            for nLi in range(NLi):
                for alpha in range(3):
                    vel_under_consideration = vel_env_prjctd[:, envi, envj, BNindx, nLi, alpha]
                    vel_under_consideration2 = vel[nLi::NLi, alpha]

                    #print(vel_under_consideration.shape)
                    #print(vel_under_consideration2.shape)

                    var1 = vel_under_consideration
                    var2 = vel_under_consideration2

                    cross_corr_result = cross_correlation_fft_zero_padding_normalized(var1, var2)
                    cross_corr_final += cross_corr_result[0: len(vel_under_consideration)]

            cross_corr_final /= (NLi * 3 * 10) # 10 is for conversion from A^2/fs to cm^2/s
            cross_corr_final_totalsum += cross_corr_final

            # Generate x-axis values (time * 0.001)
            x_values = np.arange(len(cross_corr_final)) * 1.0 # Let's keep the values on the x-axis as fs

            # Calculate the cumulative integral using cumtrapz
            cumulative_integral_values_env_based = cumtrapz(cross_corr_final, x=x_values, initial=0)
            cumulative_integral_values_env_based_2 = cumtrapz(cumulative_integral_values_env_based, x=x_values, initial=0)
            #print(cumulative_integral_values.shape)

            # Assuming x_values, cross_corr_final, and cumulative_integral_values are defined

            # Calculate the cumulative integral using trapz
            #cumulative_integral = trapz(cross_corr_final, x=x_values)

            color = colormap(norm(envi))
            #plt.plot(x_values, cross_corr_final, color='b')
            #plt.plot(x_values, cumulative_integral_values_env_based, color=color)

            ### Moving average
            # Define the window length
            #window_length = 5000
            # Calculate the moving average
            #cumulative_integral_values_moving_average = np.convolve(cumulative_integral_values_env_based, np.ones(window_length)/window_length, mode='valid')
            #x_values_moving_average = np.arange(len(cumulative_integral_values_moving_average)) * 1.0

            plt.plot(x_values / 1000, cumulative_integral_values_env_based_2, color=color)

            x_for_plot = x_values
            y_for_plot = cumulative_integral_values_env_based_2

            # Perform linear fit
            y_for_fit = y_for_plot[begin_data:end_data]
            x_for_fit = x_for_plot[begin_data:end_data]
            p = np.polyfit(x_for_fit, y_for_fit, 1)

            # Plot the linear fit
            #plt.plot(x_for_fit / 1000, np.polyval(p, x_for_fit), 'k--', linewidth=1.5)
            plt.plot(x_for_fit / 1000, np.polyval(p, x_for_fit), 'k--')

            # Display the slope in cm^2/s
            D_in_cm2_per_s = p[0] * 0.5 # the 0.5 coeff is for doeble integral with double counting! (slide 13 in the pdf by Zabaras) # NO need for 0.1 is to convert from A^2/fs to cm^2/s (applied previously) 
            #print("D (cm^2/s):", D_in_cm2_per_s)

            #Dij[envi][envj] = np.mean(cumulative_integral_values_moving_average[begin_data:end_data]) # cm^2/s
            Dij[envi][envj][BNindx] = D_in_cm2_per_s # cm^2/s
            #print("D from int_autocorr (cm^2/s):", D_in_cm2_per_s_from_int_autocorr)

# Calculate the cumulative integral using cumtrapz
cumulative_integral_cross_corr_final_totalsum = cumtrapz(cross_corr_final_totalsum, x=x_values, initial=0)
cumulative_integral_cross_corr_final_totalsum_2 = cumtrapz(cumulative_integral_cross_corr_final_totalsum, x=x_values, initial=0)
plt.plot(x_values / 1000, cumulative_integral_cross_corr_final_totalsum_2, color='black', linestyle='dashed', linewidth=2.5)

x_for_plot = x_values
y_for_plot = cumulative_integral_cross_corr_final_totalsum_2

# Perform linear fit
y_for_fit = y_for_plot[begin_data:end_data]
x_for_fit = x_for_plot[begin_data:end_data]
p = np.polyfit(x_for_fit, y_for_fit, 1)

# Plot the linear fit
plt.plot(x_for_fit / 1000, np.polyval(p, x_for_fit), 'r--', linewidth=1.5)

# Display the slope in cm^2/s
D_in_cm2_per_s = p[0] * 0.5 # the 0.5 coeff is for doeble integral with double counting! (slide 13 in the pdf by Zabaras) # NO need for 0.1 is to convert from A^2/fs to cm^2/s (applied previously) 
print("D (cm^2/s):", D_in_cm2_per_s)

# Set plot properties
plt.xlabel('$\Delta t$(ps)')
plt.ylabel('MSD (from double integ autocorr) ($\\AA^2$)')
#plt.title('Mean Squared Displacement vs. Time')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(2, 1, 2)
# Plot the cumulative integral
plt.plot(x_values / 1000, cumulative_integral_values_2, color='r') # From past

x_for_plot = x_values
y_for_plot = cumulative_integral_values_2

# Perform linear fit
y_for_fit = y_for_plot[begin_data:end_data]
x_for_fit = x_for_plot[begin_data:end_data]
p = np.polyfit(x_for_fit, y_for_fit, 1)

# Plot the linear fit
plt.plot(x_for_fit / 1000, np.polyval(p, x_for_fit), 'r--', linewidth=1.5)

# Display the slope in cm^2/s
D_in_cm2_per_s = p[0] * 0.5 # the 0.5 coeff is for doeble integral with double counting! (slide 13 in the pdf by Zabaras) # NO need for 0.1 is to convert from A^2/fs to cm^2/s (applied previously) 
print("D (cm^2/s):", D_in_cm2_per_s)

# Set plot properties
plt.xlabel('$\Delta t$(ps)')
plt.ylabel('MSD (from double integ autocorr) ($\\AA^2$)')
#plt.title('Mean Squared Displacement vs. Time')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
#plt.show()
plt.savefig('Figure_MSD_after_projection.png')


# In[ ]:


def BN_calculator(Num_4O_BNs, indx_4O_BNs, coords_O, L):
    indx_4O_BNs = indx_4O_BNs.copy()
    #indx_4O_BNs -= 1
    area_effective = np.zeros(Num_4O_BNs)
    dist_ha_BN_center = np.zeros(Num_4O_BNs)
    for num_BN in range(Num_4O_BNs):
        area = np.zeros(4)
        if num_BN < 6:
            #Triangle 1 [0,1,2]
            Tri_indx_sub = [0, 1, 2]
            Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[1]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            #print(Tri_indx)
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[1]]
            dist_cart[1] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[2]]
            dist_cart[2] = coords_O[Tri_indx[1]] - coords_O[Tri_indx[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[0] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))
            #print(area[0])
            #Triangle 2 [0,1,3]
            Tri_indx_sub = [0, 1, 3]
            Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[1]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[1]]
            dist_cart[1] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[2]]
            dist_cart[2] = coords_O[Tri_indx[1]] - coords_O[Tri_indx[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[1] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))

            #Triangle 3 [2,3,0]
            Tri_indx_sub = [2,3,0]
            Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[1]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[1]]
            dist_cart[1] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[2]]
            dist_cart[2] = coords_O[Tri_indx[1]] - coords_O[Tri_indx[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart        
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[2] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))

            #Triangle 4 [2,3,1]
            Tri_indx_sub = [2,3,1]
            Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[1]],
                        indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[1]]
            dist_cart[1] = coords_O[Tri_indx[0]] - coords_O[Tri_indx[2]]
            dist_cart[2] = coords_O[Tri_indx[1]] - coords_O[Tri_indx[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[3] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))
        else:
            coords_O_modified_for_four_corners = np.zeros((4,3))
            coords_O_modified_for_four_corners[0, :] = coords_O[indx_4O_BNs[num_BN, 0], :]
            coords_O_modified_for_four_corners[1, :] = coords_O[indx_4O_BNs[num_BN, 1], :]
            coords_O_modified_for_four_corners[2, :] = coords_O[indx_4O_BNs[num_BN, 2], :]
            coords_O_modified_for_four_corners[3, :] = coords_O[indx_4O_BNs[num_BN, 3], :]
            #if num_BN == 6:
            #    print(coords_O_modified_for_four_corners)
            if num_BN >= 6 and num_BN < 10: # Lx
                Lidx = 0
                if coords_O_modified_for_four_corners[0, Lidx] > L[Lidx,Lidx]/2:
                    coords_O_modified_for_four_corners[0, Lidx] -= L[Lidx,Lidx]
                if coords_O_modified_for_four_corners[1, Lidx] > L[Lidx,Lidx]/2:
                    coords_O_modified_for_four_corners[1, Lidx] -= L[Lidx,Lidx]
                coords_O_modified_for_four_corners[3, Lidx] -= L[Lidx,Lidx]
            elif num_BN >= 10 and num_BN < 14: # Ly
                Lidx = 1
                if coords_O_modified_for_four_corners[0, Lidx] > L[Lidx,Lidx]/2:
                    coords_O_modified_for_four_corners[0, Lidx] -= L[Lidx,Lidx]
                if coords_O_modified_for_four_corners[1, Lidx] > L[Lidx,Lidx]/2:
                    coords_O_modified_for_four_corners[1, Lidx] -= L[Lidx,Lidx]
                coords_O_modified_for_four_corners[3, Lidx] -= L[Lidx,Lidx]
            elif num_BN >= 14 and num_BN < 18: # Lz
                Lidx = 2
                if coords_O_modified_for_four_corners[0, Lidx] > L[Lidx,Lidx]/2:
                    coords_O_modified_for_four_corners[0, Lidx] -= L[Lidx,Lidx]
                if coords_O_modified_for_four_corners[1, Lidx] > L[Lidx,Lidx]/2:
                    coords_O_modified_for_four_corners[1, Lidx] -= L[Lidx,Lidx]
                coords_O_modified_for_four_corners[3, Lidx] -= L[Lidx,Lidx]
            elif num_BN >= 18 and num_BN < 20: # LxLy
                for Lidx in [0, 1]:
                    for Oid in range(4):
                        if coords_O_modified_for_four_corners[Oid, Lidx] > L[Lidx,Lidx]/2:
                            coords_O_modified_for_four_corners[Oid, Lidx] -= L[Lidx,Lidx]
            elif num_BN >= 20 and num_BN < 22: # LxLz
                for Lidx in [0, 2]:
                    for Oid in range(4):
                        if coords_O_modified_for_four_corners[Oid, Lidx] > L[Lidx,Lidx]/2:
                            coords_O_modified_for_four_corners[Oid, Lidx] -= L[Lidx,Lidx]
            #elif num_BN >= 22 and num_BN < 24: # LxLy
            else: # LyLz
                for Lidx in [1, 2]:
                    for Oid in range(4):
                        if coords_O_modified_for_four_corners[Oid, Lidx] > L[Lidx,Lidx]/2:
                            coords_O_modified_for_four_corners[Oid, Lidx] -= L[Lidx,Lidx]
            #if num_BN == 6:
            #    print(coords_O_modified_for_four_corners)
            #Triangle 1 [0,1,2]
            Tri_indx_sub = [0, 1, 2]
            #Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[1]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[1]]
            dist_cart[1] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            dist_cart[2] = coords_O_modified_for_four_corners[Tri_indx_sub[1]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[0] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))

            #Triangle 2 [0,1,3]
            Tri_indx_sub = [0, 1, 3]
            #Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[1]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[1]]
            dist_cart[1] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            dist_cart[2] = coords_O_modified_for_four_corners[Tri_indx_sub[1]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[1] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))

            #Triangle 3 [2,3,0]
            Tri_indx_sub = [2,3,0]
            #Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[1]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[1]]
            dist_cart[1] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            dist_cart[2] = coords_O_modified_for_four_corners[Tri_indx_sub[1]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart        
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[2] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))

            #Triangle 4 [2,3,1]
            Tri_indx_sub = [2,3,1]
            #Tri_indx = [indx_4O_BNs[num_BN, Tri_indx_sub[0]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[1]],
            #            indx_4O_BNs[num_BN, Tri_indx_sub[2]]]
            dist_cart = np.zeros([3,3])
            dist_cart[0] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[1]]
            dist_cart[1] = coords_O_modified_for_four_corners[Tri_indx_sub[0]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            dist_cart[2] = coords_O_modified_for_four_corners[Tri_indx_sub[1]] - coords_O_modified_for_four_corners[Tri_indx_sub[2]]
            #dist_cart_PBC_corrected = perform_PBC(L, dist_cart)
            dist_cart_PBC_corrected = dist_cart
            Natoms_trainagle_calc = 3
            d = np.zeros(Natoms_trainagle_calc)
            count = 0
            for i in np.arange(0,Natoms_trainagle_calc,1):
                d[count] = np.sqrt(np.sum(np.square(dist_cart_PBC_corrected[i])))
                count += 1
            p = np.sum(d)/2
            area[3] = np.sqrt(p*(p-d[0])*(p-d[1])*(p-d[2]))

        area_effective[num_BN] = (area[0] + area[1] + area[2] + area[3]) / 2
        #print(area[0], area[1], area[2], area[3])
            
    return area_effective


# In[ ]:


######## Related to BN calculation

def split_into_chunks(m_range, num_processes):
    chunk_size, remainder = divmod(len(m_range), num_processes)
    m_range = list(m_range)  # Convert the range to a list
    chunks = [m_range[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_processes)]    
    return chunks

def BN_calculator_call(chunk, Num_4O_BNs, indx_4O_BNs, coords_O, L, NOx):
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    temp_coords_O_chunk = coords_O[chunk[0]*NOx:(chunk[-1]+1)*NOx:, :]
    area_effective_all_temp = np.zeros((Nsnapshots3, Num_4O_BNs))
    for ns in range(Nsnapshots3):
        #if ns % 10000 == 0 and ns > 0:
        #    print(ns)
        temp_coords_O = temp_coords_O_chunk[ns*NOx:(ns+1)*NOx, :]
        area_effective_all_temp[ns, :] = BN_calculator(Num_4O_BNs, indx_4O_BNs, temp_coords_O, L)
    return area_effective_all_temp

area_effective_all = np.zeros((Nsnapshots, Num_4O_BNs_all))
m_range = range(Nsnapshots)
chunks = split_into_chunks(m_range, num_processes)

with multiprocessing.Pool(num_processes) as pool:
    results = pool.starmap(BN_calculator_call, [(chunk, Num_4O_BNs_all, indx_4O_BNs_all, coords_O, L, NOx) for chunk in chunks])

area_effective_all = np.concatenate(results, axis=0)

file_name = f"./MD_BN_vs_t.txt"
# Save the array as a text file
np.savetxt(file_name, area_effective_all, fmt="%.3f", delimiter=" ")

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a 2D NumPy array named 'area_effective_all'
# Nsnapshots rows and Num_4O_BNs columns
Num_4O_BNs_all = area_effective_all.shape[1]  # Assuming you have this variable defined

# Get the number of columns (Num_4O_BNs)
num_bns = area_effective_all.shape[1]

mean_values = np.zeros(area_effective_all.shape[1])
std_err_values = np.zeros(area_effective_all.shape[1])

# Iterate over different columns (Bottleneck Areas)
counter = 0
for bn_index in range(num_bns):
    # Extract data for a specific column (Bottleneck Area)
    data_for_bn = area_effective_all[:, bn_index]

    # Calculate the mean and standard error for the column
    mean_value = np.mean(data_for_bn)
    std_err_value = np.std(data_for_bn) # / np.sqrt(data_for_bn.shape[0]) Let's not normalize the st. deviation

    mean_values[counter] = mean_value
    std_err_values[counter] = std_err_value

    counter += 1

print(mean_values.shape)
print(std_err_values.shape)

filename_out = f"./MD_BN.txt"

with open(filename_out, "w") as file:
    for mean, std_err in zip(mean_values, std_err_values):
        file.write(f"{mean} {std_err}\n")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming area_effective_all is defined and contains your data

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
for i in range(area_effective_all.shape[1]):
    x_values = [i] * area_effective_all.shape[0]
    y_values = area_effective_all[:, i]
    if i in BN_pathway_desired[:, 2]:
        plt.scatter(x_values, y_values, alpha=0.5, color=(0, 1, 1))
    else:
        plt.scatter(x_values, y_values, alpha=0.5, color=(0, .8, 1))
    
    # Calculate mean and standard deviation
    mean = np.mean(y_values)
    std_dev = np.std(y_values)
    
    # Plot error bars with larger size and black color
    plt.errorbar(i, mean, yerr=std_dev, fmt='o', color='black', markersize=8)

plt.title('Distribution of bottleneck areas')
plt.xlabel('Bottleneck #')
plt.ylabel('Bottleneck area ($\\AA^2$)')
plt.show()
plt.savefig('Figure_distribution_of_BN_areas.png')


# In[ ]:


# Let's save the needed numbers for each pathway and BN In a vsc file
import pandas as pd

headers = [
    "site_i",
    "site_j",
    "x_site_i",
    "x_site_j",
    "BN#",
    "4O_indx",
    "BN_area_ave",
    "BN_area_std",
    "D_ij",
    "D_ji",
    "D_ave"
]

# Create a Pandas DataFrame with empty data
df = pd.DataFrame(columns=headers)

for i in range(Nsites):
    for j in range(Nsites):
        if j > i:
            if (BNO4_ids_matrix[i, j] != 0).any():
                # First BN
                thisBN = BNO4_ids_matrix[i, j, 0]
                row_data = {
                    "site_i": i,
                    "site_j": j,
                    "x_site_i": x_sites_direct[i],
                    "x_site_j": x_sites_direct[j],
                    "BN#": thisBN,
                    "4O_indx": indx_4O_BNs_all[thisBN]+1,
                    "BN_area_ave": mean_values[thisBN],
                    "BN_area_std": std_err_values[thisBN],
                    "D_ij": Dij[i][j][0],
                    "D_ji": Dij[j][i][0],
                    "D_ave": abs(Dij[i][j][0] + Dij[j][i][0])
                }
                df = df.append(row_data, ignore_index=True)
                # Second BN
                thisBN = BNO4_ids_matrix[i, j, 1]
                row_data = {
                    "site_i": i,
                    "site_j": j,
                    "x_site_i": x_sites_direct[i],
                    "x_site_j": x_sites_direct[j],
                    "BN#": thisBN,
                    "4O_indx": indx_4O_BNs_all[thisBN]+1,
                    "BN_area_ave": mean_values[thisBN],
                    "BN_area_std": std_err_values[thisBN],
                    "D_ij": Dij[i][j][1],
                    "D_ji": Dij[j][i][1],
                    "D_ave": abs(Dij[i][j][1] + Dij[j][i][1])
                }
                df = df.append(row_data, ignore_index=True)

#df.to_csv('D_ij_BN_info.csv', index=False)
df.to_csv("D_ij_BN_info.csv", float_format="%.6e", index=False)


# In[ ]:


# Plot Dij vs BN

import matplotlib.pyplot as plt

# Initialize lists to store data points
BN_values = []
BN_std_values = []
D_values = []
BN_angles1_values = []
BN_angles1_std = []
BN_angles2_values = []
BN_angles2_std = []
BN_angles3_values = []
BN_angles3_std = []
BN_angles4_values = []
BN_angles4_std = []

for i in range(Nsites):
    for j in range(Nsites):
        if j > i:
            if (BNO4_ids_matrix[i, j] != 0).any():
                # First BN
                thisBN = BNO4_ids_matrix[i, j, 0]
                print(thisBN)
                BN_values.append(mean_values[thisBN])
                BN_std_values.append(std_err_values[thisBN])
                D_values.append(abs(Dij[i][j][0] + Dij[j][i][0]))
                BN_angles1_values.append(np.mean(BN_angles_time[:, thisBN, 0]))
                BN_angles1_std.append(np.std(BN_angles_time[:, thisBN, 0]))
                BN_angles2_values.append(np.mean(BN_angles_time[:, thisBN, 1]))
                BN_angles2_std.append(np.std(BN_angles_time[:, thisBN, 1]))
                BN_angles3_values.append(np.mean(BN_angles_time[:, thisBN, 2]))
                BN_angles3_std.append(np.std(BN_angles_time[:, thisBN, 2]))
                BN_angles4_values.append(np.mean(BN_angles_time[:, thisBN, 3]))
                BN_angles4_std.append(np.std(BN_angles_time[:, thisBN, 3]))
                
                # Second BN
                thisBN = BNO4_ids_matrix[i, j, 1]
                BN_values.append(mean_values[thisBN])
                BN_std_values.append(std_err_values[thisBN])
                D_values.append(abs(Dij[i][j][1] + Dij[j][i][1]))
                BN_angles1_values.append(np.mean(BN_angles_time[:, thisBN, 0]))
                BN_angles1_std.append(np.std(BN_angles_time[:, thisBN, 0]))
                BN_angles2_values.append(np.mean(BN_angles_time[:, thisBN, 1]))
                BN_angles2_std.append(np.std(BN_angles_time[:, thisBN, 1]))
                BN_angles3_values.append(np.mean(BN_angles_time[:, thisBN, 2]))
                BN_angles3_std.append(np.std(BN_angles_time[:, thisBN, 2]))
                BN_angles4_values.append(np.mean(BN_angles_time[:, thisBN, 3]))
                BN_angles4_std.append(np.std(BN_angles_time[:, thisBN, 3]))

# Plot the scatter plot
plt.rcParams.update({'font.size': 14})
plt.scatter(BN_values, D_values)
plt.xlabel('Bottleneck area ($\\AA^2$)')
plt.ylabel('Dij ($cm^2/s$)')
#plt.xlim([7.5, 7.9])
#plt.ylim([1e-8, 3.5e-7])
#plt.title('Scatter Plot')
plt.show()
plt.savefig('Figure_Dij_vs_BN.png')

# Combine arrays horizontally
data_np = np.column_stack((D_values, BN_values, BN_std_values))
data_np2 = np.column_stack((D_values, BN_values, BN_std_values, BN_angles1_values, BN_angles2_values, BN_angles3_values, BN_angles4_values, BN_angles1_std, BN_angles2_std, BN_angles3_std, BN_angles4_std))

# Save data to a file
np.savetxt('D_BN_std.txt', data_np, fmt=['%.6e', '%1.5f', '%1.5f'], delimiter='\t', header='D_values\tBN_values\tBN_std_values', comments='')
np.savetxt('D_BN_std_angles_std.txt', data_np2, fmt=['%.6e', '%1.5f', '%1.5f', '%1.5f', '%1.5f', '%1.5f', '%1.5f', '%1.5f', '%1.5f', '%1.5f', '%1.5f'], delimiter='\t', header='D_values\tBN_values\tBN_std_values\tBN_angles1_values\tBN_angles2_values\tBN_angles3_values\tBN_angles4_values\tBN_angles1_std\tBN_angles2_std\tBN_angles3_std\tBN_angles4_std', comments='')


# In[ ]:


print(BN_angles1_values)
print(BN_angles1_std)
print(D_values)


# In[ ]:


# Plot Dij vs BN

import matplotlib.pyplot as plt

# Initialize lists to store data points
BN_values = []
D_values = []

for i in range(Nsites):
    for j in range(Nsites):
        if j > i:
            if (BNO4_ids_matrix[i, j] != 0).any():
                # First BN
                thisBN = BNO4_ids_matrix[i, j, 0]
                BN_values.append(std_err_values[thisBN])
                D_values.append(abs(Dij[i][j][0] + Dij[j][i][0]))
                
                # Second BN
                thisBN = BNO4_ids_matrix[i, j, 1]
                BN_values.append(std_err_values[thisBN])
                D_values.append(abs(Dij[i][j][1] + Dij[j][i][1]))

# Plot the scatter plot
plt.rcParams.update({'font.size': 14})
plt.scatter(BN_values, D_values)
plt.xlabel('BN std error ($\\AA^2$)')
plt.ylabel('Dij ($cm^2/s$)')
#plt.xlim([7.5, 7.9])
#plt.ylim([1e-8, 3.5e-7])
#plt.title('Scatter Plot')
plt.show()
plt.savefig('Figure_Dij_vs_BN_std_error.png')

