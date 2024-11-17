import numpy as np
from data_processing import (
    readtrajfile, get_atom_ids_by_type, get_coordinates_for_groups, downsample_coordinates,
    calculate_CN, calculate_CN_TFSI, calculate_unwrapped_coordinates, calculate_msd_for_each_atom,
    calculate_msd_between_environments, print_and_save_results, write_top_what_results
)

# Initial setup and reading trajectory
traj_filename = "./Example_polymer/traj.lammpstrj"
Natoms, L, Ntimestep, coords_xyz, coords_id_type = readtrajfile(traj_filename)
print(f"Number of atoms: {Natoms}, Number of timesteps: {Ntimestep}")

# Define atom types for groups
group1_type = [90]
group2_type = [6]
group3_type = [94]

# Get atom IDs based on specified types
group1_id, group2_id, group3_id = get_atom_ids_by_type(coords_id_type, Natoms, group1_type, group2_type, group3_type)

# Simulation parameters
Nsites = 10
dt = 2
nsample = 1000
NLi = len(group1_id)
NLa = len(group2_id)
NTFSI = len(group3_id)

# Extract coordinates
xLi, xLa, xTFSI = get_coordinates_for_groups(coords_xyz, group1_id, group2_id, group3_id, Ntimestep)

# Downsample coordinates
Neverythis = 1
xLi_ds, xLa_ds, xTFSI_ds, Nsnaps = downsample_coordinates(xLi, xLa, xTFSI, NLi, NLa, NTFSI, Ntimestep, Neverythis)

# Calculate CN
d0, nCN, mCN, topWhat = 3.18, 6, 12, 2
num_processes = 4
envLiLa = calculate_CN(xLi_ds, xLa_ds, NLi, NLa, L, d0, nCN, mCN, num_processes)
envLiTFSI = calculate_CN_TFSI(xLi_ds, xTFSI_ds, NLi, NTFSI, L, d0, nCN, mCN, num_processes)

# Define file paths for environment data
filename_values = "./site_info_continuous_spectrum_topwhat_values.txt"
filename_indices = "./site_info_continuous_spectrum_topwhat_indices.txt"

# Save top "what" results for coordination numbers
write_top_what_results(envLiLa, Nsnaps, NLi, topWhat, filename_values, filename_indices)
#write_top_what_results(envLiTFSI, Nsnaps, NLi, topWhat, filename_prefix="site_info_continuous_spectrum_LiTFSI")

# Unwrap coordinates
r_real = calculate_unwrapped_coordinates(xLi_ds, NLi, Ntimestep, L)
r_real_reformatted = r_real.reshape(NLi * Ntimestep, 3)

# Load environment data
Env_cont_val = np.loadtxt(filename_values)
Env_cont_indx = np.loadtxt(filename_indices, dtype=int)

# Calculate MSDs
msd_env = calculate_msd_between_environments(r_real_reformatted, r_real, Env_cont_val, Env_cont_indx, Nsites, Ntimestep, NLi, topWhat, num_processes)
msd_atom_ind = calculate_msd_for_each_atom(r_real, NLi)

# Save results
print_and_save_results(envLiLa, envLiTFSI, msd_env, msd_atom_ind, filename='multiple_arrays.npz')

