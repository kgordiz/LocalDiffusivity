# main.py

# One test change here (2)

from file_io import readtrajfile
from data_processing import (
    get_atom_ids_by_type, get_coordinates_for_atoms,
    downsample_coordinates, calculate_coordination_number,
    get_unwrapped_coordinates, reshape_coordinates_for_compatibility,
    calculate_Tij_parallel, create_time_array, save_variables_to_npz
)

def main(traj_filename="./Example_polymer/traj.lammpstrj", Nsites=10, dt=2, nsample=1000, Neverythis=1, d0=3.18, nCN=6, mCN=12, num_processes=4):
    Natoms, L, Ntimestep, coords_xyz, coords_id_type = readtrajfile(traj_filename)
    group1_type = [90] # type of Li atoms
    groupO1_type = [6] # type of O in PEO
    groupO2_type = [94] # type of O in TFSI
    group1_id = get_atom_ids_by_type(coords_id_type, Natoms, group1_type)
    groupO1_id = get_atom_ids_by_type(coords_id_type, Natoms, groupO1_type)
    groupO2_id = get_atom_ids_by_type(coords_id_type, Natoms, groupO2_type)

    xLi = get_coordinates_for_atoms(coords_xyz, group1_id, Ntimestep)
    xLi_ds = downsample_coordinates(xLi, len(group1_id), Ntimestep, Neverythis)
    xO1 = get_coordinates_for_atoms(coords_xyz, groupO1_id, Ntimestep)
    xO1_ds = downsample_coordinates(xO1, len(groupO1_id), Ntimestep, Neverythis)
    xO2 = get_coordinates_for_atoms(coords_xyz, groupO2_id, Ntimestep)
    xO2_ds = downsample_coordinates(xO2, len(groupO2_id), Ntimestep, Neverythis)

    r_real = get_unwrapped_coordinates(xLi_ds, L, Nsnaps=len(xLi_ds) // len(group1_id), NLi=len(group1_id))
    r_real_2d = reshape_coordinates_for_compatibility(r_real)
    
    envLiO1 = calculate_coordination_number(xLi_ds, xO1_ds, len(group1_id), len(groupO1_id), L, d0, nCN, mCN, num_processes)
    envLiO2 = calculate_coordination_number(xLi_ds, xO2_ds, len(group1_id), len(groupO2_id), L, d0, nCN, mCN, num_processes)

    calculate_Tij_parallel(r_real_2d, len(group1_id), Ntimestep, envLiO1, Nsites, num_processes)
    
    time = create_time_array(Nsnaps=Ntimestep, nsample=nsample, LAMMPS_timestep=dt)
    save_variables_to_npz("multiple_arrays_from_mbit_calcs.npz", envLiO1, envLiO2, time)
    
    print("Completed. Variables saved.")
    return

if __name__ == "__main__":
    main()

