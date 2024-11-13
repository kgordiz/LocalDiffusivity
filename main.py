# main.py

# One test change here

from file_io import readtrajfile
from data_processing import (
    get_atom_ids_by_type, get_coordinates_for_atoms,
    downsample_coordinates, calculate_coordination_number,
    get_unwrapped_coordinates, reshape_coordinates_for_compatibility,
    calculate_Tij_parallel, create_time_array, save_variables_to_npz
)

def main(traj_filename="./traj.lammpstrj", Nsites=10, dt=2, nsample=1000, Neverythis=1, d0=3.18, nCN=6, mCN=12, num_processes=4):
    Natoms, L, Ntimestep, coords_xyz, coords_id_type = readtrajfile(traj_filename)
    group1_type = [90]  # Li atoms
    group1_id = get_atom_ids_by_type(coords_id_type, Natoms, group1_type)
    
    xLi = get_coordinates_for_atoms(coords_xyz, group1_id, Ntimestep)
    xLi_ds = downsample_coordinates(xLi, len(group1_id), Ntimestep, Neverythis)
    r_real = get_unwrapped_coordinates(xLi_ds, L, Nsnaps=len(xLi_ds) // len(group1_id), NLi=len(group1_id))
    r_real_2d = reshape_coordinates_for_compatibility(r_real)
    
    envLiLa = calculate_coordination_number(xLi_ds, xLi_ds, len(group1_id), len(group1_id), L, d0, nCN, mCN, num_processes)
    envLiTFSI = calculate_coordination_number(xLi_ds, xLi_ds, len(group1_id), len(group1_id), L, d0, nCN, mCN, num_processes)

    dt_values = [5, 50] + list(range(100, Ntimestep, 100))
    calculate_Tij_parallel(r_real_2d, len(group1_id), Ntimestep, dt_values, envLiLa, Nsites, num_processes)
    
    time = create_time_array(Nsnaps=Ntimestep, nsample=nsample, LAMMPS_timestep=dt)
    save_variables_to_npz("multiple_arrays_from_mbit_calcs.npz", envLiLa, envLiTFSI, time)
    
    print("Completed. Variables saved.")
    return

if __name__ == "__main__":
    main()

