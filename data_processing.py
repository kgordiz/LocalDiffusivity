# data_processing.py

import numpy as np
from multiprocessing import Pool

# Data Processing and Utility Functions
def perform_PBC(L, dx_direct):
    """
    Applies periodic boundary correction to atomic displacements.
    
    Parameters:
    - L (numpy.ndarray): 3x3 array representing the lattice vectors of the simulation box.
    - dx_direct (numpy.ndarray): Nx3 array of atomic displacements in direct coordinates.
    
    Returns:
    - dx_cartesian (numpy.ndarray): Nx3 array of atomic displacements with PBC applied.
    """
    
    # Lattice vector magnitudes
    L0 = np.sqrt(np.sum(np.power(L[0, :], 2)))
    L1 = np.sqrt(np.sum(np.power(L[1, :], 2)))
    L2 = np.sqrt(np.sum(np.power(L[2, :], 2)))
    
    # Lattice vector unit directions
    L0hat = L[0, :] / L0
    L1hat = L[1, :] / L1
    L2hat = L[2, :] / L2
    
    # Initialize Cartesian displacement as direct
    dx_cartesian = dx_direct.copy()
    Natoms = dx_cartesian.shape[0]
    
    for i in range(Natoms):
        # Correct along each lattice vector
        d0 = np.dot(dx_cartesian[i, :], L0hat)
        if d0 >= L0 / 2:
            dx_cartesian[i, :] -= L[0, :]
        elif d0 < -L0 / 2:
            dx_cartesian[i, :] += L[0, :]
        
        d1 = np.dot(dx_cartesian[i, :], L1hat)
        if d1 >= L1 / 2:
            dx_cartesian[i, :] -= L[1, :]
        elif d1 < -L1 / 2:
            dx_cartesian[i, :] += L[1, :]
        
        d2 = np.dot(dx_cartesian[i, :], L2hat)
        if d2 >= L2 / 2:
            dx_cartesian[i, :] -= L[2, :]
        elif d2 < -L2 / 2:
            dx_cartesian[i, :] += L[2, :]
    
    return dx_cartesian

def get_atom_ids_by_type(coords_id_type, Natoms, atom_types):
    """
    Identifies atom IDs based on specified atom types from the trajectory file.

    Parameters:
    - coords_id_type (numpy.ndarray): Array of atom IDs and types, shape (Ntimestep, Natoms, 2).
    - Natoms (int): Number of atoms.
    - atom_types (list): List of atom types to identify.
    
    Returns:
    - atom_ids (list): List of atom IDs corresponding to the specified types.
    """
    
    atom_ids = []
    for i in range(Natoms):
        if coords_id_type[0, i, 1] in atom_types:
            atom_ids.append(coords_id_type[0, i, 0] - 1)  # Adjusting for 0-indexing
    
    return atom_ids

def get_coordinates_for_atoms(coords_xyz, atom_ids, Ntimestep):
    """
    Extracts coordinates for specific atoms across all timesteps.

    Parameters:
    - coords_xyz (numpy.ndarray): Array of atomic coordinates, shape (Ntimestep, Natoms, 3).
    - atom_ids (list): List of atom IDs to retrieve coordinates for.
    - Ntimestep (int): Number of timesteps in the simulation.
    
    Returns:
    - atom_coords (numpy.ndarray): Coordinates for specified atoms, shape (Ntimestep * len(atom_ids), 3).
    """
    
    num_atoms = len(atom_ids)
    atom_coords = np.zeros((Ntimestep * num_atoms, 3))
    
    for nt in range(Ntimestep):
        for i, atom_id in enumerate(atom_ids):
            atom_coords[nt * num_atoms + i, :] = coords_xyz[nt, atom_id, :]
    
    return atom_coords

def downsample_coordinates(atom_coords, num_atoms, Ntimestep, Neverythis=1):
    """
    Downsamples coordinate data for a group of atoms by selecting snapshots at regular intervals.
    
    Parameters:
    - atom_coords (numpy.ndarray): Coordinates of shape (Ntimestep * num_atoms, 3).
    - num_atoms (int): Number of atoms in the group.
    - Ntimestep (int): Total number of timesteps in the simulation.
    - Neverythis (int): Downsampling frequency; selects every Nth snapshot.
    
    Returns:
    - atom_coords_ds (numpy.ndarray): Downsampled coordinates.
    """
    
    # Define down-selected snapshots
    ds_snapshots = np.arange(0, Ntimestep, Neverythis)
    len_ds_snapshots = len(ds_snapshots)
    
    # Initialize downsampled coordinates array
    atom_coords_ds = np.zeros((len_ds_snapshots * num_atoms, 3))
    
    # Populate downsampled array
    for counter, t in enumerate(ds_snapshots):
        atom_coords_ds[counter * num_atoms : (counter + 1) * num_atoms, :] = atom_coords[t * num_atoms : (t + 1) * num_atoms, :]
    
    return atom_coords_ds

def get_unwrapped_coordinates(xLi_ds, L, Nsnaps, NLi):
    """
    Unwraps the coordinates of Li ions by canceling periodic boundary effects, necessary for MSD calculations.
    
    Parameters:
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms in Cartesian coordinates.
    - L (numpy.ndarray): Simulation box dimensions, 3x3 array.
    - Nsnaps (int): Number of snapshots.
    - NLi (int): Number of Li atoms.
    
    Returns:
    - r_real (numpy.ndarray): Unwrapped coordinates, shape (Nsnaps, NLi, 3).
    """
    
    r_cartesian = np.zeros((Nsnaps, NLi, 3))
    r_real = np.zeros((Nsnaps, NLi, 3))
    
    # Box dimensions for each direction
    x_size, y_size, z_size = L[0, 0], L[1, 1], L[2, 2]
    
    # Load initial Cartesian positions
    for nt in range(Nsnaps):
        r_cartesian[nt] = xLi_ds[0 + nt * NLi:NLi + nt * NLi]
    
    # Unwrap coordinates by removing PBC effects
    for m in range(Nsnaps):
        if m == 0:  # Set initial positions without adjustment
            r_real[m] = r_cartesian[m]
        else:
            for n in range(NLi):
                dx = r_cartesian[m, n, 0] - r_cartesian[m - 1, n, 0]
                dy = r_cartesian[m, n, 1] - r_cartesian[m - 1, n, 1]
                dz = r_cartesian[m, n, 2] - r_cartesian[m - 1, n, 2]

                # Apply periodic boundary corrections for each direction
                if dx > x_size * 0.5:
                    dx -= x_size
                elif dx <= -x_size * 0.5:
                    dx += x_size
                if dy > y_size * 0.5:
                    dy -= y_size
                elif dy <= -y_size * 0.5:
                    dy += y_size
                if dz > z_size * 0.5:
                    dz -= z_size
                elif dz <= -z_size * 0.5:
                    dz += z_size

                # Accumulate the corrected displacement
                r_real[m, n, 0] = r_real[m - 1, n, 0] + dx
                r_real[m, n, 1] = r_real[m - 1, n, 1] + dy
                r_real[m, n, 2] = r_real[m - 1, n, 2] + dz
    
    return r_real

def reshape_coordinates_for_compatibility(r_real_3d):
    """
    Reshapes 3D unwrapped coordinates array from (Ntimesteps, Natoms, 3) to (Ntimesteps * Natoms, 3).
    Useful for compatibility with older code that expects a flattened 2D format.
    
    Parameters:
    - r_real_3d (numpy.ndarray): 3D array of unwrapped coordinates, shape (Ntimesteps, Natoms, 3).
    
    Returns:
    - r_real_2d (numpy.ndarray): 2D reshaped array of coordinates, shape (Ntimesteps * Natoms, 3).
    """
    Ntimesteps, Natoms, _ = r_real_3d.shape
    r_real_2d = np.zeros((Ntimesteps * Natoms, 3))
    
    for m in range(Ntimesteps):
        for n in range(Natoms):
            indx = m * Natoms + n
            r_real_2d[indx] = r_real_3d[m, n, :]
    
    return r_real_2d

# Coordination Number Calculation Functions

def split_into_chunks(m_range, num_processes):
    """
    Splits a range of indices into chunks for parallel processing.
    
    Parameters:
    - m_range (range): Range of indices to split.
    - num_processes (int): Number of processes to parallelize over.
    
    Returns:
    - chunks (list of lists): Each inner list represents a chunk of indices for one process.
    """
    chunk_size, remainder = divmod(len(m_range), num_processes)
    m_range = list(m_range)
    chunks = [m_range[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_processes)]
    return chunks


def CN_calc_call(chunk, xLi_ds, xOther_ds, NLi, NOther, L, d0, nCN, mCN):
    """
    Calculates the coordination number (CN) for a chunk of snapshots between Li and a specified atom group (e.g., O in PEO or TFSI).
    
    Parameters:
    - chunk (list): List of snapshot indices to process.
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms.
    - xOther_ds (numpy.ndarray): Downsampled coordinates for the other atom group.
    - NLi (int): Number of Li atoms.
    - NOther (int): Number of atoms in the other group.
    - L (numpy.ndarray): Simulation box dimensions.
    - d0 (float): Cutoff distance for CN calculation.
    - nCN (int): Exponent in CN calculation formula.
    - mCN (int): Exponent in CN calculation formula.
    
    Returns:
    - envLiOther (numpy.ndarray): Coordination numbers for Li atoms over the snapshots in the chunk.
    """
    dx = np.zeros((1, 3))
    Nsnapshots3 = len(chunk)
    envLiOther = np.zeros((Nsnapshots3, NLi))
    
    xLi_ds_chunk = xLi_ds[chunk[0] * NLi:(chunk[-1] + 1) * NLi, :]
    xOther_ds_chunk = xOther_ds[chunk[0] * NOther:(chunk[-1] + 1) * NOther, :]
    
    for nt in range(Nsnapshots3):
        for itemi in range(NLi):
            for itemj in range(NOther):
                dx[0, :] = xLi_ds_chunk[nt * NLi + itemi, :] - xOther_ds_chunk[nt * NOther + itemj, :]
                dx_PBC_corrected = perform_PBC(L, dx)
                dx_abs = np.sqrt(np.sum(np.power(dx_PBC_corrected, 2)))
                temp = (1 - np.power(dx_abs / d0, nCN)) / (1 - np.power(dx_abs / d0, mCN))
                envLiOther[nt, itemi] += temp
    
    return envLiOther

def calculate_coordination_number(xLi_ds, xOther_ds, NLi, NOther, L, d0=3.18, nCN=6, mCN=12, num_processes=4):
    """
    Manages parallel processing for CN calculation across all snapshots for a specified pair of atom groups.
    
    Parameters:
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms.
    - xOther_ds (numpy.ndarray): Downsampled coordinates for the other atom group.
    - NLi (int): Number of Li atoms.
    - NOther (int): Number of atoms in the other group.
    - L (numpy.ndarray): Simulation box dimensions.
    - d0 (float): Cutoff distance for CN calculation (default=3.18).
    - nCN (int): Exponent in CN calculation formula (default=6).
    - mCN (int): Exponent in CN calculation formula (default=12).
    - num_processes (int): Number of processes for parallelization (default=4).
    
    Returns:
    - envLiOther (numpy.ndarray): Coordination numbers for Li atoms across all snapshots for the specified atom group.
    """
    Nsnaps = xLi_ds.shape[0] // NLi
    m_range = range(Nsnaps)
    chunks = split_into_chunks(m_range, num_processes)
    
    with Pool(num_processes) as pool:
        results = pool.starmap(CN_calc_call, [(chunk, xLi_ds, xOther_ds, NLi, NOther, L, d0, nCN, mCN) for chunk in chunks])
    
    return np.concatenate(results, axis=0)

import numpy as np
from multiprocessing import Pool

def mindx_calc(m_range, r_real, Natoms, Ntimesteps, dt, envLiLa, envj_length):
    """
    Calculates cross-cuts of MSD, updating the Tij matrix for specified time steps.
    
    Parameters:
    - m_range (list): Range of indices for time steps to process.
    - r_real (numpy.ndarray): Unwrapped coordinates, shape (Ntimesteps * Natoms, 3).
    - Natoms (int): Number of atoms.
    - Ntimesteps (int): Total number of time steps.
    - dt (int): Time difference between current and next time step.
    - envLiLa (numpy.ndarray): Environment data matrix for each Li atom.
    - envj_length (int): Length of environment list (number of local environments).
    
    Returns:
    - Tij (numpy.ndarray): Updated Tij matrix, shape (envj_length, envj_length).
    """
    Tij = np.zeros((envj_length, envj_length))
    
    for m in m_range:
        for n in range(Natoms):
            indx = m * Natoms + n
            indx_dt = (m + dt) * Natoms + n
            dr_tau = r_real[indx_dt] - r_real[indx]

            for mbit in range(dt):
                indx1 = (m + mbit) * Natoms + n
                indx2 = (m + mbit + 1) * Natoms + n
                bitx = (r_real[indx2][0] - r_real[indx1][0]) * dr_tau[0]
                bity = (r_real[indx2][1] - r_real[indx1][1]) * dr_tau[1]
                bitz = (r_real[indx2][2] - r_real[indx1][2]) * dr_tau[2]

                bitD = (bitx + bity + bitz) / (2.0 * Natoms * 3 * (Ntimesteps - dt))
                
                idx_i_env = int(np.round(envLiLa[m, n]))
                idx_j_env = int(np.round(envLiLa[m + dt, n]))
                Tij[idx_i_env][idx_j_env] += bitD
    
    return Tij

def calculate_Tij_parallel(r_real, Natoms, Ntimesteps, dt_values, envLiLa, envj_length, num_processes=4):
    """
    Manages parallel processing for Tij calculation across specified time steps.
    
    Parameters:
    - r_real (numpy.ndarray): Unwrapped coordinates, shape (Ntimesteps * Natoms, 3).
    - Natoms (int): Number of atoms.
    - Ntimesteps (int): Total number of time steps.
    - dt_values (list of int): List of time steps (dt) to calculate Tij matrices for.
    - envLiLa (numpy.ndarray): Environment data matrix for each Li atom.
    - envj_length (int): Length of environment list (number of local environments).
    - num_processes (int): Number of processes for parallelization.
    
    Writes:
    - Each calculated Tij matrix is saved as a text file named "Tij_<dt>.txt".
    """
    for dt in dt_values:
        print(f"Calculating Tij for dt={dt}")

        # Define the range for parallel processing and split into chunks
        m_range = range(Ntimesteps - dt)
        chunks = split_into_chunks(m_range, num_processes)

        with Pool(num_processes) as pool:
            results = pool.starmap(mindx_calc, [(chunk, r_real, Natoms, Ntimesteps, dt, envLiLa, envj_length) for chunk in chunks])

        # Sum results from all processes to get the final Tij matrix
        Tij = sum(results)

        # Save Tij matrix to a file
        with open(f"Tij_{dt}.txt", "w") as writefile_Tij:
            for row in Tij:
                writefile_Tij.write(" ".join(map(str, row)) + "\n")

# Time and Save Functions
def create_time_array(Nsnaps, nsample, LAMMPS_timestep):
    """
    Creates a time array for the specified number of snapshots with conversion to picoseconds.
    
    Parameters:
    - Nsnaps (int): Number of snapshots.
    - nsample (int): Sampling frequency in timesteps.
    - LAMMPS_timestep (float): Timestep in femtoseconds.
    
    Returns:
    - time (numpy.ndarray): Array of time values in picoseconds.
    """
    time = np.zeros(Nsnaps - 1)
    for t in range(Nsnaps - 1):
        time[t] = t * nsample * LAMMPS_timestep / 1000  # Conversion from fs to ps
    return time

def save_variables_to_npz(filename, envLiLa, envLiTFSI, time):
    """
    Saves the given arrays to an .npz file for post-processing.
    
    Parameters:
    - filename (str): Path to save the .npz file.
    - envLiLa (numpy.ndarray): Environment matrix for Li-PEO coordination.
    - envLiTFSI (numpy.ndarray): Environment matrix for Li-TFSI coordination.
    - time (numpy.ndarray): Time array in picoseconds.
    """
    print("envLiLa shape:", envLiLa.shape)
    print("envLiTFSI shape:", envLiTFSI.shape)
    print("time shape:", time.shape)
    
    np.savez(filename, envLiLa=envLiLa, envLiTFSI=envLiTFSI, time=time)
