import numpy as np
import multiprocessing
from multiprocessing import Pool
from fft_utils import msd_fft_cross

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
    dx_cartesian = dx_direct
    Natoms = dx_cartesian.shape[0]
    
    for i in range(Natoms):
        # Correct along the 1st vector
        d0 = np.dot(dx_cartesian[i, :], L0hat)
        if d0 >= L0 / 2:
            dx_cartesian[i, :] -= L[0, :]
        elif d0 < -L0 / 2:
            dx_cartesian[i, :] += L[0, :]
        
        # Correct along the 2nd vector
        d1 = np.dot(dx_cartesian[i, :], L1hat)
        if d1 >= L1 / 2:
            dx_cartesian[i, :] -= L[1, :]
        elif d1 < -L1 / 2:
            dx_cartesian[i, :] += L[1, :]
        
        # Correct along the 3rd vector
        d2 = np.dot(dx_cartesian[i, :], L2hat)
        if d2 >= L2 / 2:
            dx_cartesian[i, :] -= L[2, :]
        elif d2 < -L2 / 2:
            dx_cartesian[i, :] += L[2, :]
    
    return dx_cartesian

def readtrajfile(filename):
    """
    Reads atomic positions and types from a LAMMPS trajectory file.
    
    Parameters:
    - filename (str): Path to the trajectory file.
    
    Returns:
    - Natoms (int): Number of atoms.
    - L (numpy.ndarray): 3x3 array representing the lattice dimensions.
    - Ntimestep (int): Number of timesteps in the file.
    - coords_xyz (numpy.ndarray): Trajectory coordinates of shape (Ntimestep, Natoms, 3) for x, y, z.
    - coords_id_type (numpy.ndarray): Array of atom IDs and types of shape (Ntimestep, Natoms, 2).
    """
    Ntimestep = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        
        # Count timesteps based on 'TIMESTEP' markers
        for line in lines:
            words = line.split()
            if len(words) > 0 and words[-1] == 'TIMESTEP':
                Ntimestep += 1
        
        Natoms = int(lines[3].split()[0])
        L = np.zeros((3, 3))
        L[0, 0] = float(lines[5].split()[1])
        L[1, 1] = float(lines[6].split()[1])
        L[2, 2] = float(lines[7].split()[1])
        
        # Adjust for incomplete file
        Ntimestep -= 1
        coords_xyz = np.zeros((Ntimestep, Natoms, 3))  # 3 for x, y, z
        coords_id_type = np.zeros((Ntimestep, Natoms, 2), dtype=int)  # 2 for atom ID and type
        
        linecounter = 0
        timestepcounter = 0
        while True:
            words = lines[linecounter].split()
            linecounter += 1
            if len(words) > 0 and words[-1] == 'iz':
                for n in range(Natoms):
                    words = lines[linecounter].split()
                    coords_xyz[timestepcounter, n, :] = [float(words[5]), float(words[6]), float(words[7])]
                    coords_id_type[timestepcounter, n, :] = [int(words[0]), int(words[2])]
                    linecounter += 1
                timestepcounter += 1
                if timestepcounter == Ntimestep:
                    break
                    
    return Natoms, L, Ntimestep, coords_xyz, coords_id_type

def get_atom_ids_by_type(coords_id_type, Natoms, group1_type, group2_type, group3_type):
    """
    Identifies atom IDs for specified atom types from the trajectory file.

    Parameters:
    - coords_id_type (numpy.ndarray): Array of atom IDs and types, shape (Ntimestep, Natoms, 2).
    - Natoms (int): Number of atoms.
    - group1_type (list): Atom types for group 1 (e.g., Li atoms).
    - group2_type (list): Atom types for group 2 (e.g., O in PEO).
    - group3_type (list): Atom types for group 3 (e.g., O in TFSI).
    
    Returns:
    - group1_id (list): List of atom IDs for group 1.
    - group2_id (list): List of atom IDs for group 2.
    - group3_id (list): List of atom IDs for group 3.
    """
    group1_id = []
    group2_id = []
    group3_id = []

    # Identify atom IDs based on specified types
    for i in range(Natoms):
        if coords_id_type[0, i, 1] in group1_type:
            group1_id.append(coords_id_type[0, i, 0] - 1)  # Adjusting for 0-indexing
        elif coords_id_type[0, i, 1] in group2_type:
            group2_id.append(coords_id_type[0, i, 0] - 1)
        elif coords_id_type[0, i, 1] in group3_type:
            group3_id.append(coords_id_type[0, i, 0] - 1)
    
    return group1_id, group2_id, group3_id

def get_coordinates_for_groups(coords_xyz, group1_id, group2_id, group3_id, Ntimestep):
    """
    Extracts coordinates for specified groups across all timesteps.

    Parameters:
    - coords_xyz (numpy.ndarray): Array of atomic coordinates, shape (Ntimestep, Natoms, 3).
    - group1_id (list): List of atom IDs for group 1 (e.g., Li atoms).
    - group2_id (list): List of atom IDs for group 2 (e.g., O in PEO).
    - group3_id (list): List of atom IDs for group 3 (e.g., O in TFSI).
    - Ntimestep (int): Number of timesteps in the simulation.

    Returns:
    - xLi (numpy.ndarray): Coordinates of group 1 atoms over timesteps.
    - xLa (numpy.ndarray): Coordinates of group 2 atoms over timesteps.
    - xTFSI (numpy.ndarray): Coordinates of group 3 atoms over timesteps.
    """
    NLi = len(group1_id)
    NLa = len(group2_id)
    NTFSI = len(group3_id)
    xLi = np.zeros((Ntimestep * NLi, 3))
    xLa = np.zeros((Ntimestep * NLa, 3))
    xTFSI = np.zeros((Ntimestep * NTFSI, 3))

    for nt in range(Ntimestep):
        for i in range(NLi):
            xLi[nt * NLi + i, :] = coords_xyz[nt, group1_id[i], :]
        for j in range(NLa):
            xLa[nt * NLa + j, :] = coords_xyz[nt, group2_id[j], :]
        for k in range(NTFSI):
            xTFSI[nt * NTFSI + k, :] = coords_xyz[nt, group3_id[k], :]

    return xLi, xLa, xTFSI

def downsample_coordinates(xLi, xLa, xTFSI, NLi, NLa, NTFSI, Ntimestep, Neverythis=1):
    """
    Downsamples coordinate data for each group by selecting snapshots at regular intervals.
    
    Parameters:
    - xLi (numpy.ndarray): Coordinates of group 1 atoms over all timesteps.
    - xLa (numpy.ndarray): Coordinates of group 2 atoms over all timesteps.
    - xTFSI (numpy.ndarray): Coordinates of group 3 atoms over all timesteps.
    - NLi (int): Number of atoms in group 1.
    - NLa (int): Number of atoms in group 2.
    - NTFSI (int): Number of atoms in group 3.
    - Ntimestep (int): Total number of timesteps in the simulation.
    - Neverythis (int): Downsampling frequency; selects every Nth snapshot.
    
    Returns:
    - xLi_ds (numpy.ndarray): Downsampled coordinates for group 1 atoms.
    - xLa_ds (numpy.ndarray): Downsampled coordinates for group 2 atoms.
    - xTFSI_ds (numpy.ndarray): Downsampled coordinates for group 3 atoms.
    """
    # Define down-selected snapshots
    ds_snapshots = np.arange(0, Ntimestep, Neverythis)
    len_ds_snapshots = len(ds_snapshots)
    
    # Initialize downsampled coordinates arrays
    xLi_ds = np.zeros((len_ds_snapshots * NLi, 3))
    xLa_ds = np.zeros((len_ds_snapshots * NLa, 3))
    xTFSI_ds = np.zeros((len_ds_snapshots * NTFSI, 3))
    
    # Populate downsampled arrays
    counter = 0
    for t in ds_snapshots:
        xLi_ds[counter * NLi:(counter + 1) * NLi, :] = xLi[t * NLi:(t + 1) * NLi, :]
        xLa_ds[counter * NLa:(counter + 1) * NLa, :] = xLa[t * NLa:(t + 1) * NLa, :]
        xTFSI_ds[counter * NTFSI:(counter + 1) * NTFSI, :] = xTFSI[t * NTFSI:(t + 1) * NTFSI, :]
        counter += 1

    return xLi_ds, xLa_ds, xTFSI_ds, counter

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
    m_range = list(m_range)  # Convert the range to a list
    chunks = [m_range[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_processes)]    
    return chunks

def CN_calc_call(chunk, xLi_ds, xLa_ds, NLi, NLa, L, d0, nCN, mCN):
    """
    Calculates the coordination number (CN) for a chunk of snapshots between Li and a specified atom group (e.g., O in PEO).
    
    Parameters:
    - chunk (list): List of snapshot indices to process.
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms.
    - xLa_ds (numpy.ndarray): Downsampled coordinates for the other atom group.
    - NLi (int): Number of Li atoms.
    - NLa (int): Number of atoms in the other group.
    - L (numpy.ndarray): Simulation box dimensions.
    - d0 (float): Cutoff distance for CN calculation.
    - nCN (int): Exponent in CN calculation formula.
    - mCN (int): Exponent in CN calculation formula.
    
    Returns:
    - envLiLa (numpy.ndarray): Coordination numbers for Li atoms over the snapshots in the chunk.
    """
    dx = np.zeros((1,3))
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    envLiLa = np.zeros((Nsnapshots3, NLi))
    
    xLi_ds_chunk = xLi_ds[chunk[0] * NLi:(chunk[-1] + 1) * NLi, :]
    xLa_ds_chunk = xLa_ds[chunk[0] * NLa:(chunk[-1] + 1) * NLa, :]
    
    for nt in range(Nsnapshots3):
        for itemi in range(NLi):
            for itemj in range(NLa):
                dx[0, :] = xLi_ds_chunk[nt * NLi + itemi, :] - xLa_ds_chunk[nt * NLa + itemj, :]
                dx_PBC_corrected = perform_PBC(L, dx)
                dx_abs = np.sqrt(np.sum(np.power(dx_PBC_corrected, 2)))
                temp = (1 - np.power(dx_abs / d0, nCN)) / (1 - np.power(dx_abs / d0, mCN))
                envLiLa[nt, itemi] += temp
    return envLiLa

def calculate_CN(xLi_ds, xLa_ds, NLi, NLa, L, d0, nCN, mCN, num_processes):
    """
    Manages parallel processing for CN calculation across all snapshots for a specified pair of atom groups.
    
    Parameters:
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms.
    - xLa_ds (numpy.ndarray): Downsampled coordinates for the other atom group.
    - NLi (int): Number of Li atoms.
    - NLa (int): Number of atoms in the other group.
    - L (numpy.ndarray): Simulation box dimensions.
    - d0 (float): Cutoff distance for CN calculation.
    - nCN (int): Exponent in CN calculation formula.
    - mCN (int): Exponent in CN calculation formula.
    - num_processes (int): Number of processes for parallelization.
    
    Returns:
    - envLiLa (numpy.ndarray): Coordination numbers for Li atoms across all snapshots for the specified atom group.
    """
    Nsnaps = xLi_ds.shape[0] // NLi
    m_range = range(Nsnaps)
    chunks = split_into_chunks(m_range, num_processes)
    
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(CN_calc_call, [(chunk, xLi_ds, xLa_ds, NLi, NLa, L, d0, nCN, mCN) for chunk in chunks])
    
    return np.concatenate(results, axis=0)


def CN_calc_TFSI_call(chunk, xLi_ds, xTFSI_ds, NLi, NTFSI, L, d0, nCN, mCN):
    """
    Calculates the coordination number (CN) for a chunk of snapshots between Li and TFSI atoms.
    
    Parameters:
    - chunk (list): List of snapshot indices to process.
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms.
    - xTFSI_ds (numpy.ndarray): Downsampled coordinates for TFSI atoms.
    - NLi (int): Number of Li atoms.
    - NTFSI (int): Number of TFSI atoms.
    - L (numpy.ndarray): Simulation box dimensions.
    - d0 (float): Cutoff distance for CN calculation.
    - nCN (int): Exponent in CN calculation formula.
    - mCN (int): Exponent in CN calculation formula.
    
    Returns:
    - envLiTFSI (numpy.ndarray): Coordination numbers for Li atoms with TFSI atoms over the snapshots in the chunk.
    """
    dx = np.zeros((1, 3))
    chunk = list(chunk)
    Nsnapshots3 = len(chunk)
    envLiTFSI = np.zeros((Nsnapshots3, NLi))
    
    xLi_ds_chunk = xLi_ds[chunk[0] * NLi:(chunk[-1] + 1) * NLi, :]
    xTFSI_ds_chunk = xTFSI_ds[chunk[0] * NTFSI:(chunk[-1] + 1) * NTFSI, :]
    
    for nt in range(Nsnapshots3):
        for itemi in range(NLi):
            for itemj in range(NTFSI):
                dx[0, :] = xLi_ds_chunk[nt * NLi + itemi, :] - xTFSI_ds_chunk[nt * NTFSI + itemj, :]
                dx_PBC_corrected = perform_PBC(L, dx)
                dx_abs = np.sqrt(np.sum(np.power(dx_PBC_corrected, 2)))
                temp = (1 - np.power(dx_abs / d0, nCN)) / (1 - np.power(dx_abs / d0, mCN))
                envLiTFSI[nt, itemi] += temp
    
    return envLiTFSI

def calculate_CN_TFSI(xLi_ds, xTFSI_ds, NLi, NTFSI, L, d0, nCN, mCN, num_processes):
    """
    Manages parallel processing for CN calculation between Li and TFSI atoms across all snapshots.
    
    Parameters:
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms.
    - xTFSI_ds (numpy.ndarray): Downsampled coordinates for TFSI atoms.
    - NLi (int): Number of Li atoms.
    - NTFSI (int): Number of TFSI atoms.
    - L (numpy.ndarray): Simulation box dimensions.
    - d0 (float): Cutoff distance for CN calculation.
    - nCN (int): Exponent in CN calculation formula.
    - mCN (int): Exponent in CN calculation formula.
    - num_processes (int): Number of processes for parallelization.
    
    Returns:
    - envLiTFSI (numpy.ndarray): Coordination numbers for Li atoms with TFSI atoms across all snapshots.
    """
    Nsnaps = xLi_ds.shape[0] // NLi
    m_range = range(Nsnaps)
    chunks = split_into_chunks(m_range, num_processes)
    
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(CN_calc_TFSI_call, [(chunk, xLi_ds, xTFSI_ds, NLi, NTFSI, L, d0, nCN, mCN) for chunk in chunks])
    
    return np.concatenate(results, axis=0)

def calculate_unwrapped_coordinates(xLi_ds, NLi, Nsnaps, L):
    """
    Calculates unwrapped coordinates for Li atoms by removing the effects of periodic boundaries.
    
    Parameters:
    - xLi_ds (numpy.ndarray): Downsampled coordinates for Li atoms in Cartesian coordinates.
    - NLi (int): Number of Li atoms.
    - Nsnaps (int): Number of snapshots.
    - L (numpy.ndarray): Simulation box dimensions, 3x3 array.
    
    Returns:
    - r_real (numpy.ndarray): Unwrapped coordinates for Li atoms, shape (Nsnaps, NLi, 3).
    """
    # Initialize Cartesian and real (unwrapped) coordinates arrays
    r_cartesian = np.zeros((Nsnaps, NLi, 3))
    r_real = np.zeros((Nsnaps, NLi, 3))

    # Copy the downsampled Cartesian coordinates
    for nt in range(Nsnaps):
        r_cartesian[nt] = xLi_ds[0 + nt * NLi:NLi + nt * NLi, :] 

    # Box dimensions for each direction
    x_size, y_size, z_size = L[0, 0], L[1, 1], L[2, 2]

    # Calculate unwrapped coordinates
    dx = np.zeros((1, 3))
    for m in range(Nsnaps):
        if m == 0:  # Initialize positions at time step zero
            r_real[m] = r_cartesian[m]
        else:
            for n in range(NLi):
                dx[0, 0] = r_cartesian[m, n, 0] - r_cartesian[m - 1, n, 0]
                dx[0, 1] = r_cartesian[m, n, 1] - r_cartesian[m - 1, n, 1]
                dx[0, 2] = r_cartesian[m, n, 2] - r_cartesian[m - 1, n, 2]

                # Correct for periodic boundaries
                dx_PBC_corrected = perform_PBC(L, dx)

                # Accumulate unwrapped position
                r_real[m, n, 0] = r_real[m - 1, n, 0] + dx_PBC_corrected[0, 0]
                r_real[m, n, 1] = r_real[m - 1, n, 1] + dx_PBC_corrected[0, 1]
                r_real[m, n, 2] = r_real[m - 1, n, 2] + dx_PBC_corrected[0, 2]

    return r_real

def calculate_msd_for_each_atom(r_real, NLi):
    """
    Calculates the Mean Square Displacement (MSD) for each Li atom individually.

    Parameters:
    - r_real (numpy.ndarray): Unwrapped coordinates of Li atoms, shape (Nsnaps, NLi, 3).
    - NLi (int): Number of Li atoms.

    Returns:
    - numpy.ndarray: MSD for each Li atom, shape (NLi, Nsnaps - 1).
    """
    # Calculate MSD for each Li atom using msd_fft_cross
    msd_atom_ind = [msd_fft_cross(r_real[:-1, :, :], r_real[:-1, :, :], n) for n in range(NLi)]
    return np.array(msd_atom_ind)

def r_real_env_prjctd_call(chunk, Nsnaps, r_real_reformatted, r_real, Env_cont_val, Env_cont_indx, topWhat, NLi, Nsites):
    """
    Calculates MSD between environments for a chunk of Li atoms.
    
    Parameters:
    - chunk (list): Indices of Li atoms in the current chunk.
    - Nsnaps (int): Number of snapshots.
    - r_real_reformatted (numpy.ndarray): Reformatted unwrapped coordinates, shape (NLi * Nsnaps, 3).
    - Env_cont_val (numpy.ndarray): Values representing the environment (e.g., coordination values).
    - Env_cont_indx (numpy.ndarray): Indices representing environment states.
    - topWhat (int): Number of top environments to consider.
    - NLi (int): Number of Li atoms.
    - Nsites (int): Total number of environmental sites.

    Returns:
    - numpy.ndarray: MSD for each environment pair, shape (Nsites, Nsites, Nsnaps - 1).
    """
    if not chunk:
        return np.zeros((Nsites, Nsites, Nsnaps - 1))  # Return empty array for empty chunk
    
    NLi3 = len(chunk)
    Nsnapshots3 = Nsnaps - 1
    r_real_env_for_fft = np.zeros((Nsnapshots3, 1, 3))
    r_real_for_fft = np.zeros((Nsnapshots3, 1, 3))
    r_real_reformatted_chunk = r_real_reformatted[chunk[0]*Nsnaps:(chunk[-1]+1)*Nsnaps, :]
    msd_env_temp = np.zeros((Nsites, Nsites, Nsnapshots3))
    
    for nLi_indx in range(NLi3):  # Loop over Li atoms
        nLi = chunk[nLi_indx]
        r_real_env_prjctd_temp = np.zeros((Nsnapshots3, Nsites, Nsites, 1, 3))
        
        for alpha in range(3):  # Loop over x, y, and z
            r_real_under_consideration = r_real_reformatted_chunk[nLi_indx * Nsnaps:(nLi_indx + 1) * Nsnaps, alpha]
            
            for nt in range(Nsnapshots3):
                sum_for_normalization = 0
                indx1 = nt * NLi + nLi
                indx2 = (nt + 1) * NLi + nLi
                
                for envi in range(topWhat):
                    for envj in range(topWhat):
                        idx_i_env = Env_cont_indx[indx1][envi]
                        idx_j_env = Env_cont_indx[indx2][envj]
                        sum_for_normalization += Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]
                
                for envi in range(topWhat):
                    for envj in range(topWhat):
                        idx_i_env = Env_cont_indx[indx1][envi]
                        idx_j_env = Env_cont_indx[indx2][envj]
                        r_real_env_prjctd_temp[nt, idx_i_env, idx_j_env, 0, alpha] += (
                            r_real_under_consideration[nt] *
                            (Env_cont_val[indx1][envi] * Env_cont_val[indx2][envj]) /
                            sum_for_normalization
                        )
        
        for envi in range(Nsites):
            for envj in range(Nsites):
                r_real_env_for_fft[:, 0, :] = r_real_env_prjctd_temp[:, envi, envj, 0, :]
                #r_real_for_fft[:, 0, :] = r_real_reformatted[:Nsnaps-1, nLi, :]
                r_real_for_fft[:, 0, :] = r_real[:Nsnaps-1, nLi, :]
                msd = np.array([msd_fft_cross(r_real_env_for_fft, r_real_for_fft, n) for n in range(1)])
                msd_env_temp[envi, envj, :] += msd[0, :]
                
    return msd_env_temp

def calculate_msd_between_environments(r_real_reformatted, r_real, Env_cont_val, Env_cont_indx, Nsites, Nsnaps, NLi, topWhat, num_processes):
    """
    Manages parallel processing for MSD calculation between environmental states.
    
    Parameters:
    - r_real_reformatted (numpy.ndarray): Reformatted unwrapped coordinates, shape (NLi * Nsnaps, 3).
    - r_real (numpy.ndarray): Unwrapped coordinates of Li atoms, shape (Nsnaps, NLi, 3).
    - Env_cont_val (numpy.ndarray): Values representing the environment.
    - Env_cont_indx (numpy.ndarray): Indices representing environment states.
    - Nsites (int): Total number of environmental sites.
    - Nsnaps (int): Number of snapshots.
    - NLi (int): Number of Li atoms.
    - topWhat (int): Number of top environments to consider.
    - num_processes (int): Number of processes for parallelization.
    
    Returns:
    - numpy.ndarray: MSD for each environment pair, shape (Nsites, Nsites, Nsnaps - 1).
    """
    m_range = range(NLi)
    chunks = split_into_chunks(m_range, num_processes)
    
    with Pool(num_processes) as pool:
        results = pool.starmap(r_real_env_prjctd_call, [(chunk, Nsnaps, r_real_reformatted, r_real, Env_cont_val, Env_cont_indx, topWhat, NLi, Nsites) for chunk in chunks])
    
    return np.sum(np.array(results), axis=0) / NLi

def write_top_what_results(env_data, Nsnaps, NLi, topWhat, filename_values, filename_indices):
    """
    Processes and writes the top 'what' largest values and their indices for coordination numbers to files.

    Parameters:
    - env_data (numpy.ndarray): Coordination numbers, such as envLiLa or envLiTFSI.
    - Nsnaps (int): Number of snapshots.
    - NLi (int): Number of Li atoms.
    - topWhat (int): Number of top values to save.
    - filename_prefix (str): Prefix for the output files.
    """
    # Arrays to store top values and indices
    normalized_top_what_largest = np.zeros((Nsnaps * NLi, topWhat))
    indices_top_what_largest = np.zeros((Nsnaps * NLi, topWhat), dtype=int)

    for n in range(Nsnaps):
        if n % 50000 == 0:
            print(f"Processing snapshot {n}")
        for i in range(NLi):
            if env_data[n, i] - np.floor(env_data[n, i]) < 0.5:
                indices_top_what_largest[n * NLi + i, :] = [int(np.floor(env_data[n, i])), int(np.ceil(env_data[n, i]))]
                normalized_top_what_largest[n * NLi + i, :] = [
                    np.ceil(env_data[n, i]) - env_data[n, i],
                    env_data[n, i] - np.floor(env_data[n, i])
                ]
            else:
                indices_top_what_largest[n * NLi + i, :] = [int(np.ceil(env_data[n, i])), int(np.floor(env_data[n, i]))]
                normalized_top_what_largest[n * NLi + i, :] = [
                    env_data[n, i] - np.floor(env_data[n, i]),
                    np.ceil(env_data[n, i]) - env_data[n, i]
                ]

    # Write top values and indices to separate files
    values_filename = filename_values
    indices_filename = filename_indices
    
    # Save values
    with open(values_filename, "w") as f_values:
        for n in range(Nsnaps):
            for i in range(NLi):
                f_values.write(" ".join(map(str, normalized_top_what_largest[n * NLi + i, :])) + "\n")

    # Save indices
    with open(indices_filename, "w") as f_indices:
        for n in range(Nsnaps):
            for i in range(NLi):
                f_indices.write(" ".join(map(str, indices_top_what_largest[n * NLi + i, :])) + "\n")

def print_and_save_results(envLiLa, envLiTFSI, msd_env, msd_atom_ind, filename):
    """
    Prints the shapes of key arrays and saves them to an .npz file.

    Parameters:
    - envLiLa (numpy.ndarray): Coordination numbers for Li-PEO interactions.
    - envLiTFSI (numpy.ndarray): Coordination numbers for Li-TFSI interactions.
    - msd_env (numpy.ndarray): MSD between environmental states.
    - msd_atom_ind (numpy.ndarray): MSD for each Li atom.
    - filename (str): Name of the output .npz file.
    """
    # Print shapes of each array
    print("envLiLa shape:", envLiLa.shape)
    print("envLiTFSI shape:", envLiTFSI.shape)
    print("msd_env shape:", msd_env.shape)
    print("msd_atom_ind shape:", msd_atom_ind.shape)

    # Save the arrays to an .npz file
    np.savez(filename, envLiLa=envLiLa, envLiTFSI=envLiTFSI, msd_env=msd_env, msd_atom_ind=msd_atom_ind)

