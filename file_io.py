# file_io.py
import numpy as np

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
        for line in lines:
            words = line.split()
            if len(words) > 0 and words[-1] == 'TIMESTEP':
                Ntimestep += 1
            
        Natoms = int(lines[3].split()[0])
        L = np.zeros((3, 3))
        L[0, 0] = float(lines[5].split()[1])
        L[1, 1] = float(lines[6].split()[1])
        L[2, 2] = float(lines[7].split()[1])
        
        Ntimestep -= 1
        coords_xyz = np.zeros((Ntimestep, Natoms, 3))
        coords_id_type = np.zeros((Ntimestep, Natoms, 2), dtype=int)
        
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

