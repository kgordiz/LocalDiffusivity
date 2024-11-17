import numpy as np

def crosscorFFT(x, y):
    """
    Calculates the cross-correlation of two signals using FFT with zero-padding.

    Parameters:
    - x, y (numpy.ndarray): Input signals of the same length.

    Returns:
    - numpy.ndarray: Cross-correlation result, normalized by (N-m).
    """
    assert len(x) == len(y), "Signals x and y must have the same length"
    N = len(x)
    F_x = np.fft.fft(x, n=2 * N)  # Zero-padding to 2*N
    F_y = np.fft.fft(y, n=2 * N)  # Zero-padding to 2*N
    PSD = F_x * F_y.conjugate()
    res = np.fft.ifft(PSD)
    res = res[:N].real  # Cross-correlation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # Normalize by (N-m)
    return res / n  # Cross-correlation in convention A

def msd_fft_cross(r1, r2, n):
    """
    Computes the Mean Square Displacement (MSD) for an atom by cross-correlating
    its coordinates across frames using FFT.

    Parameters:
    - r1, r2 (numpy.ndarray): Position arrays of shape (nframes, natoms, 3).
    - n (int): Index of the atom for which MSD is calculated.

    Returns:
    - numpy.ndarray: MSD for the specified atom across frames.
    """
    # Check if dimensions of r1 and r2 match
    if r1.shape != r2.shape:
        raise ValueError("The dimensions of r1 and r2 must be the same")
    
    N = r1.shape[0]  # Number of frames

    # Compute the dot product across Cartesian coordinates, resulting in shape (nframes, natoms)
    D = np.sum(r1 * r2, axis=2)

    # Pad D with a zero for each atom, resulting in shape (nframes+1, natoms)
    npad = ((0, 1), (0, 0))
    D = np.pad(D, pad_width=npad, mode='constant', constant_values=0)
    nframes, natoms = D.shape

    # Calculate cross-correlation for each Cartesian component
    S2_xy = sum([crosscorFFT(r1[:, n, a], r2[:, n, a]) for a in range(r1.shape[2])])
    S2_yx = sum([crosscorFFT(r2[:, n, a], r1[:, n, a]) for a in range(r1.shape[2])])

    # Sum over all time steps for all atoms
    Q = 2 * D.sum(axis=0)

    Ndim = 3  # Number of spatial dimensions

    # Initialize S1 and calculate for each frame
    S1 = np.zeros(N)
    for m in range(N):
        Q[n] = Q[n] - D[m - 1, n] - D[N - m, n]
        S1[m] = Q[n] / (N - m)
    
    # MSD calculation
    return (S1 - S2_xy - S2_yx) / (2 * Ndim)

