
# LocalDiffusivity

This repository provides tools for analyzing local diffusion and coordination environments of ions in molecular dynamics (MD) simulations. The package includes functions for calculating coordination numbers, unwrapping coordinates, downsampling, and generating **Tij matrices** based on cross-cuts of mean square displacements (MSD). 

## Directory Structure

```
LocalDiffusivity/
├── local_diffusivity/
│   ├── __init__.py
│   ├── file_io.py
│   ├── data_processing.py
│   └── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

To install, clone this repository and install the dependencies listed in `requirements.txt`:

```bash
git clone <repository_url>
cd LocalDiffusivity
pip install -r requirements.txt
```

## Usage

Run the main analysis script with:

```bash
python main.py
```

This will perform a series of computations on a given LAMMPS trajectory file and output results to `.npz` files for further post-processing. Please customize parameters within `main.py` as needed for your specific dataset.

### Example Workflow

1. **Trajectory Processing**: Reads atomic trajectories from LAMMPS files.
2. **Coordination Number Calculation**: Calculates local coordination environments based on atom types.
3. **MSD and Tij Matrices**: Computes cross-cuts of local MSDs for each environment to estimate local diffusion.
4. **Saving Results**: Saves computed matrices in `.npz` format for external analysis.

## Key Functions

- **Trajectory Reading**: `readtrajfile` loads atomic positions and types from a LAMMPS trajectory file.
- **Periodic Boundary Correction**: `perform_PBC` applies periodic boundary corrections to atomic displacements.
- **Coordination Number (CN) Calculation**: `calculate_coordination_number` computes CN between Li ions and specific atom types.
- **Unwrapped Coordinates**: `get_unwrapped_coordinates` removes periodic boundary effects for MSD calculation.
- **Tij Matrix Calculation**: `calculate_Tij_parallel` manages parallel processing to calculate cross-cuts of local MSDs.
- **Time Array Creation**: `create_time_array` generates a time array for snapshots in picoseconds.
- **Saving Data**: `save_variables_to_npz` saves critical variables (e.g., environment matrices, time array) to an `.npz` file.

## Citation

If you use this package in your research, please cite the following paper:

**DOI**: [10.1021/acsnano.4c09552](https://doi.org/10.1021/acsnano.4c09552)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
