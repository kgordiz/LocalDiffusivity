
# Local Diffusion Analysis

This repository provides tools for analyzing local diffusion and coordination environments of ions in molecular dynamics (MD) simulations. The package includes functions for calculating coordination numbers, unwrapping coordinates, downsampling, and generating **full MSD curves** for each environment-to-environment diffusion using FFT, removing the need for previously used Tij matrices.

## Directory Structure

```
local_diffusion_analysis/
├── __init__.py
├── file_io.py
├── data_processing.py
├── fft_utils.py
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

To install the package, clone this repository and install the dependencies:

```bash
git clone https://github.com/kgordiz/LocalDiffusivity
cd LocalDiffusivity
pip install -r requirements.txt
```

Alternatively, you can install the package directly with `setup.py`:

```bash
pip install .
```

## Usage

To run the main analysis, execute:

```bash
python main.py
```

This script performs computations on a specified LAMMPS trajectory file and outputs results in `.npz` format for further analysis. Modify parameters within `main.py` as needed for your dataset.

### Example Workflow

1. **Trajectory Processing**: Reads atomic trajectories from LAMMPS files.
2. **Coordination Number Calculation**: Computes local coordination environments based on atom types.
3. **Full MSD Curves**: Calculates full MSD curves for each environment-to-environment diffusion using FFT, eliminating the need for Tij matrices.
4. **Saving Results**: Stores computed matrices in `.npz` format for external post-processing.

## Key Functions

- **Trajectory Reading**: `readtrajfile` loads atomic positions and types from a LAMMPS trajectory file.
- **Periodic Boundary Correction**: `perform_PBC` applies periodic boundary corrections to atomic displacements.
- **Coordination Number Calculation**: `calculate_coordination_number` computes coordination numbers between Li ions and specific atom types.
- **Unwrapped Coordinates**: `calculate_unwrapped_coordinates` removes periodic boundary effects for MSD calculation.
- **Full MSD Calculation**: `calculate_msd_between_environments` manages parallel processing to calculate full MSD curves between environments using FFT.
- **Saving Data**: `print_and_save_results` saves critical variables (e.g., environment matrices, MSD data) to an `.npz` file.

## Citation

If you use this package in your research, please cite:

**DOI**: [10.1021/acsnano.4c09552](https://doi.org/10.1021/acsnano.4c09552)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
