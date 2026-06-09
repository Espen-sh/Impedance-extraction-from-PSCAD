# Impedance-extraction-from-PSCAD
## Impedance Extraction and Stability Analysis

This repository contains Python scripts used in the masters thesis conducted by Espen Solberg Hågensen and Christoffer Strugstad Jenssen. 

PSCAD output files for base case and tuned case can be found here: https://drive.google.com/drive/folders/1Zf5BuiLuLfIVf_3Fcok90znTn522aLff?usp=drive_link 

### `ImpedanceExtractionTutorial.py`
Main script.

- Computes analytical impedance of the RLC load and source.
- Reads PSCAD results (`zero_inj.out`, `pos_inj.out`, `neg_inj.out`).
- Extracts measured impedances from the PSCAD data.
- Plots:
  - Bode plots (magnitude and phase) for load impedance.
  - Bode plots for source impedance.
  - Nyquist plot of loop-gain eigenvalues.
- Prints error metrics between analytical and measured impedances.

### `Functions.py`
Helper functions used by `Use.py`.

Includes functions for:
- Analytical RLC impedance in dq and modified sequence domains.
- Loading PSCAD `.out` files.
- FFT and symmetrical components.
- Finding the best time window for FFT.
- Extracting Zpn for load and source.
- Making the Bode-style plot panels.


## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Pillow (PIL)



