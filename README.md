# Impedance-extraction-from-PSCAD
# Impedance Extraction and Stability Analysis for an RLC Test System

This repository contains Python scripts for validating a multi-tone impedance
measurement method on a simple three-phase RLC test system modeled in PSCAD.

### `Use.py`
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



