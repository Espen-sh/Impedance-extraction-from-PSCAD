# Impedance Extraction and Stability Analysis

This repository contains the Python post-processing scripts used for impedance extraction and impedance-based stability analysis in the master thesis. The scripts are intended to support the methodology described in the thesis chapter on impedance extraction.

The workflow is based on frequency-domain extraction from PSCAD time-domain simulations. Multi-tone positive- and negative-sequence perturbation files are used to identify 2×2 modified-sequence impedance matrices. These matrices can then be converted to the dq domain and used for impedance plots, passivity analysis, and Nyquist-based stability assessment.

### `ImpedanceExtractionTutorial.py`

This is the main script. It defines:

- PSCAD output file locations
- signal column mapping
- voltage and current scaling
- extraction time window
- multi-tone frequency lists
- base-case and shaped-case simulation runs
- plotting and stability-analysis workflow

The script performs both a single-run example and a full multi-run extraction for a base case and a shaped case.
This script is intended as a tutorial on how to use and understand the implemented 

### `Functions.py`

This file contains the supporting functions used by the tutorial script, including routines for:

- loading PSCAD `.out` files
- extracting fixed time windows
- computing FFT spectra
- calculating symmetrical components
- aligning spectra to the fundamental positive-sequence voltage angle
- extracting modified-sequence sidebands
- identifying 2×2 modified-sequence impedance matrices
- converting modified-sequence impedance matrices to the dq domain
- checking multi-tone frequency lists for sideband collisions
- collecting several positive/negative injection runs into one frequency sweep
- plotting Bode diagrams, passivity indices, Nyquist eigenvalue loci, and CoF-corrected diagonal impedances
- computing signal-to-background quality metrics for extraction validation

## Method overview

The extraction uses one positive-sequence and one negative-sequence perturbation file for each frequency list. For each injected modified frequency, the script extracts the corresponding upper and mirror sidebands around the fundamental frequency. These sideband quantities are used to solve the modified-sequence impedance matrix

```text
[V_p]   [Z_pp  Z_pn] [I_p]
[V_n] = [Z_np  Z_nn] [I_n]
```

for both the load side and the source side of the interface.

After extraction, the modified-sequence impedance matrices can be transformed to the dq domain. The dq-domain impedance matrices are then used for passivity and generalized Nyquist analysis.

## Requirements

The scripts require Python 3.10 or newer. Python 3.12 was used during development.

Install the required Python packages with:

```bash
pip install numpy matplotlib pillow
```

The scripts also assume that the PSCAD `.out` files are available locally. These output files are not included in this repository unless explicitly added.

## Expected PSCAD output files

By default, the tutorial script expects the PSCAD output files to be located in the same directory as `ImpedanceExtractionTutorial.py`.

The output files for the base case and tuned case can be found and downloaded here: 
https://drive.google.com/drive/folders/1Zf5BuiLuLfIVf_3Fcok90znTn522aLff?usp=drive_link

The base-case files are expected to use the following naming convention:

```text
Ybase_pos1.out   Ybase_neg1.out
Ybase_pos2.out   Ybase_neg2.out
Ybase_pos3.out   Ybase_neg3.out
Ybase_pos4.out   Ybase_neg4.out
Ybase_pos5.out   Ybase_neg5.out
Ybase_pos6.out   Ybase_neg6.out
```

The shaped-case files are expected to use:

```text
Ytuned_pos1.out   Ytuned_neg1.out
Ytuned_pos2.out   Ytuned_neg2.out
Ytuned_pos3.out   Ytuned_neg3.out
Ytuned_pos4.out   Ytuned_neg4.out
Ytuned_pos5.out   Ytuned_neg5.out
Ytuned_pos6.out   Ytuned_neg6.out
```

The signal-to-background quality check also expects an unperturbed file:

```text
YwnoInj.out
```

If your PSCAD files use other names, update the `RUN_BASE`, `RUN_SHAPED`, and `FILE_BASE` variables in `ImpedanceExtractionTutorial.py`.

## PSCAD signal layout

The PSCAD output column layout is defined in the `colmap` dictionary in `ImpedanceExtractionTutorial.py`:

```python
colmap = {
    "t": 0,
    "Va": 9, "Vb": 2, "Vc": 4,
    "Ia": 6, "Ib": 7, "Ic": 8,
    "dVa": 1, "dVb": 3, "dVc": 5
}
```

The voltage and current scaling factors are also defined in the tutorial script:

```python
v_scale = 1000.0
i_scale = 1000.0
```

This assumes that the PSCAD output channels are given in kV and kA. If your PSCAD model exports values directly in V and A, set these factors to `1.0`.

## Frequency lists

The script uses six multi-tone frequency lists:

- Lists 1–4: broad frequency coverage
- List 5: dense frequency coverage around the first selected interaction region
- List 6: dense frequency coverage around the second selected interaction region

Each list should be checked to avoid modified-sequence sideband collisions. The helper function

```python
check_msd_multitone_list(f_meas_list, f1)
```

checks for:

- duplicate mirror bins
- upper/mirror sideband overlap
- special cases at 0 Hz, 50 Hz, and 100 Hz

The frequency lists in the tutorial script were selected to avoid these collisions for a 50 Hz fundamental frequency.

## Extraction window

The extraction window is defined as:

```python
T_win = 6.0
FIXED_T_START = 3.999
FIXED_T_END = FIXED_T_START + T_win
```

The window should be placed after the system has reached steady state. It should also contain an integer number of periods for the relevant frequency resolution to reduce FFT leakage.

## Running the script

Place the required PSCAD `.out` files in the same folder as the Python scripts, then run:

```bash
python ImpedanceExtractionTutorial.py
```

The script creates a local `Figures` directory and saves generated plots there.

## Generated outputs

The tutorial script produces figures for:

- single-run dq-domain load impedance
- base-case vs shaped-case load impedance in the modified-sequence domain
- base-case vs shaped-case load impedance in the dq domain
- passivity index comparison
- Nyquist loop-gain eigenvalue comparison
- CoF-corrected diagonal impedance comparison
- sideband signal-to-background quality metrics

The exact output filenames are defined by the `title` arguments passed to the plotting functions.
