# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:53:09 2026

@author: espen

Tutorial script for impedance extraction and stability analysis.

This script is meant as a commented example of how the functions in
Functions_v2.py can be used.

The workflow is:

1. Define the PSCAD signal layout, scaling, time window, and frequency lists.
2. Extract one 2x2 impedance matrix set from one positive/negative file pair.
3. Extract a full base case from several positive/negative file pairs.
4. Extract a full shaped case in the same way.
5. Compare the base case and shaped case using impedance, passivity, and
   Nyquist loop-gain plots.

The script assumes that the PSCAD .out files and Functions_v2.py are located
in the same folder as this script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from Functions import *


# ============================================================
# 1. Paths and general settings
# ============================================================

# Use the folder containing this script as the working folder.
# This means all PSCAD .out files are expected to be placed in the same folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All generated figures are stored in a local Figures folder.
FIG_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Fundamental grid frequency.
# The modified-sequence extraction uses this to locate the sidebands
# around the fundamental frequency.
f1 = 50.0

# Column map for the PSCAD .out files.
colmap = {
    "t": 0, # Time
    "Va": 9, "Vb": 2, "Vc": 4, # Load side voltages
    "Ia": 6, "Ib": 7, "Ic": 8, # Load side currents
    "dVa": 1, "dVb": 3, "dVc": 5 # Source side voltages
}

# Scaling from PSCAD output channels to SI units.
# In this case, the PSCAD output is assumed to be in kV and kA,
# so both voltage and current are multiplied by 1000.
v_scale = 1000.0
i_scale = 1000.0

skiprows = 0 # Number of rows to skip when reading the .out files.

# The extraction window should be placed after the simulation has reached
# steady state and should contain an integer number of fundamental periods.
#
# Here, a 6 second window is used from 3.999 s to 9.999 s.
T_win = 6.0
FIXED_T_START = 3.999
FIXED_T_END = FIXED_T_START + T_win

# If the same frequency appears in several runs, this controls whether the
# extracted values are averaged.
AVERAGE_DUPLICATES = False

# Plot labels for modified-sequence and dq-domain impedance matrices.
pn_labels = {"11": "pp", "12": "pn", "21": "np", "22": "nn"}
dq_labels = {"11": "dd", "12": "dq", "21": "qd", "22": "qq"}

# Common plot limits.
LOAD_MAG_YLIM = (1e1, 1e5)
PHASE_YLIM = (-190, 190)

# Nyquist axis limits.
NYQ_XLIM = (-1.05, 1.05)
NYQ_YLIM = (-0.1, 1.5)

# Optional vertical markers in the impedance and passivity plots.
CRITICAL_FREQS = None # Disabled
# CRITICAL_FREQS = (176.0, 272.0)


# ============================================================
# 2. Frequencies
# ============================================================

# The frequencies correspond to the injected in one PSCAD simulation.


f_meas_list1 = np.array([
    1, 10, 53, 56, 103, 106, 211, 216,
    5, 16, 65, 68, 126, 130, 286, 301,
    9, 24, 80, 172, 155, 159, 477, 515
], dtype=int)

f_meas_list2 = np.array([
    2, 12, 54, 57, 108, 111, 222, 176,
    6, 22, 40, 70, 133, 136, 317, 333,
    26, 28, 82, 86, 163, 167, 556, 600
], dtype=int)

f_meas_list3 = np.array([
    3, 11, 59, 62, 114, 117, 239, 245,
    18, 20, 72, 76, 140, 144, 351, 379,
    32, 36, 88, 93, 84, 67, 682, 775
], dtype=int)

f_meas_list4 = np.array([
    4, 14, 60, 63, 120, 123, 258, 272,
    8, 7, 74, 78, 147, 151, 409, 442,
    233, 44, 91, 95, 190, 200, 880, 1000
], dtype=int)
# Dense frequency list around the second critical interaction region.
f_meas_list5 = np.array([
    173, 174, 175, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 201, 202, 203, 204, 205,
    206, 207, 208, 209, 210, 212, 213, 214, 215, 217
], dtype=int)
# Dense frequency list around the second critical interaction region.
f_meas_list6 = np.array([
    273, 275, 277, 279, 281, 282, 283, 284, 285, 287,
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297,
    298, 299, 302, 303, 304, 305, 306, 307, 308, 309,
    310, 311, 312, 313, 314, 315, 316, 318, 320, 323
], dtype=int)

# A list can be checked for sideband collisions with check_msd_multitone_list(..., f1).

check_msd_multitone_list(f_meas_list1, f1)

# ============================================================
# 3. Single positive/negative file-pair example
# ============================================================

# This section shows the smallest useful extraction example.

file_pos_single = os.path.join(BASE_DIR, "Ybase_pos1.out") # Define positive sequence injection.
file_neg_single = os.path.join(BASE_DIR, "Ybase_neg1.out") # Define negative sequence injection.
f_single = f_meas_list1 # The frequencies injected


# Extract load-side and source-side impedance matrices from one file pair.
ZL_single_pn, ZS_single_pn = Zpn_load_and_source_from_two_files(
    file_pos_inj=file_pos_single,
    file_neg_inj=file_neg_single,
    t_start=FIXED_T_START,
    t_end=FIXED_T_END,
    f1=f1,
    f_sweep_list=f_single,
    skiprows=skiprows,
    colmap=colmap,
    v_scale=v_scale,
    i_scale=i_scale,
)

# Sort the single-run result by frequency before plotting.
# Makes the Bode plot lines connect the points in the correct frequency order.
idx = np.argsort(f_single)

f_single = f_single[idx]
ZL_single_pn = ZL_single_pn[idx]
ZS_single_pn = ZS_single_pn[idx]


# Convert the single-run result from modified sequence domain to dq domain.
ZL_single_dq = np.array([Zpn_to_dq(Z) for Z in ZL_single_pn])

# Plot the load-side impedance from this one run in the dq domain.
# Here, only one case is passed to the function. The same function can also
# be used later to compare two or three cases by adding Zmat2 and Zmat3.
plot_bode_matrix_overlay_3(
    f_hz=f_single,                       # Frequency vector [Hz] for this run only.
    Zmat1=ZL_single_dq,                  # Load-side dq impedance matrix from the single run.
    names=("Base case, run 1",),         # Legend label for Zmat1.
    colors=("0.10",),                    # Curve color for Zmat1.
    title="Tutorial_single_run_load_impedance_dq",  # Figure is saved with this filename.
    phase_wrapped=True,                  # Wrap phase to the interval [-180, 180] degrees.
    phase_ylim=PHASE_YLIM,               # Common y-axis limits for all phase panels.
    mag_ylim=LOAD_MAG_YLIM,              # Common y-axis limits for all magnitude panels.
    labels=dq_labels,                    # Use dq labels: dd, dq, qd, qq.
    critical_freqs=CRITICAL_FREQS,       # Optional vertical frequency markers.
    save_dir=FIG_DIR,                    # Folder where the figure is saved.
)


# ============================================================
# 4. Multirun case definitions
# ============================================================

# The complete case extraction uses several positive/negative file pairs.
#
# RUN_BASE uses the base-case weak-grid files:
#   Ybase_posX.out / Ybase_negX.out
#
# RUN_SHAPED uses the final shaped-case files:
#   Ytuned_posX.out / Ytuned_negX.out

# Each dictionary entry contains:
#   file_pos : positive-sequence injection file
#   file_neg : negative-sequence injection file
#   f_list   : frequencies injected in that file pair

RUN_BASE = {
    "run1": {"file_pos": "Ybase_pos1.out", "file_neg": "Ybase_neg1.out", "f_list": f_meas_list1},
    "run2": {"file_pos": "Ybase_pos2.out", "file_neg": "Ybase_neg2.out", "f_list": f_meas_list2},
    "run3": {"file_pos": "Ybase_pos3.out", "file_neg": "Ybase_neg3.out", "f_list": f_meas_list3},
    "run4": {"file_pos": "Ybase_pos4.out", "file_neg": "Ybase_neg4.out", "f_list": f_meas_list4},
    "run5": {"file_pos": "Ybase_pos5.out", "file_neg": "Ybase_neg5.out", "f_list": f_meas_list5},
    "run6": {"file_pos": "Ybase_pos6.out", "file_neg": "Ybase_neg6.out", "f_list": f_meas_list6},
}

RUN_SHAPED = {
    "run1": {"file_pos": "Ytuned_pos1.out", "file_neg": "Ytuned_neg1.out", "f_list": f_meas_list1},
    "run2": {"file_pos": "Ytuned_pos2.out", "file_neg": "Ytuned_neg2.out", "f_list": f_meas_list2},
    "run3": {"file_pos": "Ytuned_pos3.out", "file_neg": "Ytuned_neg3.out", "f_list": f_meas_list3},
    "run4": {"file_pos": "Ytuned_pos4.out", "file_neg": "Ytuned_neg4.out", "f_list": f_meas_list4},
    "run5": {"file_pos": "Ytuned_pos5.out", "file_neg": "Ytuned_neg5.out", "f_list": f_meas_list5},
    "run6": {"file_pos": "Ytuned_pos6.out", "file_neg": "Ytuned_neg6.out", "f_list": f_meas_list6},
}


# ============================================================
# 5. Extract complete base case
# ============================================================

# The complete base case is extracted with collect_all_runs(...).
#
# This function loops through all entries in RUN_BASE. For each run, it:
#   1. Loads one positive-sequence injection file and one negative-sequence file.
#   2. Extracts the selected time window.
#   3. Uses the FFT sidebands to identify the 2x2 modified-sequence impedance.
#   4. Appends the result to the complete frequency vector.
#
# The output matrices have shape:
#   (number of frequencies, 2, 2)

f_base, ZL_base_pn, ZS_base_pn = collect_all_runs(
    base_dir=BASE_DIR,
    runs=RUN_BASE,
    f1=f1,
    skiprows=skiprows,
    colmap=colmap,
    v_scale=v_scale,
    i_scale=i_scale,
    average_duplicates=AVERAGE_DUPLICATES,
    use_fixed_window=True,
    fixed_t_start=FIXED_T_START,
    fixed_t_end=FIXED_T_END,
)

# The extraction is performed in the modified sequence domain,
# but can be convert to the dq-domain.
ZL_base_dq = np.array([Zpn_to_dq(Z) for Z in ZL_base_pn])
ZS_base_dq = np.array([Zpn_to_dq(Z) for Z in ZS_base_pn])


# ============================================================
# 6. Extract complete shaped case
# ============================================================

# The final shaped case is extracted in exactly the same way.
# Only the file names change through the RUN_SHAPED dictionary.

f_shaped, ZL_shaped_pn, ZS_shaped_pn = collect_all_runs(
    base_dir=BASE_DIR,
    runs=RUN_SHAPED,
    f1=f1,
    skiprows=skiprows,
    colmap=colmap,
    v_scale=v_scale,
    i_scale=i_scale,
    average_duplicates=AVERAGE_DUPLICATES,
    use_fixed_window=True,
    fixed_t_start=FIXED_T_START,
    fixed_t_end=FIXED_T_END,
)

ZL_shaped_dq = np.array([Zpn_to_dq(Z) for Z in ZL_shaped_pn])
ZS_shaped_dq = np.array([Zpn_to_dq(Z) for Z in ZS_shaped_pn])


# ============================================================
# 7. Compare load-side impedance in modified sequence domain
# ============================================================

# This plot compares the extracted load-side terminal impedance in the
# modified sequence domain.

plot_bode_matrix_overlay_3(
    f_hz=f_base,
    Zmat1=ZL_base_pn,
    Zmat2=ZL_shaped_pn,
    names=("Base case", "Final shaped case"),
    colors=("0.10", "#CC3311"),
    title="Tutorial_base_vs_final_shaped_load_impedance_pn",
    phase_wrapped=True,
    phase_ylim=PHASE_YLIM,
    mag_ylim=(1e-1, 1e5),
    labels=pn_labels,
    critical_freqs=CRITICAL_FREQS,
    save_dir=FIG_DIR,
)


# ============================================================
# 8. Compare load-side impedance in dq domain
# ============================================================

# The same impedance matrices can also be viewed in the dq domain.
# The dq-domain view is useful when coupling is low and phase becomes erratic.

plot_bode_matrix_overlay_3(
    f_hz=f_base,
    Zmat1=ZL_base_dq,
    Zmat2=ZL_shaped_dq,
    names=("Base case", "Final shaped case"),
    colors=("0.10", "#CC3311"),
    title="Tutorial_base_vs_final_shaped_load_impedance_dq",
    phase_wrapped=True,
    phase_ylim=PHASE_YLIM,
    mag_ylim=LOAD_MAG_YLIM,
    labels=dq_labels,
    critical_freqs=CRITICAL_FREQS,
    save_dir=FIG_DIR,
)


# ============================================================
# 9. Compare passivity index
# ============================================================

# The passivity index is calculated from the Hermitian part of the impedance:
#
#   H = (Z + Z^H) / 2
#
# The plotted value is the minimum eigenvalue of H at each frequency.
#
# Positive values indicate passive behaviour at that frequency.
# Negative values indicate non-passive behaviour.


plot_passivity_index_overlay_3(
    f_hz=f_base,                         # Frequency vector [Hz].
    Zmat1=ZL_base_dq,                    # First impedance matrix set: base case.
    Zmat2=ZL_shaped_dq,                  # Second impedance matrix set: final shaped case.
    names=("Base case", "Final shaped case"),  # Legend labels.
    colors=("0.10", "#CC3311"),          # Curve colors.
    title="Tutorial_base_vs_final_shaped_passivity",  # File name for the saved figure.
    ylabel=r'$\rho_Z$ [$\Omega$]',       # y-axis label for the passivity index.
    ylim=None,                           # Let the function choose the main y-axis limits.
    ylim_mode="full",                    # Show the full value range on the main axis.
    critical_freqs=CRITICAL_FREQS,       # Optional vertical markers at selected frequencies.
    shade_nonpassive=True,               # Shade the region below zero.
    inset_zero=True,                     # Add a zoomed inset around the zero/passivity boundary.
    inset_xlim=(5, 800),                 # Frequency range shown in the inset [Hz].
    inset_ylim=(-300, 300),              # Passivity-index range shown in the inset [ohm].
    marker=None,                         # No point markers on the curves.
    save_dir=FIG_DIR,                    # Folder where the figure is saved.
    save=True,                           # Save the figure to file.
    show=True,                           # Display the figure after saving.
    formats=("pdf",),                    # Save as PDF.
)


# ============================================================
# 10. Compare Nyquist loop-gain eigenvalues
# ============================================================

# The impedance-based stability analysis uses the source-load loop gain.
#
# For each frequency, the function plot_nyquist_eigs_compare(...) computes:
#
#   L = Z_source * inv(Z_load)
#
# and plots the eigenvalues of L in the complex plane.

plot_nyquist_eigs_compare(
    f_hz=f_base,                         # Frequency vector [Hz].
    ZL_cases=[ZL_base_dq, ZL_shaped_dq], # Load-side dq impedance matrices for each case.
    ZS_cases=[ZS_base_dq, ZS_shaped_dq], # Source-side dq impedance matrices for each case.
    names=("Base case", "Final shaped case"),  # Legend labels.
    colors=("0.10", "#CC3311"),          # Curve colors.
    title="Tutorial_base_vs_final_shaped_Nyquist",  # File name for the saved figure.
    arrow_freqs=(3, 170.0, 190.0, 270.0, 350.0),  # Frequencies where direction arrows are drawn.
    xlim=NYQ_XLIM,                       # Real-axis plot limits.
    ylim=NYQ_YLIM,                       # Imaginary-axis plot limits.
)


# ============================================================
# 11. CoF correction for diagonal dq impedance plot
# ============================================================

# The CoF method uses the full MIMO loop-gain eigenvalues to correct the
# diagonal source-side dq impedances.

cof_base = compute_cof(
    ZL_dq=ZL_base_dq,
    ZS_dq=ZS_base_dq,
    swap_branches=False,
)

ZS_base_dq_cof = cof_base["ZS_dq_corr"]

cross_info = plot_cof_diagonal(
    f_hz=f_base,
    ZL_dq=ZL_base_dq,
    ZS_dq=ZS_base_dq,
    ZS_dq_corr=ZS_base_dq_cof,
    title="Tutorial_base_case_CoF_diagonal_impedance",
    phase_wrapped=True,
    mag_ylim=(1e0, 1e5),
    phase_ylim=(-180, 180),
    label_source="Source",
    label_source_corr="CoF-corrected source",
    label_load="Load",
    critical_freqs=CRITICAL_FREQS,
    annotate=True,
    save_dir=FIG_DIR,
    save=True,
    show=True,
    formats=("pdf",),
)


# ============================================================
# 12. Signal-to-background quality check
# ============================================================

# The extraction quality can be checked by comparing the injected sideband
# response with the background sideband level from an unperturbed simulation.
#
# build_spectra_cache(...) first computes and stores the aligned sequence
# spectra for the base file and all injection files. 
#
# compute_quality_metrics_from_cache(...) then computes signal-to-background
# ratios in dB for the load-side voltage, load-side current, and source-side
# voltage sidebands.
#

FILE_BASE = "YwnoInj.out"   # Unperturbed file.
SNR_DB_THRESHOLD = 10.0

spectra_cache_base = build_spectra_cache(
    base_dir=BASE_DIR,
    runs=RUN_BASE,
    file_base=FILE_BASE,
    f1=f1,
    t_start=FIXED_T_START,
    t_end=FIXED_T_END,
    skiprows=skiprows,
    colmap=colmap,
    v_scale=v_scale,
    i_scale=i_scale,
)

qm_base = compute_quality_metrics_from_cache(
    runs=RUN_BASE,
    spectra_cache=spectra_cache_base,
    f1=f1,
    average_duplicates=AVERAGE_DUPLICATES,
)

# Build the same masks that can be used in masked impedance plots.
# Frequency points below the set threshold will then be grayed out.
# For the load side, these masks are based on load-side current quality.
# For the source side, these masks are based on source-side voltage quality.
masks_load_base, masks_source_base = build_masks(
    qm_base,
    snr_db_threshold=SNR_DB_THRESHOLD,
    use_cond=False,
)

full_matrix_mask_base = combine_masks_for_full_matrix_two_sides(
    masks_load_base,
    masks_source_base,
)

print(
    f"\nAccepted full-matrix points: "
    f"{np.count_nonzero(full_matrix_mask_base)} / {len(full_matrix_mask_base)}"
)

# Optional summary and plots.
print_sideband_quality(
    qm=qm_base,
    side="load",
    quantity="current",
    label="Base case load-side current",
    threshold_db=SNR_DB_THRESHOLD,
)

plot_sideband_quality(
    qm=qm_base,
    side="load",
    quantity="current",
    title="Tutorial_base_case_load_current_sideband_quality",
    threshold_db=SNR_DB_THRESHOLD,
    save_dir=FIG_DIR,
    save=True,
    show=True,
    formats=("pdf",),
)