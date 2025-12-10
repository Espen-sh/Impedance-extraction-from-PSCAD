# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 19:02:31 2025

@author: Espen



Main post-processing script for the RLC test system.

- Computes the analytical modified-sequence impedance Z_pn(jω) of the
  series RLC load.
- Loads PSCAD multi-tone sweep results (positive- and negative-sequence
  injections).
- Automatically finds the best common time window (most periodic) for FFT.
- Extracts Z_pn(f) from the PSCAD simulations (via Functions.py).
- Compares analytical and measured impedances by:
    * Generating 4×2 Bode-style plots of all Z_pn entries.
    * Printing statistical error metrics (magnitude and phase).
    * Computes and plots Nyquist of loop-gain eigenvalues using analytic and measured source/load impedances.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Local functions: FFT, sequence transforms, impedance extraction, plotting, etc.
from Functions import *

# ---------------------------------------------------------------------------
# Global parameters (must be consistent with PSCAD model)
# ---------------------------------------------------------------------------

R_L = 10.0       # Load resistance [Ω]
L_L = 10e-3      # Load inductance [H]
C_L = 100e-6     # Load capacitance [F]

R_S = 1.0       # Source resistance [Ω]
L_S = 1e-3      # Source inductance [H]
C_S = 10e-6     # Source capacitance [F]

f1 = 50.0      # Fundamental frequency [Hz]

# Frequency grid for analytical curves
f_min = 0.1
f_max = 2000.0
n = 1000
f = np.logspace(np.log10(f_min), np.log10(f_max), n)
w = 2 * np.pi * f
w1 = 2 * np.pi * f1

# Base directory (so the script can be run from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Analytical modified-sequence impedance of the series RLC
# ---------------------------------------------------------------------------

# Complex frequency points s = jω
s_vec = 1j * w

#  Arrays for the 4 entries of Z_pn(jω)
Zpp_L = np.zeros_like(s_vec, dtype=complex)
Zpn_L = np.zeros_like(s_vec, dtype=complex)
Znp_L = np.zeros_like(s_vec, dtype=complex)
Znn_L = np.zeros_like(s_vec, dtype=complex)

Zpp_S = np.zeros_like(s_vec, dtype=complex)
Zpn_S = np.zeros_like(s_vec, dtype=complex)
Znp_S = np.zeros_like(s_vec, dtype=complex)
Znn_S = np.zeros_like(s_vec, dtype=complex)

for k, s in enumerate(s_vec):
    # dq-domain impedance of the RLC
    Zdq_L = Z_dq_seriesRLC(s, R = R_L, L = L_L, C = C_L)
    Zdq_S = Z_dq_seriesRLC(s, R = R_S, L = L_S, C = C_S)
    # Transform to modified-sequence domain
    Zpn_matrix_L = Z_pn_from_Zdq(Zdq_L)
    Zpn_matrix_S = Z_pn_from_Zdq(Zdq_S)

    # Store each entry as a function of frequency
    Zpp_L[k] = Zpn_matrix_L[0, 0]
    Zpn_L[k] = Zpn_matrix_L[0, 1]
    Znp_L[k] = Zpn_matrix_L[1, 0]
    Znn_L[k] = Zpn_matrix_L[1, 1]
    
    Zpp_S[k] = Zpn_matrix_S[0, 0]
    Zpn_S[k] = Zpn_matrix_S[0, 1]
    Znp_S[k] = Zpn_matrix_S[1, 0]
    Znn_S[k] = Zpn_matrix_S[1, 1]

# Convert analytic impedances to magnitude/phase
Zpp_mag_L, Zpp_phase_L = mag_phase(Zpp_L)
Zpn_mag_L, Zpn_phase_L = mag_phase(Zpn_L)
Znp_mag_L, Znp_phase_L = mag_phase(Znp_L)
Znn_mag_L, Znn_phase_L = mag_phase(Znn_L)

Zpp_mag_S, Zpp_phase_S = mag_phase(Zpp_S)
Zpn_mag_S, Zpn_phase_S = mag_phase(Zpn_S)
Znp_mag_S, Znp_phase_S = mag_phase(Znp_S)
Znn_mag_S, Znn_phase_S = mag_phase(Znn_S)

# ---------------------------------------------------------------------------
# Load PSCAD output files and perform impedance extraction
# ---------------------------------------------------------------------------

# Paths to PSCAD .out files
file_unperturbed = os.path.join(BASE_DIR, "zero_inj.out")  # No injection
file_pos         = os.path.join(BASE_DIR, "pos_inj.out")   # +seq injection
file_neg         = os.path.join(BASE_DIR, "neg_inj.out")   # -seq injection

# List of sweep frequencies present in the PSCAD injections (Hz)
f_meas_list = np.array([1, 2, 3, 4, 5, 7, 9, 12, 16, 22, 30, 40, 54, 73,
                        99, 134, 181, 244, 330, 445, 601, 812, 1000])


# ---------------------------------------------------------------------------
# Find a common, highly periodic window for FFT
# ---------------------------------------------------------------------------

T    = 1.0    # Length of FFT window [s]
step = 0.005  # Step size used when scanning candidate windows [s]

best_t_start, best_t_end, best_score = find_best_common_window(
    file_pos, file_neg, T=T, base_period=1.0, skiprows=0, step=step)

print(f"Best window: [{best_t_start:.2f}, {best_t_end:.2f}] s")
print(f"Periodicity score (lower is better): {best_score:.10f}")

# ---------------------------------------------------------------------------
# Compute dq reference angle θ1 from the unperturbed simulation
# ---------------------------------------------------------------------------

theta1 = compute_theta1_from_unperturbed(
    file_unperturbed, best_t_start, best_t_end, f1, skiprows=0)

# ---------------------------------------------------------------------------
# Extract Z_pn(f) for load and source from the perturbed simulations
# ---------------------------------------------------------------------------

Z_load_meas, Z_source_meas = Zpn_load_and_source_from_two_files_multi(
    file_pos, file_neg,
    best_t_start, best_t_end,
    f1, f_meas_list, theta1,
    skiprows=0)

# Extract entries as 1D arrays over f_meas_list
Zpp_meas_load = Z_load_meas[:, 0, 0]
Zpn_meas_load = Z_load_meas[:, 0, 1]
Znp_meas_load = Z_load_meas[:, 1, 0]
Znn_meas_load = Z_load_meas[:, 1, 1]

Zpp_meas_source = Z_source_meas[:, 0, 0]
Zpn_meas_source = Z_source_meas[:, 0, 1]
Znp_meas_source = Z_source_meas[:, 1, 0]
Znn_meas_source = Z_source_meas[:, 1, 1]

Zpp_meas_mag_load, Zpp_meas_phase_load = mag_phase(Zpp_meas_load)
Zpn_meas_mag_load, Zpn_meas_phase_load = mag_phase(Zpn_meas_load)
Znp_meas_mag_load, Znp_meas_phase_load = mag_phase(Znp_meas_load)
Znn_meas_mag_load, Znn_meas_phase_load = mag_phase(Znn_meas_load)

Zpp_meas_mag_source, Zpp_meas_phase_source = mag_phase(Zpp_meas_source)
Zpn_meas_mag_source, Zpn_meas_phase_source = mag_phase(Zpn_meas_source)
Znp_meas_mag_source, Znp_meas_phase_source = mag_phase(Znp_meas_source)
Znn_meas_mag_source, Znn_meas_phase_source = mag_phase(Znn_meas_source)

# ---------------------------------------------------------------------------
# Bode plots: analytical curves + measured points for load
# ---------------------------------------------------------------------------

charts = []
charts.append(chart_to_image(
    f, Zpp_mag_L, 'log', 'log',
    '', 'Magnitude [Ω]', '|Z_{pp}|',
    markers=[{'x': f_meas_list, 'y': Zpp_meas_mag_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Zpn_mag_L, 'log', 'log',
    ' ', '', '|Z_{pn}|',
    markers=[{'x': f_meas_list, 'y': Zpn_meas_mag_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Zpp_phase_L, 'log', 'linear',
    '', 'Phase [deg]', '∠Z_{pp}',
    markers=[{'x': f_meas_list, 'y': Zpp_meas_phase_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Zpn_phase_L, 'log', 'linear',
    '', '', '∠Z_{pn}',
    markers=[{'x': f_meas_list, 'y': Zpn_meas_phase_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znp_mag_L, 'log', 'log',
    '', 'Magnitude [Ω]', '|Z_{np}|',
    markers=[{'x': f_meas_list, 'y': Znp_meas_mag_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znn_mag_L, 'log', 'log',
    '', '', '|Z_{nn}|',
    markers=[{'x': f_meas_list, 'y': Znn_meas_mag_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znp_phase_L, 'log', 'linear',
    'Frequency [Hz]', 'Phase [deg]', '∠Z_{np}',
    markers=[{'x': f_meas_list, 'y': Znp_meas_phase_load, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znn_phase_L, 'log', 'linear',
    'Frequency [Hz]', '', '∠Z_{nn}',
    markers=[{'x': f_meas_list, 'y': Znn_meas_phase_load, 'fmt': 'x'}]
))

# Stitch the 8 panels into a 4×2 grid image (used in the report)
cell_w = max(im.width for im in charts)
cell_h = max(im.height for im in charts)
cols, rows = 2, 4

grid = Image.new("RGB", (cell_w * cols, cell_h * rows), (255, 255, 255))
for idx, im in enumerate(charts):
    r = idx // cols
    c = idx % cols
    x = c * cell_w + (cell_w - im.width) // 2
    y = r * cell_h + (cell_h - im.height) // 2
    grid.paste(im, (x, y))

plt.figure(figsize=(12, 4 * rows))
plt.imshow(grid)
plt.title('Load impedance')
plt.axis("off")
plt.tight_layout()


# ---------------------------------------------------------------------------
# Bode plots: analytical curves + measured points for source
# ---------------------------------------------------------------------------

charts = []

charts.append(chart_to_image(
    f, Zpp_mag_S, 'log', 'log',
    '', 'Magnitude [Ω]', '|Z_{pp}|',
    markers=[{'x': f_meas_list, 'y': Zpp_meas_mag_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Zpn_mag_S, 'log', 'log',
    ' ', '', '|Z_{pn}|',
    markers=[{'x': f_meas_list, 'y': Zpn_meas_mag_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Zpp_phase_S, 'log', 'linear',
    '', 'Phase [deg]', '∠Z_{pp}',
    markers=[{'x': f_meas_list, 'y': Zpp_meas_phase_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Zpn_phase_S, 'log', 'linear',
    '', '', '∠Z_{pn}',
    markers=[{'x': f_meas_list, 'y': Zpn_meas_phase_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znp_mag_S, 'log', 'log',
    '', 'Magnitude [Ω]', '|Z_{np}|',
    markers=[{'x': f_meas_list, 'y': Znp_meas_mag_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znn_mag_S, 'log', 'log',
    '', '', '|Z_{nn}|',
    markers=[{'x': f_meas_list, 'y': Znn_meas_mag_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znp_phase_S, 'log', 'linear',
    'Frequency [Hz]', 'Phase [deg]', '∠Z_{np}',
    markers=[{'x': f_meas_list, 'y': Znp_meas_phase_source, 'fmt': 'x'}]
))

charts.append(chart_to_image(
    f, Znn_phase_S, 'log', 'linear',
    'Frequency [Hz]', '', '∠Z_{nn}',
    markers=[{'x': f_meas_list, 'y': Znn_meas_phase_source, 'fmt': 'x'}]
))

# Stitch the 8 panels into a 4×2 grid image (used in the report)
cell_w = max(im.width for im in charts)
cell_h = max(im.height for im in charts)
cols, rows = 2, 4

grid = Image.new("RGB", (cell_w * cols, cell_h * rows), (255, 255, 255))
for idx, im in enumerate(charts):
    r = idx // cols
    c = idx % cols
    x = c * cell_w + (cell_w - im.width) // 2
    y = r * cell_h + (cell_h - im.height) // 2
    grid.paste(im, (x, y))

plt.figure(figsize=(12, 4 * rows))
plt.imshow(grid)
plt.title('Source impedance')
plt.axis("off")
plt.tight_layout()

# ---------------------------------------------------------------------------
# Statistical error metrics (measured vs analytic)
# ---------------------------------------------------------------------------

print("\nStatistical errors (measured – analytic):")

names = ["Zpp", "Zpn", "Znp", "Znn"]

# Measured entries as 1D arrays for load and source
Z_load_list = [Zpp_meas_load,   Zpn_meas_load,   Znp_meas_load,   Znn_meas_load]
Z_source_list = [Zpp_meas_source, Zpn_meas_source, Znp_meas_source, Znn_meas_source]

mag_threshold = 1e-6  # skip analytic points that are essentially zero magnitude


def compute_error(Z_measured_list, R_sys, L_sys, C_sys, system_name):
    """
    Compute and print statistical errors for one system (load or source).

    Z_measured_list : list [Zpp_meas, Zpn_meas, Znp_meas, Znn_meas]
    R_sys, L_sys, C_sys : RLC parameters for the analytic model
    system_name : string used in print header ("Load" or "Source")
    """
    print(f"\n=== {system_name} impedance ===")
    for name, Z_mea_vec in zip(names, Z_measured_list):
        rel_mag_err = []
        abs_mag_err = []
        phase_err   = []
        freqs_used  = []

        for i, f_m in enumerate(f_meas_list):
            # Analytic impedance at the exact measurement frequency f_m
            s_m   = 1j * 2 * np.pi * f_m
            Zdq_m = Z_dq_seriesRLC(s_m, R=R_sys, L=L_sys, C=C_sys)
            Zpn_m = Z_pn_from_Zdq(Zdq_m)

            if name == "Zpp":
                Za = Zpn_m[0, 0]
            elif name == "Zpn":
                Za = Zpn_m[0, 1]
            elif name == "Znp":
                Za = Zpn_m[1, 0]
            elif name == "Znn":
                Za = Zpn_m[1, 1]
            else:
                continue  # should never happen

            Zm = Z_mea_vec[i]

            mag_a, ph_a = mag_phase(Za)
            mag_m, ph_m = mag_phase(Zm)

            # Skip points where analytic magnitude is ~0
            if mag_a < mag_threshold:
                continue

            dmag     = mag_m - mag_a
            dmag_rel = dmag / mag_a * 100.0  # relative error in %

            dphi = ph_m - ph_a
            # Wrap phase error to [-180, 180)
            dphi = (dphi + 180) % 360 - 180

            rel_mag_err.append(dmag_rel)
            abs_mag_err.append(dmag)
            phase_err.append(dphi)
            freqs_used.append(f_m)

        rel_mag_err = np.array(rel_mag_err)
        abs_mag_err = np.array(abs_mag_err)
        phase_err   = np.array(phase_err)
        freqs_used  = np.array(freqs_used)

        if len(rel_mag_err) == 0:
            print(f"\n{name}: no valid points (analytic magnitude ~0 everywhere).")
            continue

        print(f"\n{name}:")
        print(
            "  |Z| relative error (%): "
            f"mean|·|={np.mean(np.abs(rel_mag_err)):.6f}, "
            f"median|·|={np.median(np.abs(rel_mag_err)):.6f}, "
            f"max|·|={np.max(np.abs(rel_mag_err)):.6f}"
        )
        print(
            "  phase error (deg):       "
            f"mean={np.mean(phase_err):+.6f}, "
            f"std={np.std(phase_err):.6f}, "
            f"max|·|={np.max(np.abs(phase_err)):.6f}"
        )


# Use R,L,C of the load for the load block
compute_error(Z_measured_list = Z_load_list, R_sys = R_L, 
                    L_sys = L_L, C_sys = C_L, system_name="Load")

# Use R,L,C of the source for the source block
compute_error(Z_measured_list=Z_source_list, R_sys = R_S,
                    L_sys = L_S, C_sys = C_S, system_name="Source")



# ---------------------------------------------------------------------------
# Nyquist of loop-gain eigenvalues
# ---------------------------------------------------------------------------

eig1 = np.zeros(n, dtype=complex)
eig2 = np.zeros(n, dtype=complex)

for k in range(n):
    # Assemble analytic load and source Z_pn matrices at f[k]
    ZL = np.array([[Zpp_L[k], Zpn_L[k]],
                   [Znp_L[k], Znn_L[k]]], dtype=complex)
    ZS = np.array([[Zpp_S[k], Zpn_S[k]],
                   [Znp_S[k], Znn_S[k]]], dtype=complex)

    YL  = np.linalg.inv(ZL)
    Lpn = ZS @ YL

    vals = np.linalg.eigvals(Lpn)
    eig1[k], eig2[k] = vals[0], vals[1]

# Measured loop-gain eigenvalues at the discrete sweep frequencies
Nf = len(f_meas_list)
eig1_meas = np.zeros(Nf, dtype=complex)
eig2_meas = np.zeros(Nf, dtype=complex)

for i in range(Nf):
    ZL_m = Z_load_meas[i, :, :]
    ZS_m = Z_source_meas[i, :, :]

    YL_m  = np.linalg.inv(ZL_m)
    Lpn_m = ZS_m @ YL_m

    vals = np.linalg.eigvals(Lpn_m)
    eig1_meas[i], eig2_meas[i] = vals[0], vals[1]

fig, ax = plt.subplots(figsize=(6, 6))   # square Nyquist

# Analytic curves
ax.plot(eig1.real, eig1.imag, label=r'$\lambda_1$ (analytic)',
        color='tab:blue', linewidth=1.6)
ax.plot(eig2.real, eig2.imag, label=r'$\lambda_2$ (analytic)',
        color='tab:orange', linewidth=1.6)

# Measured points
ax.plot(eig1_meas.real, eig1_meas.imag, 'kx',
        label=r'$\lambda_1$ (measured)', markersize=7, markeredgewidth=1.4)
ax.plot(eig2_meas.real, eig2_meas.imag, 'k+',
        label=r'$\lambda_2$ (measured)', markersize=7, markeredgewidth=1.4)

# -1 point (stability reference)
ax.plot(-1, 0, 'r+', markersize=15, markeredgewidth=2.0, label='-1')
ax.annotate(r'$-1$', xy=(-1, 0), xytext=(6, 6),
            textcoords='offset points', color='red', fontsize=15)

# Axes lines
ax.axhline(0, color='grey', linewidth=0.6)
ax.axvline(0, color='grey', linewidth=0.6)



# Helper to mark specific frequencies on λ1
def annotate_freq_point(ax, f_mark, f_array, eig_array, text, color='k'):
    idx = np.argmin(np.abs(f_array - f_mark))
    lam = eig_array[idx]
    ax.plot(lam.real, lam.imag, 'o', color=color,
            markersize=6, markeredgewidth=1.2)
    ax.annotate(text, xy=(lam.real, lam.imag), xytext=(6, 6),
                textcoords='offset points', color=color, fontsize=15)

# Mark 1, 50, 1000 Hz on λ1
for f_mark, col in [(1.0, 'tab:green'),
                    (50.0, 'tab:purple'),
                    (1000.0, 'tab:red')]:
    annotate_freq_point(ax, f_mark, f, eig1,
                        text=f'{int(f_mark)} Hz', color=col)

ax.set_xlabel(r'$\mathrm{Re}\{\lambda\}$', fontsize=20)
ax.set_ylabel(r'$\mathrm{Im}\{\lambda\}$', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.grid(True, linewidth=0.6, alpha=0.5)
ax.legend(loc='upper left', fontsize=25, framealpha=0.9)
ax.set_aspect('equal', adjustable='box')

fig.tight_layout()
plt.show()



