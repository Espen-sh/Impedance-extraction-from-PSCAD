# -*- coding: utf-8 -*-
"""
RunExtraction.py

Simple driver for impedance extraction and stability assessment. 
It is meant for the common case: you have
PSCAD injection output files, you want the load and source dq impedance, and
you want to judge stability.

How to use
----------
1. Put this file in the same folder as Functions.py and your PSCAD .out files.
2. Set the loading parameters (colmap, scales, time window) to match your runs.
3. List the injected frequencies in the FREQUENCY SETS section. They must match
   the frequencies injected in PSCAD.
4. Point CASES at your positive and negative sequence injection files.
5. Choose what to run in ACTIVE_CASES and flip the toggles in WHAT TO RUN.
6. Run the script.

Notes
-----
- You can run a single case (one positive and negative file) or a multi case 
  (multiple positive and negative files with different injection frequencies). 
- You can also list up to three cases in ACTIVE_CASES to overlay them in the Bode
  and passivity plots. The Nyquist and CoF plots are produced one per case.
- The Bode plots can be drawn in the dq domain or the pn (modified sequence)
  domain. Set BODE_DOMAIN in WHAT TO RUN.
- Figures that save to disk are written to a Figures folder next to this file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from Functions import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# LOADING PARAMETERS
# Match these to how your PSCAD .out columns are arranged.
# ============================================================
f1 = 50.0

# Column map into the .out file. Voltages and currents at the PCC are the load
# side. dVa, dVb, dVc are the voltage drop across the source impedance.
colmap = {
    "t": 0, # Time
    "Va": 9, "Vb": 2, "Vc": 4, # Load side voltages
    "Ia": 6, "Ib": 7, "Ic": 8, # Load side currents
    "dVa": 1, "dVb": 3, "dVc": 5 # Source side voltages
}

# Alternative column layout, kept here for convenience:
# colmap = {
#     "t": 0,
#     "Va": 1, "Vb": 3, "Vc": 9,
#     "Ia": 5, "Ib": 6, "Ic": 7,
#     "dVa": 2, "dVb": 4, "dVc": 8,
# }

v_scale = 1000.0
i_scale = 1000.0
skiprows = 0

# Time window used for the FFT. A fixed window is the simplest and most
# repeatable choice. Pick a steady state interval that contains a whole number
# of fundamental periods.
T_START = 3.999
T_WIN = 6.0
T_END = T_START + T_WIN

# Average impedance values when the same frequency appears in more than one run.
AVERAGE_DUPLICATES = False

# Plot limits.
LOAD_MAG_YLIM = (1e1, 1e5)
SOURCE_MAG_YLIM = (1e-4, 1e4)
PHASE_YLIM = (-180, 180)

# Panel labels for the Bode plots, one set per domain.
DQ_LABELS = {"11": "dd", "12": "dq", "21": "qd", "22": "qq"}
PN_LABELS = {"11": "pp", "12": "pn", "21": "np", "22": "nn"}

# ============================================================
# FREQUENCY SETS
# These are the modified frequencies injected in PSCAD. Each list is one
# injection run. A case can use one list or several lists in sequence.
# Replace these with the frequencies you actually injected.
# ============================================================
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

# A short single run set, handy for a quick first extraction.
f_meas_list_quick = np.array([
    1, 3, 5, 7, 9, 11, 14, 18,
    22, 24, 28, 32, 36, 53, 59, 65,
    72, 80, 84, 88, 95, 106, 114, 123,
    130, 140, 147, 151, 155, 159, 163, 167,
    172, 176, 190, 200, 258, 272, 286, 301
], dtype=int)

# Named groups of runs. "mode" in a case picks one of these.
FREQ_SETS = {
    "quick": [f_meas_list_quick],
    "full": [f_meas_list1, 
             f_meas_list2, 
             f_meas_list3, 
             f_meas_list4, 
             f_meas_list5, 
             f_meas_list6],
}

# ============================================================
# CASE HELPERS
# ============================================================
def case_single(file_pos, file_neg, mode="quick"):
    """A case made of one positive file and one negative file."""
    return {"mode": mode, "file_pos": file_pos, "file_neg": file_neg}


def case_numbered(pos_template, neg_template, mode="full", start=1):
    """A case made of several numbered runs, for example Ypos1.out, Ypos2.out.

    Write the templates with a placeholder, for example "Ypos{i}.out", and set
    the starting index with start.
    """
    return {"mode": mode, "pos_template": pos_template,
            "neg_template": neg_template, "start": start}


def build_runs(case_cfg):
    """Turn a case into the runs dict that collect_all_runs expects."""
    f_lists = FREQ_SETS[case_cfg["mode"]]
    runs = {}

    if "file_pos" in case_cfg:
        runs["run1"] = {
            "file_pos": case_cfg["file_pos"],
            "file_neg": case_cfg["file_neg"],
            "f_list": f_lists[0],
        }
        return runs

    start = int(case_cfg.get("start", 1))
    for idx, f_list in enumerate(f_lists, start=start):
        runs[f"run{idx - start + 1}"] = {
            "file_pos": case_cfg["pos_template"].format(i=idx),
            "file_neg": case_cfg["neg_template"].format(i=idx),
            "f_list": f_list,
        }
    return runs


# ============================================================
# DEFINE AND ACTIVE CASES (EDIT THIS SECTION)
# ============================================================

"""
Mode can be "full", "quick" or self defined to select which frequency sets to use for each case. 
Setup the with this format: 
 - case_single() for single-run cases with custom file names
 - case_numbered() for multi-run cases with file names following a numbered pattern (e.g. Ypos1.out, Ypos2.out, ...). 
For case_numbered(), write as "Ypos{i}.out" and specify the starting index with "start". 
"""

# Define your cases. The key is a label used in plots and the summary.
CASES = {
    "Base case": case_numbered("Ybase_pos{i}.out", "Ybase_neg{i}.out", mode="full", start=1),
    "Shaped case": case_numbered("Ytuned_pos{i}.out", "Ytuned_neg{i}.out", mode="full", start=1),
    "Single case": case_single("Ypos.out", "Yneg.out", mode="quick"),
}

# Which cases to actually run. One case is the normal use. List up to three to
# overlay them in the Bode and passivity plots.
ACTIVE_CASES = [
    "Base case",
    "Shaped case",
    ]

# ------------------------------------------------------------
# WHAT TO RUN
# ------------------------------------------------------------
PRINT_SUMMARY = True            # Printed stability readout per case
CHECK_FREQ_COLLISIONS = True    # Pre flight check of the injected frequencies

SHOW_LOAD_BODE = True           # Bode of the load impedance
SHOW_SOURCE_BODE = True         # Bode of the source impedance
BODE_DOMAIN = "pn"              # Domain for the Bode plots: "dq" or "pn"
SHOW_NYQUIST = True             # Nyquist of the loop gain eigenvalues
SHOW_PASSIVITY = True           # Load passivity index across frequency
SHOW_MIN_DISTANCE = True        # Distance of the eigenvalues to the point -1
SHOW_COF = True                 # CoF corrected diagonal overlay

SAVE_SUMMARY_CSV = False        # Save the stability summary as a CSV file in the base directory
SUMMARY_CSV_NAME = "stability_summary.csv"


# ============================================================
# ENGINE
# ============================================================
def load_case(case_name, case_cfg):
    """Run the extraction for one case and return frequency, load and source pn."""
    runs = build_runs(case_cfg)
    return collect_all_runs(
        base_dir=BASE_DIR,
        runs=runs,
        f1=f1,
        skiprows=skiprows,
        colmap=colmap,
        v_scale=v_scale,
        i_scale=i_scale,
        average_duplicates=AVERAGE_DUPLICATES,
        use_fixed_window=True,
        fixed_t_start=T_START,
        fixed_t_end=T_END,
    )


def branch_min_distance(eigs):
    """Smallest distance of each eigenvalue branch to the point -1."""
    d1 = np.abs(1.0 + eigs[:, 0])
    d2 = np.abs(1.0 + eigs[:, 1])
    i1 = int(np.argmin(d1))
    i2 = int(np.argmin(d2))
    return (d1[i1], i1), (d2[i2], i2)


if not ACTIVE_CASES:
    raise SystemExit("ACTIVE_CASES is empty. Add at least one case to run.")

if BODE_DOMAIN not in ("dq", "pn"):
    raise SystemExit('BODE_DOMAIN must be "dq" or "pn".')

if CHECK_FREQ_COLLISIONS:
    print("\n=== Frequency collision check ===")
    for name in ACTIVE_CASES:
        runs = build_runs(CASES[name])
        for run_name, cfg in runs.items():
            print(f"\n{name} / {run_name}:")
            check_msd_multitone_list(cfg["f_list"], f1=f1)

case_data = {}
f_ref = None

for name in ACTIVE_CASES:
    print(f"\n=== Extracting case: {name} ===")
    f_all, ZL_pn, ZS_pn = load_case(name, CASES[name])

    if f_ref is None:
        f_ref = f_all
    elif len(f_all) != len(f_ref) or not np.allclose(f_all, f_ref):
        raise ValueError(
            f"Frequency vector for case '{name}' does not match the first case. "
            "Use the same frequency set for all cases you compare."
        )

    # Convert each modified sequence matrix to dq. Keep both, so the Bode plots
    # can be drawn in either domain.
    ZL_dq = np.array([Zpn_to_dq(Z) for Z in ZL_pn])
    ZS_dq = np.array([Zpn_to_dq(Z) for Z in ZS_pn])

    eigs = compute_loop_eigs_from_impedances(ZL_dq, ZS_dq)
    pidx = compute_passivity_index_from_Z(f_all, ZL_dq)

    case_data[name] = {
        "f": f_all,
        "ZL_pn": ZL_pn,
        "ZS_pn": ZS_pn,
        "ZL_dq": ZL_dq,
        "ZS_dq": ZS_dq,
        "eigs": eigs,
        "pidx": pidx,
    }

# ============================================================
# STABILITY READOUT
# ============================================================
summary_rows = []

if PRINT_SUMMARY:
    print("\n========================================")
    print(" Stability readout")
    print("========================================")
    print(
        "The generalised Nyquist criterion is the authoritative check: look at "
        "whether the eigenvalue loci encircle the point -1 in the Nyquist plot. "
        "The numbers below are practical margins."
    )

for name in ACTIVE_CASES:
    d = case_data[name]
    eigs = d["eigs"]
    f_all = d["f"]
    pidx = d["pidx"]

    (d1, i1), (d2, i2) = branch_min_distance(eigs)
    worst_dist = min(d1, d2)
    i_pass = int(np.argmin(pidx))
    min_pass = float(pidx[i_pass])

    if PRINT_SUMMARY:
        print(f"\n--- {name} ---")
        print(f"  min |1 + lam1| = {d1:.4f} @ {f_all[i1]:.1f} Hz")
        print(f"  min |1 + lam2| = {d2:.4f} @ {f_all[i2]:.1f} Hz")
        print(f"  worst distance to -1 = {worst_dist:.4f}")
        print(f"  min load passivity index = {min_pass:.4f} @ {f_all[i_pass]:.1f} Hz")
        if min_pass < 0:
            print("  load is non passive at one or more frequencies")
        # Phase and gain margin style detail from Functions.py
        summarize_eig_margins(f_all, eigs)

    summary_rows.append({
        "case": name,
        "min_dist_lam1": d1,
        "f_lam1_Hz": float(f_all[i1]),
        "min_dist_lam2": d2,
        "f_lam2_Hz": float(f_all[i2]),
        "worst_distance": worst_dist,
        "min_passivity": min_pass,
        "f_min_passivity_Hz": float(f_all[i_pass]),
    })

if SAVE_SUMMARY_CSV and summary_rows:
    csv_path = os.path.join(BASE_DIR, SUMMARY_CSV_NAME)
    headers = list(summary_rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in summary_rows:
            fh.write(",".join(str(row[h]) for h in headers) + "\n")
    print(f"\nSaved summary CSV to: {csv_path}")

# ============================================================
# PLOTS
# ============================================================
# Up to three cases can be overlaid in the Bode and passivity plots.
overlay_names = ACTIVE_CASES[:3]
if len(ACTIVE_CASES) > 3:
    print("\nNote: only the first three active cases are overlaid in the "
          "Bode and passivity plots.")


def bode_overlay(which):
    """Matrices and panel labels for the load or source Bode overlay.

    which : "load" or "source". The domain follows BODE_DOMAIN.
    Returns the list of matrices for the overlay cases and the panel labels.
    """
    if BODE_DOMAIN == "pn":
        key = "ZL_pn" if which == "load" else "ZS_pn"
        labels = PN_LABELS
    else:
        key = "ZL_dq" if which == "load" else "ZS_dq"
        labels = DQ_LABELS
    mats = [case_data[n][key] for n in overlay_names]
    return mats, labels


if SHOW_LOAD_BODE:
    mats, labels = bode_overlay("load")
    plot_bode_matrix_overlay_3(
        f_ref,
        *mats,
        names=overlay_names,
        title=f"Load {BODE_DOMAIN} impedance",
        labels=labels,
        mag_ylim=LOAD_MAG_YLIM,
        phase_ylim=PHASE_YLIM,
        save_dir=FIG_DIR,
    )

if SHOW_SOURCE_BODE:
    mats, labels = bode_overlay("source")
    plot_bode_matrix_overlay_3(
        f_ref,
        *mats,
        names=overlay_names,
        title=f"Source {BODE_DOMAIN} impedance",
        labels=labels,
        mag_ylim=SOURCE_MAG_YLIM,
        phase_ylim=PHASE_YLIM,
        save_dir=FIG_DIR,
    )

if SHOW_PASSIVITY:
    # The passivity index is defined on the dq load impedance, so it stays in dq
    # regardless of BODE_DOMAIN.
    plot_passivity_index_overlay_3(
        f_ref,
        *[case_data[n]["ZL_dq"] for n in overlay_names],
        names=overlay_names,
        title="Load passivity index",
        save_dir=FIG_DIR,
    )

if SHOW_NYQUIST:
    # plot_nyquist_eigs shows the case name in the plot title.
    for name in ACTIVE_CASES:
        d = case_data[name]
        plot_nyquist_eigs(
            d["f"],
            d["ZL_dq"],
            d["ZS_dq"],
            title=f"Nyquist of loop gain eigenvalues - {name}",
            show_freq_markers=True,
        )

if SHOW_MIN_DISTANCE:
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name in ACTIVE_CASES:
        eigs = case_data[name]["eigs"]
        axs[0].semilogx(f_ref, np.abs(1.0 + eigs[:, 0]), lw=1.8, label=name)
        axs[1].semilogx(f_ref, np.abs(1.0 + eigs[:, 1]), lw=1.8, label=name)
    axs[0].set_title(r"$|1 + \lambda_1|$")
    axs[1].set_title(r"$|1 + \lambda_2|$")
    axs[1].set_xlabel("Frequency [Hz]")
    for ax in axs:
        ax.set_ylabel("Distance to -1")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("Distance of loop eigenvalues to the point -1", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

if SHOW_COF:
    for name in ACTIVE_CASES:
        d = case_data[name]
        cof = compute_cof(d["ZL_dq"], d["ZS_dq"])
        plot_cof_diagonal(
            d["f"],
            d["ZL_dq"],
            d["ZS_dq"],
            cof["ZS_dq_corr"],
            title=f"CoF corrected diagonal overlay - {name}",
            mag_ylim=SOURCE_MAG_YLIM,
            phase_ylim=PHASE_YLIM,
            save_dir=FIG_DIR,
            show=False,
        )
        # plot_cof_diagonal writes the case name into the saved file name only,
        # so add a visible label in the top left of the on screen figure. The
        # final plt.show() then displays it.
        plt.gcf().text(0.01, 0.985, name, ha="left", va="top",
                       fontsize=12, fontweight="bold")

plt.show()