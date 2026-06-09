# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 09:44:22 2026

@author: espen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ---- Parameters ----
f1 = 50.0
f_min = 0.1
f_max = 2000.0
n = 1000
f = np.logspace(np.log10(f_min), np.log10(f_max), n)
w = 2*np.pi*f
w1 = 2*np.pi*f1


# ============================================================
# Analytic impedance
# ============================================================

def Z_dq_seriesRLC(s, R, L, C):
    """
    Return the 2×2 dq-domain impedance matrix of a symmetric series RLC branch.

    Parameters
    ----------
    s : complex or array_like
        Laplace variable s = jω at which the impedance is evaluated.
    R : float
        Phase resistance [Ω].
    L : float
        Phase inductance [H].
    C : float
        Phase capacitance [F].

    Returns
    -------
    Z_dq : (2, 2) complex ndarray
        dq-domain impedance matrix of the RLC branch.
    """
    return np.array(
        [[R + L*s + s / (C*s**2 + C*w1**2), -w1 * L + w1 / (C*s**2 + C*w1**2)],
         [w1 * L - w1 / (C*s**2 + C*w1**2), R + s * L + s / (C*s**2 + C*w1**2)]],
        dtype=complex)


def Z_pn_from_Zdq(Zdq):
    AZ = (1/np.sqrt(2)) * np.array([[1,  1j],
                                    [1, -1j]], dtype=complex)

    AZ_inv = (1/np.sqrt(2)) * np.array([[1,   1],
                                        [-1j, 1j]], dtype=complex)

    return AZ @ Zdq @ AZ_inv



def Z_vsc_ohm(s: complex, p: dict) -> np.ndarray:
    w0 = p["w0"]; R = p["R_pu"]; X = p["X_pu"]; Zb = p["Z_base"]
    kp_i = p["kp_i"]; ki_i = p["ki_i"]; Kd = p.get("Kd", X)
    kp_pll = p["kp_pll"]; ki_pll = p["ki_pll"]
    Vd0 = p["Vd0"]
    Id0 = p["Id0"]; Iq0 = p["Iq0"]
    Cd0 = p["Cd0"]; Cq0 = p["Cq0"]

    Ti = p.get("Ti", 0.0)
    Tv = p.get("Tv", 0.0)

    Hi_meas = 1.0/(1.0 + Ti*s) if Ti else (1.0 + 0j)
    Hv_meas = 1.0/(1.0 + Tv*s) if Tv else (1.0 + 0j)

    I2 = np.eye(2, dtype=complex)
    J  = np.array([[0, -1], [1, 0]], dtype=complex)

    Zout = (R + (s/w0)*X)*I2 + X*J

    Hcc  = kp_i + ki_i/s
    Gcc  = Hcc*I2
    Gdec = np.array([[0, -Kd], [Kd, 0]], dtype=complex)
    F = (Gcc - Gdec) * Hi_meas

    Hpll = kp_pll + ki_pll/s
    Tpll = Hpll / (s + Vd0*Hpll) * Hv_meas

    GiPLL = Tpll * np.array([[0,  Iq0],
                             [0, -Id0]], dtype=complex)
    GdPLL = Tpll * np.array([[0, -Cq0],
                             [0,  Cd0]], dtype=complex)

    Zdq_pu = np.linalg.inv(I2 - F @ GiPLL - GdPLL) @ (Zout + F)
    return Zdq_pu * Zb



def Zpn_to_dq(Zpn):
    
    AZ = (1/np.sqrt(2)) * np.array([[1,  1j],
                                    [1, -1j]], dtype=complex)

    AZ_inv = (1/np.sqrt(2)) * np.array([[1,   1],
                                        [-1j, 1j]], dtype=complex)

    return AZ_inv @ Zpn @ AZ


# ============================================================
# Base impedance extraction functionality
# ============================================================

def mag_phase(Z, phase_wrapped=True):
    """
    Convert a complex quantity to magnitude and phase.

    Parameters
    ----------
    Z : complex or array_like of complex
    phase_wrapped : bool
        If True, return phase wrapped to [-180, 180).
        If False, return unwrapped phase in degrees.

    Returns
    -------
    mag : ndarray
        Magnitude |Z|.
    phase_deg : ndarray
        Phase in degrees.
    """
    Z = np.asarray(Z)
    mag = np.abs(Z)

    if phase_wrapped:
        phase_deg = np.angle(Z, deg=True)
        phase_deg = (phase_deg + 180) % 360 - 180
    else:
        phase_deg = np.rad2deg(np.unwrap(np.angle(Z)))

    return mag, phase_deg



def check_msd_multitone_list(f_list, f1=50.0, tol=1e-9):
    """
    Check collisions for modified sequence domain extraction.

    For each modified frequency fm:
        upper bin  = f1 + fm
        mirror bin = abs(f1 - fm)
    """

    f = np.asarray(f_list, dtype=float)

    same_mirror = []
    cross_coll = []
    special = []

    # Single-frequency special cases
    for fm in f:
        if abs(fm - 0.0) < tol:
            special.append((fm, "hits 50 Hz fundamental"))
        if abs(fm - f1) < tol:
            special.append((fm, "hits DC mirror bin"))
        if abs(fm - 2*f1) < tol:
            special.append((fm, "mirror bin lands on 50 Hz fundamental"))

    # Pairwise collisions
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            fi, fj = f[i], f[j]

            # |f1-fi| = |f1-fj|
            if abs(abs(f1 - fi) - abs(f1 - fj)) < tol:
                same_mirror.append((fi, fj))

            # f1+fi = |f1-fj|  OR  f1+fj = |f1-fi|
            if abs((f1 + fi) - abs(f1 - fj)) < tol or \
               abs((f1 + fj) - abs(f1 - fi)) < tol:
                cross_coll.append((fi, fj))

    print("Pairs with same mirror bin:", same_mirror if same_mirror else "None")
    print("Pairs with upper/mirror overlap:", cross_coll if cross_coll else "None")
    print("Special single-tone issues:", special if special else "None")



def find_bin(freqs_hz: np.ndarray, f_target_hz: float) -> int:
    """Finds index of FFT bin closest to f_target_hz."""
    return int(np.argmin(np.abs(freqs_hz - f_target_hz)))



def fft_three_phase(t: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    FFT of three-phase signals.
    """
    t = np.asarray(t)
    a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
    N = len(t)

    dt = t[1] - t[0]

    # Remove DC to reduce leakage
    a = a - np.mean(a)
    b = b - np.mean(b)
    c = c - np.mean(c)

    A = np.fft.fft(a) / N
    B = np.fft.fft(b) / N
    C = np.fft.fft(c) / N
    freqs = np.fft.fftfreq(N, d=dt)
    return freqs, A, B, C



def sym_components(Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray):
    """Symmetrical components (0, +, -)."""
    a = np.exp(1j * 2*np.pi/3)
    T = (1/3) * np.array([[1, 1,    1   ],
                          [1, a,    a**2],
                          [1, a**2, a   ]], dtype=complex)
    stacked = np.vstack([Va, Vb, Vc])
    V0, Vp, Vn = T @ stacked
    return V0, Vp, Vn



def load_pscad_segment(path: str,
                       t_start: float,
                       t_end: float,
                       skiprows: int = 0,
                       colmap: dict | None = None,
                       v_scale: float = 1000.0,
                       i_scale: float = 1000.0,
                       dv_scale: float | None = None):
    """
    Loads a window [t_start, t_end] from PSCAD .out and returns:
      t, Va,Vb,Vc, Ia,Ib,Ic, dVa,dVb,dVc

    - Va,Vb,Vc and Ia,Ib,Ic are "load-side" (PCC) quantities.
    - dVa,dVb,dVc are "source-side ΔV" across source impedance.
      If not present in colmap, they are returned as None.

    Colmap needs to match columns in INFX file.
    """
    if colmap is None:
        colmap = {"t": 0, "Va": 1, "Vb": 2, "Vc": 3, "Ia": 4, "Ib": 5, "Ic": 6,
                  "dVa": 7, "dVb": 8, "dVc": 9} 

    data = np.loadtxt(path, skiprows=skiprows)

    t  = data[:, colmap["t"]]
    Va = data[:, colmap["Va"]] * v_scale
    Vb = data[:, colmap["Vb"]] * v_scale
    Vc = data[:, colmap["Vc"]] * v_scale
    Ia = data[:, colmap["Ia"]] * i_scale
    Ib = data[:, colmap["Ib"]] * i_scale
    Ic = data[:, colmap["Ic"]] * i_scale
    

    if dv_scale is None:
        dv_scale = v_scale

    has_dv = all(k in colmap for k in ("dVa", "dVb", "dVc"))
    dVa = data[:, colmap["dVa"]] * dv_scale if has_dv else None
    dVb = data[:, colmap["dVb"]] * dv_scale if has_dv else None
    dVc = data[:, colmap["dVc"]] * dv_scale if has_dv else None

    mask = (t >= t_start) & (t < t_end)

    t_seg = t[mask] - t[mask][0]
    Va_seg, Vb_seg, Vc_seg = Va[mask], Vb[mask], Vc[mask]
    Ia_seg, Ib_seg, Ic_seg = Ia[mask], Ib[mask], Ic[mask]

    dVa_seg = dVa[mask] if has_dv else None
    dVb_seg = dVb[mask] if has_dv else None
    dVc_seg = dVc[mask] if has_dv else None

    return t_seg, Va_seg, Vb_seg, Vc_seg, Ia_seg, Ib_seg, Ic_seg, dVa_seg, dVb_seg, dVc_seg



def theta_from_file(path: str,
                    t_start: float,
                    t_end: float,
                    f1: float,
                    skiprows: int = 0,
                    colmap: dict | None = None,
                    v_scale: float = 1000.0,
                    i_scale: float = 1000.0) -> float:
    """
    theta = angle{ Vp(f1) } computed from this file and this window.
    """
    t, Va, Vb, Vc, *_ = load_pscad_segment(path, t_start, t_end,
                                          skiprows=skiprows, colmap=colmap,
                                          v_scale=v_scale, i_scale=i_scale)
    freqs, Va_f, Vb_f, Vc_f = fft_three_phase(t, Va, Vb, Vc)
    _, Vp_f, _ = sym_components(Va_f, Vb_f, Vc_f)
    k1 = find_bin(freqs, f1)
    return float(np.angle(Vp_f[k1]))



def aligned_seq_spectra(path: str,
                        t_start: float,
                        t_end: float,
                        theta: float,
                        which_voltage: str = "load",   # "load" or "source"
                        skiprows: int = 0,
                        colmap: dict | None = None,
                        v_scale: float = 1000.0,
                        i_scale: float = 1000.0):
    """
    Returns aligned spectra dict:
      Vp_al = Vp * exp(-j theta)
      Ip_al = Ip * exp(-j theta)
      Vn_al = Vn * exp(+j theta)
      In_al = In * exp(+j theta)

    which_voltage:
      - "load": uses Va,Vb,Vc
      - "source": uses dVa,dVb,dVc (requires those columns)
    """
    t, Va, Vb, Vc, Ia, Ib, Ic, dVa, dVb, dVc = load_pscad_segment(
        path, t_start, t_end, skiprows=skiprows, colmap=colmap,
        v_scale=v_scale, i_scale=i_scale
    )
    
    if which_voltage == "load":
        vA, vB, vC = Va, Vb, Vc
    elif which_voltage == "source":
        if dVa is None:
            raise ValueError("No source ΔV channels. Provide dVa/dVb/dVc in colmap.")
        vA, vB, vC = dVa, dVb, dVc
    else:
        raise ValueError("which_voltage must be 'load' or 'source'.")

    freqs, vA_f, vB_f, vC_f = fft_three_phase(t, vA, vB, vC)
    _,     Ia_f, Ib_f, Ic_f = fft_three_phase(t, Ia, Ib, Ic)

    _, Vp_f, Vn_f = sym_components(vA_f, vB_f, vC_f)
    _, Ip_f, In_f = sym_components(Ia_f, Ib_f, Ic_f)

    rot_p = np.exp(-1j * theta)
    rot_n = np.exp(+1j * theta)

    return {
        "freqs": freqs,
        "Vp_al": Vp_f * rot_p,
        "Ip_al": Ip_f * rot_p,
        "Vn_al": Vn_f * rot_n,
        "In_al": In_f * rot_n,
    }



def extract_sidebands_from_spectra(spectra, f1, f_inj_list):
    """
    Extracts sideband magnitudes corrisponding to injected frequencies.
    """
    freqs = spectra["freqs"]
    Vp_al, Ip_al = spectra["Vp_al"], spectra["Ip_al"]
    Vn_al, In_al = spectra["Vn_al"], spectra["In_al"]

    f_inj_list = np.atleast_1d(f_inj_list).astype(float)
    Nf = len(f_inj_list)

    Vp = np.zeros(Nf, dtype=complex)
    Ip = np.zeros(Nf, dtype=complex)
    Vn = np.zeros(Nf, dtype=complex)
    In = np.zeros(Nf, dtype=complex)

    for i, f_inj in enumerate(f_inj_list):
        # p-sideband: f1 + f
        k_p = find_bin(freqs, f1 + f_inj)
        Vp[i] = Vp_al[k_p]
        Ip[i] = Ip_al[k_p]

        if f_inj >= f1:
            # n-sideband: f - f1 (if positive)
            k_n = find_bin(freqs, f_inj - f1)
            Vn[i] = Vn_al[k_n]
            In[i] = In_al[k_n]
        else:
            # n-sideband if negative f - f1 -> use +seq at (f1 - f) and conjugate
            k_n = find_bin(freqs, f1 - f_inj)
            Vn[i] = np.conj(Vp_al[k_n])
            In[i] = np.conj(Ip_al[k_n])
            

    return {"f_inj": f_inj_list, "Vp": Vp, "Ip": Ip, "Vn": Vn, "In": In}



def Zpn_load_and_source_from_two_files(file_pos_inj: str,
                                             file_neg_inj: str,
                                             t_start: float,
                                             t_end: float,
                                             f1: float,
                                             f_sweep_list: np.ndarray,
                                             skiprows: int = 0,
                                             colmap: dict | None = None,
                                             v_scale: float = 1000.0,
                                             i_scale: float = 1000.0):
    """
    Identify Z_pn (2x2 modified-sequence impedance matrix) using:
      - one +seq injection file
      - one -seq injection file

    Returns:
      ZL_all: (Nf,2,2) load-side modified-sequence impedance
      ZS_all: (Nf,2,2) source-side modified-sequence impedance (ΔV), or zeros if ΔV not available
    """

    #Find phase angle from fundamental frequency
    theta_pos = theta_from_file(file_pos_inj, t_start, t_end, f1,
                                skiprows=skiprows, colmap=colmap,
                                v_scale=v_scale, i_scale=i_scale)
    theta_neg = theta_from_file(file_neg_inj, t_start, t_end, f1,
                                skiprows=skiprows, colmap=colmap,
                                v_scale=v_scale, i_scale=i_scale)

    # Full load spectra aligned with fundamental angle
    spec_pos_L = aligned_seq_spectra(file_pos_inj, t_start, t_end, theta_pos,
                                     which_voltage="load",
                                     skiprows=skiprows, colmap=colmap,
                                     v_scale=v_scale, i_scale=i_scale)
    spec_neg_L = aligned_seq_spectra(file_neg_inj, t_start, t_end, theta_neg,
                                     which_voltage="load",
                                     skiprows=skiprows, colmap=colmap,
                                     v_scale=v_scale, i_scale=i_scale)
    
    
    # Find sidebands corrisponding to injected frequencies
    sb_pos_L = extract_sidebands_from_spectra(spec_pos_L, f1, f_sweep_list)
    sb_neg_L = extract_sidebands_from_spectra(spec_neg_L, f1, f_sweep_list)
    
    have_source = True
    # Same sequence for source side
    try:
        spec_pos_S = aligned_seq_spectra(file_pos_inj, t_start, t_end, theta_pos,
                                         which_voltage="source",
                                         skiprows=skiprows, colmap=colmap,
                                         v_scale=v_scale, i_scale=i_scale)
        spec_neg_S = aligned_seq_spectra(file_neg_inj, t_start, t_end, theta_neg,
                                         which_voltage="source",
                                         skiprows=skiprows, colmap=colmap,
                                         v_scale=v_scale, i_scale=i_scale)
        
        sb_pos_S = extract_sidebands_from_spectra(spec_pos_S, f1, f_sweep_list)
        sb_neg_S = extract_sidebands_from_spectra(spec_neg_S, f1, f_sweep_list)

    except ValueError:
        have_source = False
        

    f_sweep_list = np.atleast_1d(f_sweep_list)
    Nf = len(f_sweep_list)

    ZL_all = np.zeros((Nf, 2, 2), dtype=complex)
    ZS_all = np.zeros((Nf, 2, 2), dtype=complex)

    for i in range(Nf):
        # LOAD: build matrices and solve Z = V * I^{-1}
        Vp1, Ip1 = sb_pos_L["Vp"][i], sb_pos_L["Ip"][i]
        Vn1, In1 = sb_pos_L["Vn"][i], sb_pos_L["In"][i]
        Vp2, Ip2 = sb_neg_L["Vp"][i], sb_neg_L["Ip"][i]
        Vn2, In2 = sb_neg_L["Vn"][i], sb_neg_L["In"][i]

        Vmat = np.array([[Vp1, Vp2],
                         [Vn1, Vn2]], dtype=complex)
        Imat = np.array([[Ip1, Ip2],
                         [In1, In2]], dtype=complex)
        
        # Solve instead of inv
        ZL_all[i] = np.linalg.solve(Imat.T, Vmat.T).T
        
        # SOURCE
        if have_source:
            Vp1, Ip1 = sb_pos_S["Vp"][i], sb_pos_S["Ip"][i]
            Vn1, In1 = sb_pos_S["Vn"][i], sb_pos_S["In"][i]
            Vp2, Ip2 = sb_neg_S["Vp"][i], sb_neg_S["Ip"][i]
            Vn2, In2 = sb_neg_S["Vn"][i], sb_neg_S["In"][i]

            VmatS = np.array([[Vp1, Vp2],
                              [Vn1, Vn2]], dtype=complex)
            ImatS = -np.array([[Ip1, Ip2], # Defined the other way compared to load
                              [In1, In2]], dtype=complex) 

            ZS_all[i] = np.linalg.solve(ImatS.T, VmatS.T).T

    return ZL_all, ZS_all



def compute_error(
    f_meas_list,
    Z_meas_mat,                       # (Nf,2,2) complex measured
    f_ana,                            # (N,) analytic frequency vector
    Zpp_ana, Zpn_ana, Znp_ana, Znn_ana,  # (N,) analytic complex arrays on f_ana
    mag_threshold=1e-6,
    name="",
    denom="meas",                     # <-- "meas" (prefered) or "ana"
):
    """
    Compare measured vs analytic impedance matrices.

    """
    f_meas = np.asarray(f_meas_list, float).copy()
    f_ana = np.asarray(f_ana, float).copy()
    Z_meas = np.asarray(Z_meas_mat, complex)
    assert Z_meas.ndim == 3 and Z_meas.shape[1:] == (2, 2), "Z_meas_mat must be (Nf,2,2)"

    # ---- Sort measured frequencies
    idx = np.argsort(f_meas)
    f_meas = f_meas[idx]
    Z_meas = Z_meas[idx]

    # ---- Ensure analytic frequencies are sorted too
    ia = np.argsort(f_ana)
    f_ana = f_ana[ia]
    Zpp_ana = np.asarray(Zpp_ana, complex)[ia]
    Zpn_ana = np.asarray(Zpn_ana, complex)[ia]
    Znp_ana = np.asarray(Znp_ana, complex)[ia]
    Znn_ana = np.asarray(Znn_ana, complex)[ia]

    # ---- Complex interpolation helper
    def cinterp(y):
        y = np.asarray(y, complex)
        re = np.interp(f_meas, f_ana, np.real(y))
        im = np.interp(f_meas, f_ana, np.imag(y))
        return re + 1j * im

    # Build analytic matrix at measured freqs
    Z_ana = np.zeros_like(Z_meas)
    Z_ana[:, 0, 0] = cinterp(Zpp_ana)
    Z_ana[:, 0, 1] = cinterp(Zpn_ana)
    Z_ana[:, 1, 0] = cinterp(Znp_ana)
    Z_ana[:, 1, 1] = cinterp(Znn_ana)

    # ---- Matrix Frobenius errors
    fro_abs = np.linalg.norm(Z_meas - Z_ana, ord="fro", axis=(1, 2))

    if denom.lower() == "meas":
        fro_den = np.linalg.norm(Z_meas, ord="fro", axis=(1, 2))
    elif denom.lower() == "ana":
        fro_den = np.linalg.norm(Z_ana, ord="fro", axis=(1, 2))
    else:
        raise ValueError("denom must be 'meas' or 'ana'")

    fro_rel_pct = np.full_like(fro_abs, np.nan, dtype=float)
    validF = np.isfinite(fro_den) & (fro_den >= mag_threshold)
    fro_rel_pct[validF] = 100.0 * fro_abs[validF] / fro_den[validF]

    # ---- Elementwise errors
    def wrap_deg(d):
        return (d + 180.0) % 360.0 - 180.0

    def element_errors(Zm, Za, Zden_elem):
        # magnitude + phase
        mag_m = np.abs(Zm)
        mag_a = np.abs(Za)
        ph_m = np.degrees(np.angle(Zm))
        ph_a = np.degrees(np.angle(Za))

        dmag_abs = mag_m - mag_a

        dmag_rel = np.full_like(dmag_abs, np.nan, dtype=float)
        valid = np.isfinite(Zden_elem) & (Zden_elem >= mag_threshold)
        dmag_rel[valid] = 100.0 * dmag_abs[valid] / Zden_elem[valid]

        dphi = wrap_deg(ph_m - ph_a)
        dphi[~valid] = np.nan
        dmag_abs[~valid] = np.nan
        return dmag_abs, dmag_rel, dphi

    # choose per-element denom for relative magnitude error
    if denom.lower() == "meas":
        Zden_mat = np.abs(Z_meas)
    else:
        Zden_mat = np.abs(Z_ana)

    out = {
        "name": name,
        "denom": denom.lower(),
        "f_meas": f_meas,
        "fro_abs_ohm": fro_abs,
        "fro_rel_pct": fro_rel_pct,
        "Z_ana_mat": Z_ana,
    }

    names = {"Zpp": (0, 0), "Zpn": (0, 1), "Znp": (1, 0), "Znn": (1, 1)}
    for key, (r, c) in names.items():
        Zm = Z_meas[:, r, c]
        Za = Z_ana[:, r, c]
        Zden_elem = Zden_mat[:, r, c]
        dmag_abs, dmag_rel, dphi = element_errors(Zm, Za, Zden_elem)
        out[key] = {
            "dmag_abs_ohm": dmag_abs,
            "dmag_rel_pct": dmag_rel,
            "dphi_deg": dphi,
        }

    # ---- Summary print
    def summarize(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        return dict(
            mean=float(np.mean(x)),
            std=float(np.std(x)),
            mean_abs=float(np.mean(np.abs(x))),
            median_abs=float(np.median(np.abs(x))),
            p95_abs=float(np.percentile(np.abs(x), 95)),
            max_abs=float(np.max(np.abs(x))),
            n=int(x.size),
        )

    print(f"\n=== Error metrics: {name} ===")
    print(f"Denominator for relative errors: ||Z_{out['denom']}||")
    sF = summarize(fro_rel_pct)
    if sF:
        print(f"Frobenius rel err (%): mean|·|={sF['mean_abs']:.3f}, p95|·|={sF['p95_abs']:.3f}, max|·|={sF['max_abs']:.3f} (n={sF['n']})")
    else:
        print("Frobenius rel err (%): no valid points")

    for key in ["Zpp", "Zpn", "Znp", "Znn"]:
        sM = summarize(out[key]["dmag_rel_pct"])
        sP = summarize(out[key]["dphi_deg"])
        if not sM:
            print(f"{key}: no valid points (|Z_denom| < threshold)")
            continue
        print(f"\n{key}:")
        print(f"  |Z| rel err (%): mean|·|={sM['mean_abs']:.3f}, median|·|={sM['median_abs']:.3f}, p95|·|={sM['p95_abs']:.3f}, max|·|={sM['max_abs']:.3f}")
        if sP:
            print(f"  phase err (deg): mean={sP['mean']:+.2f}, std={sP['std']:.2f}, mean|·|={sP['mean_abs']:.2f}, max|·|={sP['max_abs']:.2f}")

    return out



def window_periodicity_score_norm(t, signals, base_period=1.0):
    """
    Calculates a periodicity score
    """
    t = np.asarray(t)
    dt = t[1] - t[0]
    N_period = int(round(base_period / dt))
    if len(t) < 2 * N_period:
        raise ValueError("Window too short for two base_period segments.")

    start0, end0 = 0, N_period
    end1 = len(t)
    start1 = end1 - N_period

    scores = []
    for x in signals:
        x = np.asarray(x)
        x0 = x[start0:end0]
        x1 = x[start1:end1]
        diff_rms = np.sqrt(np.mean((x1 - x0)**2))
        x_rms = np.sqrt(np.mean(x0**2)) + 1e-12
        scores.append(diff_rms / x_rms)

    return float(np.mean(scores))



def find_best_common_window(file_pos, file_neg,
                            T=1.0, base_period=1.0,
                            skiprows=0, step=0.005,
                            colmap=None, v_scale=1000.0, i_scale=1000.0):
    
    """
    Finds the time window that is most periodic
    """

    tp, Vap, Vbp, Vcp, Iap, Ibp, Icp, *_ = load_pscad_segment(
        file_pos, 0.0, 1e9, skiprows=skiprows, colmap=colmap, v_scale=v_scale, i_scale=i_scale)
    tn, Van, Vbn, Vcn, Ian, Ibn, Icn, *_ = load_pscad_segment(
        file_neg, 0.0, 1e9, skiprows=skiprows, colmap=colmap, v_scale=v_scale, i_scale=i_scale)
    
    t = tp
    dt = t[1] - t[0]
    
    #-------------------------------#
    t_min = 0.2 #float(t[0])
    #-------------------------------#
    
    t_max = float(t[-1])
    last_start = t_max - T - dt
    t_candidates = np.arange(t_min, last_start + 0.5*step, step)

    effective_bp = min(base_period, T/2.0 - 1e-9)
    if effective_bp <= 0:
        raise ValueError("Window T too small.")

    best_score = np.inf
    best_t0 = None

    for t0 in t_candidates:
        t1 = t0 + T
        mask = (t >= t0) & (t <= t1)
        if np.count_nonzero(mask) < 20:
            continue

        t_win = t[mask]
        try:
            score_p = window_periodicity_score_norm(
                t_win, [Vap[mask], Vbp[mask], Vcp[mask], Iap[mask], Ibp[mask], Icp[mask]],
                base_period=effective_bp
            )
            score_n = window_periodicity_score_norm(
                t_win, [Van[mask], Vbn[mask], Vcn[mask], Ian[mask], Ibn[mask], Icn[mask]],
                base_period=effective_bp
            )
        except ValueError:
            continue

        score = 0.5*(score_p + score_n)
        if score < best_score:
            best_score = score
            best_t0 = t0

    if best_t0 is None:
        raise RuntimeError("No valid window found.")
    return best_t0, best_t0 + T, best_score



# ============================================================
# Multirun helpers
# ============================================================

def _average_duplicates_by_freq(f, Z):
    """
    Average values over identical frequencies.
    Works for 1D arrays and (N,2,2) arrays.
    """
    f = np.asarray(f, float)

    if Z.ndim == 1:
        Z = Z[:, None]

    out_f = []
    out_Z = []

    i = 0
    while i < len(f):
        j = i + 1
        while j < len(f) and f[j] == f[i]:
            j += 1
        out_f.append(f[i])
        out_Z.append(np.mean(Z[i:j], axis=0))
        i = j

    out_f = np.array(out_f, float)
    out_Z = np.stack(out_Z, axis=0)

    if out_Z.shape[1] == 1:
        out_Z = out_Z[:, 0]

    return out_f, out_Z



def angle_error_to_neg_real_deg(z):
    """
    Smallest angular error [deg] between z and the negative real axis (-180 deg).
    """
    return float(
        np.degrees(
            abs(np.arctan2(np.sin(np.angle(z) - np.pi),
                           np.cos(np.angle(z) - np.pi)))
        ))



def compute_loop_eigs_from_impedances(ZL_list, ZS_list):
    """
    Loop gain L = ZS * inv(ZL). Return eigenvalues with continuity tracking.
    """
    N = ZL_list.shape[0]
    eigs = np.zeros((N, 2), dtype=complex)
    prev = None

    for k in range(N):
        M = np.linalg.solve(ZL_list[k].T, ZS_list[k].T).T
        vals = np.linalg.eigvals(M)

        if prev is None:
            order = np.lexsort((vals.imag, vals.real))
            vals = vals[order]
        else:
            cost_keep = abs(vals[0] - prev[0]) + abs(vals[1] - prev[1])
            cost_swap = abs(vals[1] - prev[0]) + abs(vals[0] - prev[1])
            if cost_swap < cost_keep:
                vals = vals[::-1]

        eigs[k, :] = vals
        prev = eigs[k, :]

    return eigs



def collect_all_runs(
    base_dir,
    runs,
    f1,
    skiprows=0,
    colmap=None,
    v_scale=1000.0,
    i_scale=1000.0,
    average_duplicates=True,
    use_fixed_window=True,
    fixed_t_start=None,
    fixed_t_end=None,
    T_win=1.0,
    base_period=1.0,
    step=0.005
):
    """
    Load all runs in `runs`, identify load/source impedance, and return
    concatenated arrays sorted by frequency.

    Parameters
    ----------
    base_dir : str
        Directory containing the PSCAD output files.
    runs : dict
        Dict like:
            {"run1": {"file_pos": "...", "file_neg": "...", "f_list": np.array([...])}, ...}
    f1 : float
        Fundamental frequency [Hz].
    skiprows, colmap, v_scale, i_scale :
        Passed to PSCAD-loading helpers.
    average_duplicates : bool
        If True, average repeated frequencies after concatenation.
    use_fixed_window : bool
        If True, use [fixed_t_start, fixed_t_end].
        Otherwise find the best common window from the run pair.
    fixed_t_start, fixed_t_end : float or None
        Fixed window bounds when `use_fixed_window=True`.
    T_win, base_period, step :
        Used only when `use_fixed_window=False`.

    Returns
    -------
    f_all : ndarray, shape (N,)
    ZL_all : ndarray, shape (N,2,2)
    ZS_all : ndarray, shape (N,2,2)
    """
    f_cat = []
    ZL_cat = []
    ZS_cat = []

    print("\n--- RUN SUMMARY ---")
    for name, cfg in runs.items():
        file_pos = os.path.join(base_dir, cfg["file_pos"])
        file_neg = os.path.join(base_dir, cfg["file_neg"])
        f_list = np.asarray(cfg["f_list"], float)

        if use_fixed_window:
            if fixed_t_start is None or fixed_t_end is None:
                raise ValueError("fixed_t_start and fixed_t_end must be provided when use_fixed_window=True")
            t0, t1 = fixed_t_start, fixed_t_end
            score = np.nan
        else:
            t0, t1, score = find_best_common_window(
                file_pos, file_neg,
                T=T_win, base_period=base_period,
                skiprows=skiprows, step=step,
                colmap=colmap, v_scale=v_scale, i_scale=i_scale
            )

        print(f"{name}: window=[{t0:.3f}, {t1:.3f}] s  score={score}")

        ZL, ZS = Zpn_load_and_source_from_two_files(
            file_pos, file_neg,
            t0, t1,
            f1, f_list,
            skiprows=skiprows,
            colmap=colmap,
            v_scale=v_scale,
            i_scale=i_scale
        )

        f_cat.append(f_list)
        ZL_cat.append(ZL)
        ZS_cat.append(ZS)

    f_all = np.concatenate(f_cat)
    ZL_all = np.concatenate(ZL_cat, axis=0)
    ZS_all = np.concatenate(ZS_cat, axis=0)

    idx = np.argsort(f_all)
    f_sorted = f_all[idx]
    ZL_sorted = ZL_all[idx]
    ZS_sorted = ZS_all[idx]

    if average_duplicates:
        f_u, ZL_u = _average_duplicates_by_freq(f_sorted, ZL_sorted)
        _,   ZS_u = _average_duplicates_by_freq(f_sorted, ZS_sorted)
        return f_u, ZL_u, ZS_u

    return f_sorted, ZL_sorted, ZS_sorted



# ============================================================
# Signal to noise ratio 
# ============================================================


def build_spectra_cache(
    base_dir,
    runs,
    file_base,
    f1,
    t_start,
    t_end,
    skiprows=0,
    colmap=None,
    v_scale=1000.0,
    i_scale=1000.0
):
    """
    Precompute aligned spectra once per file and voltage side.

    Returns
    -------
    cache : dict
        Keys:
            ("base", "load"), ("base", "source"),
            ("run1_pos", "load"), ("run1_pos", "source"),
            ("run1_neg", "load"), ("run1_neg", "source"), ...
    """
    cache = {}

    def _compute(path, which_voltage):
        theta = theta_from_file(
            path, t_start, t_end, f1,
            skiprows=skiprows,
            colmap=colmap,
            v_scale=v_scale,
            i_scale=i_scale
        )
        return aligned_seq_spectra(
            path, t_start, t_end, theta,
            which_voltage=which_voltage,
            skiprows=skiprows,
            colmap=colmap,
            v_scale=v_scale,
            i_scale=i_scale
        )

    file_base_full = os.path.join(base_dir, file_base)
    cache[("base", "load")] = _compute(file_base_full, "load")
    cache[("base", "source")] = _compute(file_base_full, "source")

    for run_name, cfg in runs.items():
        file_pos = os.path.join(base_dir, cfg["file_pos"])
        file_neg = os.path.join(base_dir, cfg["file_neg"])

        cache[(f"{run_name}_pos", "load")] = _compute(file_pos, "load")
        cache[(f"{run_name}_pos", "source")] = _compute(file_pos, "source")

        cache[(f"{run_name}_neg", "load")] = _compute(file_neg, "load")
        cache[(f"{run_name}_neg", "source")] = _compute(file_neg, "source")

    return cache



def compute_quality_metrics_from_cache(
    runs,
    spectra_cache,
    f1,
    eps=1e-30,
    average_duplicates=True):
    """
    Compute quality metrics using precomputed spectra cache.

    Returns same structure as old compute_quality_metrics(...).
    """
    f_all = []

    # load-side voltage
    VpL_pos_db_all = []
    VnL_pos_db_all = []
    VpL_neg_db_all = []
    VnL_neg_db_all = []

    # load-side current
    IpL_pos_db_all = []
    InL_pos_db_all = []
    IpL_neg_db_all = []
    InL_neg_db_all = []

    # source-side voltage
    VpS_pos_db_all = []
    VnS_pos_db_all = []
    VpS_neg_db_all = []
    VnS_neg_db_all = []

    cond_all = []

    base_load = spectra_cache[("base", "load")]
    base_source = spectra_cache[("base", "source")]

    for run_name, cfg in runs.items():
        f_list = np.asarray(cfg["f_list"], float)

        sb_base_L = extract_sidebands_from_spectra(base_load, f1, f_list)
        sb_pos_L  = extract_sidebands_from_spectra(spectra_cache[(f"{run_name}_pos", "load")], f1, f_list)
        sb_neg_L  = extract_sidebands_from_spectra(spectra_cache[(f"{run_name}_neg", "load")], f1, f_list)

        sb_base_S = extract_sidebands_from_spectra(base_source, f1, f_list)
        sb_pos_S  = extract_sidebands_from_spectra(spectra_cache[(f"{run_name}_pos", "source")], f1, f_list)
        sb_neg_S  = extract_sidebands_from_spectra(spectra_cache[(f"{run_name}_neg", "source")], f1, f_list)

        # load increments
        dVpL_pos = sb_pos_L["Vp"] - sb_base_L["Vp"]
        dVnL_pos = sb_pos_L["Vn"] - sb_base_L["Vn"]
        dIpL_pos = sb_pos_L["Ip"] - sb_base_L["Ip"]
        dInL_pos = sb_pos_L["In"] - sb_base_L["In"]

        dVpL_neg = sb_neg_L["Vp"] - sb_base_L["Vp"]
        dVnL_neg = sb_neg_L["Vn"] - sb_base_L["Vn"]
        dIpL_neg = sb_neg_L["Ip"] - sb_base_L["Ip"]
        dInL_neg = sb_neg_L["In"] - sb_base_L["In"]

        # source increments
        dVpS_pos = sb_pos_S["Vp"] - sb_base_S["Vp"]
        dVnS_pos = sb_pos_S["Vn"] - sb_base_S["Vn"]
        dVpS_neg = sb_neg_S["Vp"] - sb_base_S["Vp"]
        dVnS_neg = sb_neg_S["Vn"] - sb_base_S["Vn"]

        # dB metrics
        VpL_pos_db = ratio_db(dVpL_pos, sb_base_L["Vp"], eps=eps)
        VnL_pos_db = ratio_db(dVnL_pos, sb_base_L["Vn"], eps=eps)
        IpL_pos_db = ratio_db(dIpL_pos, sb_base_L["Ip"], eps=eps)
        InL_pos_db = ratio_db(dInL_pos, sb_base_L["In"], eps=eps)

        VpL_neg_db = ratio_db(dVpL_neg, sb_base_L["Vp"], eps=eps)
        VnL_neg_db = ratio_db(dVnL_neg, sb_base_L["Vn"], eps=eps)
        IpL_neg_db = ratio_db(dIpL_neg, sb_base_L["Ip"], eps=eps)
        InL_neg_db = ratio_db(dInL_neg, sb_base_L["In"], eps=eps)

        VpS_pos_db = ratio_db(dVpS_pos, sb_base_S["Vp"], eps=eps)
        VnS_pos_db = ratio_db(dVnS_pos, sb_base_S["Vn"], eps=eps)
        VpS_neg_db = ratio_db(dVpS_neg, sb_base_S["Vp"], eps=eps)
        VnS_neg_db = ratio_db(dVnS_neg, sb_base_S["Vn"], eps=eps)

        cond_run = np.zeros(len(f_list), dtype=float)
        for i in range(len(f_list)):
            DI = np.array([
                [dIpL_pos[i], dIpL_neg[i]],
                [dInL_pos[i], dInL_neg[i]]
            ], dtype=complex)
            cond_run[i] = np.linalg.cond(DI)

        f_all.append(f_list)

        VpL_pos_db_all.append(VpL_pos_db)
        VnL_pos_db_all.append(VnL_pos_db)
        IpL_pos_db_all.append(IpL_pos_db)
        InL_pos_db_all.append(InL_pos_db)

        VpL_neg_db_all.append(VpL_neg_db)
        VnL_neg_db_all.append(VnL_neg_db)
        IpL_neg_db_all.append(IpL_neg_db)
        InL_neg_db_all.append(InL_neg_db)

        VpS_pos_db_all.append(VpS_pos_db)
        VnS_pos_db_all.append(VnS_pos_db)
        VpS_neg_db_all.append(VpS_neg_db)
        VnS_neg_db_all.append(VnS_neg_db)

        cond_all.append(cond_run)

    f_all = np.concatenate(f_all)

    VpL_pos_db_all = np.concatenate(VpL_pos_db_all)
    VnL_pos_db_all = np.concatenate(VnL_pos_db_all)
    IpL_pos_db_all = np.concatenate(IpL_pos_db_all)
    InL_pos_db_all = np.concatenate(InL_pos_db_all)

    VpL_neg_db_all = np.concatenate(VpL_neg_db_all)
    VnL_neg_db_all = np.concatenate(VnL_neg_db_all)
    IpL_neg_db_all = np.concatenate(IpL_neg_db_all)
    InL_neg_db_all = np.concatenate(InL_neg_db_all)

    VpS_pos_db_all = np.concatenate(VpS_pos_db_all)
    VnS_pos_db_all = np.concatenate(VnS_pos_db_all)
    VpS_neg_db_all = np.concatenate(VpS_neg_db_all)
    VnS_neg_db_all = np.concatenate(VnS_neg_db_all)

    cond_all = np.concatenate(cond_all)

    idx = np.argsort(f_all)
    f_all = f_all[idx]

    VpL_pos_db_all = VpL_pos_db_all[idx]
    VnL_pos_db_all = VnL_pos_db_all[idx]
    IpL_pos_db_all = IpL_pos_db_all[idx]
    InL_pos_db_all = InL_pos_db_all[idx]

    VpL_neg_db_all = VpL_neg_db_all[idx]
    VnL_neg_db_all = VnL_neg_db_all[idx]
    IpL_neg_db_all = IpL_neg_db_all[idx]
    InL_neg_db_all = InL_neg_db_all[idx]

    VpS_pos_db_all = VpS_pos_db_all[idx]
    VnS_pos_db_all = VnS_pos_db_all[idx]
    VpS_neg_db_all = VpS_neg_db_all[idx]
    VnS_neg_db_all = VnS_neg_db_all[idx]

    cond_all = cond_all[idx]

    if average_duplicates:
        f_u, VpL_pos_db_all = _average_duplicates_by_freq(f_all, VpL_pos_db_all)
        _,   VnL_pos_db_all = _average_duplicates_by_freq(f_all, VnL_pos_db_all)
        _,   IpL_pos_db_all = _average_duplicates_by_freq(f_all, IpL_pos_db_all)
        _,   InL_pos_db_all = _average_duplicates_by_freq(f_all, InL_pos_db_all)

        _,   VpL_neg_db_all = _average_duplicates_by_freq(f_all, VpL_neg_db_all)
        _,   VnL_neg_db_all = _average_duplicates_by_freq(f_all, VnL_neg_db_all)
        _,   IpL_neg_db_all = _average_duplicates_by_freq(f_all, IpL_neg_db_all)
        _,   InL_neg_db_all = _average_duplicates_by_freq(f_all, InL_neg_db_all)

        _,   VpS_pos_db_all = _average_duplicates_by_freq(f_all, VpS_pos_db_all)
        _,   VnS_pos_db_all = _average_duplicates_by_freq(f_all, VnS_pos_db_all)
        _,   VpS_neg_db_all = _average_duplicates_by_freq(f_all, VpS_neg_db_all)
        _,   VnS_neg_db_all = _average_duplicates_by_freq(f_all, VnS_neg_db_all)

        _,   cond_all = _average_duplicates_by_freq(f_all, cond_all)
        f_all = f_u

    return {
        "f": f_all,
        "cond": cond_all,
        "load": {
            "Vp_pos_db": VpL_pos_db_all,
            "Vn_pos_db": VnL_pos_db_all,
            "Ip_pos_db": IpL_pos_db_all,
            "In_pos_db": InL_pos_db_all,
            "Vp_neg_db": VpL_neg_db_all,
            "Vn_neg_db": VnL_neg_db_all,
            "Ip_neg_db": IpL_neg_db_all,
            "In_neg_db": InL_neg_db_all,
        },
        "source": {
            "Vp_pos_db": VpS_pos_db_all,
            "Vn_pos_db": VnS_pos_db_all,
            "Vp_neg_db": VpS_neg_db_all,
            "Vn_neg_db": VnS_neg_db_all,
        }
    }




def get_sidebands(
    path,
    f_list,
    f1,
    t_start,
    t_end,
    which_voltage="load",
    skiprows=0,
    colmap=None,
    v_scale=1000.0,
    i_scale=1000.0
):
    """
    Compute aligned sideband quantities for one file.

    Parameters
    ----------
    path : str
        PSCAD output file.
    f_list : array_like
        Injected / evaluated frequencies [Hz].
    f1 : float
        Fundamental frequency [Hz].
    t_start, t_end : float
        Time window [s].
    which_voltage : {'load', 'source'}
        Whether to use load-side voltages or source-side delta voltages.
    skiprows, colmap, v_scale, i_scale :
        Passed to PSCAD-loading helpers.

    Returns
    -------
    sb : dict
        Output of extract_sidebands_from_spectra(...)
        with keys like 'Vp', 'Ip', 'Vn', 'In'.
    """
    theta = theta_from_file(
        path, t_start, t_end, f1,
        skiprows=skiprows,
        colmap=colmap,
        v_scale=v_scale,
        i_scale=i_scale
    )

    spec = aligned_seq_spectra(
        path, t_start, t_end, theta,
        which_voltage=which_voltage,
        skiprows=skiprows,
        colmap=colmap,
        v_scale=v_scale,
        i_scale=i_scale
    )

    return extract_sidebands_from_spectra(spec, f1, f_list)



def ratio_db(delta, base, eps=1e-30):
    return 20.0 * np.log10((np.abs(delta) + eps) / (np.abs(base) + eps))



def build_masks(qm, snr_db_threshold=10.0, use_cond=False, cond_threshold=10.0):
    """
    Build masks for load and source impedance screening.

    Load side:
        uses load current metrics
    Source side:
        uses source voltage metrics
    """
    cond_all = qm["cond"]

    masks_load = {
        (0, 0): np.asarray(qm["load"]["Ip_pos_db"]) >= snr_db_threshold,
        (0, 1): np.asarray(qm["load"]["Ip_neg_db"]) >= snr_db_threshold,
        (1, 0): np.asarray(qm["load"]["In_pos_db"]) >= snr_db_threshold,
        (1, 1): np.asarray(qm["load"]["In_neg_db"]) >= snr_db_threshold,
    }

    masks_source = {
        (0, 0): np.asarray(qm["source"]["Vp_pos_db"]) >= snr_db_threshold,
        (0, 1): np.asarray(qm["source"]["Vp_neg_db"]) >= snr_db_threshold,
        (1, 0): np.asarray(qm["source"]["Vn_pos_db"]) >= snr_db_threshold,
        (1, 1): np.asarray(qm["source"]["Vn_neg_db"]) >= snr_db_threshold,
    }

    if use_cond:
        good_cond = cond_all <= cond_threshold
        for k in masks_load:
            masks_load[k] = masks_load[k] & good_cond
        for k in masks_source:
            masks_source[k] = masks_source[k] & good_cond

    return masks_load, masks_source



def combine_masks_for_full_matrix_two_sides(masks_load, masks_source):
    full_load = masks_load[(0, 0)] & masks_load[(0, 1)] & masks_load[(1, 0)] & masks_load[(1, 1)]
    full_src  = masks_source[(0, 0)] & masks_source[(0, 1)] & masks_source[(1, 0)] & masks_source[(1, 1)]
    return full_load & full_src



# ============================================================
# Sideband quality plotting
# ============================================================

def _sideband_quality_metrics(qm, side="load", quantity="current"):
    """
    Select sideband signal-to-background metrics from the quality-metrics dict.

    Parameters
    ----------
    qm : dict
        Output from compute_quality_metrics_from_cache(...).
    side : {"load", "source"}
        Which side to show.
    quantity : {"voltage", "current"}
        Which signal quantity to show.

    Notes
    -----
    Load side supports both voltage and current.
    Source side supports voltage, since the source-side extraction is based on
    the source-side voltage difference channels.

    Returns
    -------
    f_hz : ndarray
        Frequency vector [Hz].
    metrics : dict
        Four signal-to-background metrics:
            p_pos : p component, positive-sequence injection
            n_pos : n component, positive-sequence injection
            p_neg : p component, negative-sequence injection
            n_neg : n component, negative-sequence injection
    """
    side = side.lower()
    quantity = quantity.lower()

    f_hz = np.asarray(qm["f"], dtype=float)

    if side == "load" and quantity == "voltage":
        metrics = {
            "p_pos": qm["load"]["Vp_pos_db"],
            "n_pos": qm["load"]["Vn_pos_db"],
            "p_neg": qm["load"]["Vp_neg_db"],
            "n_neg": qm["load"]["Vn_neg_db"],
        }

    elif side == "load" and quantity == "current":
        metrics = {
            "p_pos": qm["load"]["Ip_pos_db"],
            "n_pos": qm["load"]["In_pos_db"],
            "p_neg": qm["load"]["Ip_neg_db"],
            "n_neg": qm["load"]["In_neg_db"],
        }

    elif side == "source" and quantity == "voltage":
        metrics = {
            "p_pos": qm["source"]["Vp_pos_db"],
            "n_pos": qm["source"]["Vn_pos_db"],
            "p_neg": qm["source"]["Vp_neg_db"],
            "n_neg": qm["source"]["Vn_neg_db"],
        }

    else:
        raise ValueError(
            "Invalid side/quantity combination. Use "
            "side='load', quantity='voltage'; "
            "side='load', quantity='current'; or "
            "side='source', quantity='voltage'."
        )

    return f_hz, metrics



def _plot_sideband_quality_panel(
    ax,
    f_hz,
    y_db,
    title,
    threshold_db=10.0,
    ylabel=None,
):
    """
    Plot one sideband signal-to-background metric.

    Points above the threshold are shown in black.
    Points below the threshold are shown in red.
    """
    f_hz = np.asarray(f_hz, dtype=float)
    y_db = np.asarray(y_db, dtype=float)

    good = np.isfinite(f_hz) & np.isfinite(y_db) & (f_hz > 0.0)

    accepted = good & (y_db >= threshold_db)
    rejected = good & (y_db < threshold_db)

    if np.any(accepted):
        ax.semilogx(
            f_hz[accepted],
            y_db[accepted],
            "o",
            ms=3.0,
            color="0.10",
            zorder=3,
        )

    if np.any(rejected):
        ax.semilogx(
            f_hz[rejected],
            y_db[rejected],
            "o",
            ms=3.0,
            color="#CC3311",
            zorder=2,
        )

    ax.axhline(
        threshold_db,
        color="red",
        ls="--",
        lw=1.0,
        zorder=1,
    )

    ax.set_title(title, fontsize=11, pad=4)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10.5, labelpad=2)

    ax.grid(True, which="major", alpha=0.24, color="0.78", lw=0.8)
    ax.grid(True, which="minor", alpha=0.10, color="0.88", lw=0.45)

    ax.tick_params(axis="both", which="major", labelsize=9.2, pad=2)
    ax.tick_params(axis="both", which="minor", labelsize=8.5, pad=2)

    for spine in ax.spines.values():
        spine.set_color("0.35")



def print_sideband_quality(
    qm,
    side="load",
    quantity="current",
    label=None,
    threshold_db=10.0,
):
    """
    Print a short signal-to-background summary.

    This reports the lowest signal-to-background value for each sideband
    metric and how many points fall below the selected threshold.
    """
    f_hz, metrics = _sideband_quality_metrics(
        qm,
        side=side,
        quantity=quantity,
    )

    if label is None:
        label = f"{side} {quantity}"

    names = {
        "p_pos": "p component, positive-sequence injection",
        "n_pos": "n component, positive-sequence injection",
        "p_neg": "p component, negative-sequence injection",
        "n_neg": "n component, negative-sequence injection",
    }

    print("\n============================================================")
    print(f"Sideband signal-to-background summary: {label}")
    print(f"Threshold: {threshold_db:.1f} dB")
    print("============================================================")

    worst_value = np.inf
    worst_freq = np.nan
    worst_key = None

    f = np.asarray(f_hz, dtype=float)

    for key, values in metrics.items():
        y = np.asarray(values, dtype=float)
        good = np.isfinite(f) & np.isfinite(y) & (f > 0.0)

        if not np.any(good):
            print(f"{names.get(key, key)}: no finite points")
            continue

        y_good = y[good]
        f_good = f[good]

        i_min = int(np.argmin(y_good))
        y_min = float(y_good[i_min])
        f_min = float(f_good[i_min])

        n_total = int(np.count_nonzero(good))
        n_rej = int(np.count_nonzero(y_good < threshold_db))

        print(
            f"{names.get(key, key)}: "
            f"min = {y_min:.2f} dB @ {f_min:.2f} Hz, "
            f"below threshold = {n_rej}/{n_total}"
        )

        if y_min < worst_value:
            worst_value = y_min
            worst_freq = f_min
            worst_key = key

    if worst_key is not None:
        print(
            f"\nWorst point: {names.get(worst_key, worst_key)}, "
            f"{worst_value:.2f} dB @ {worst_freq:.2f} Hz"
        )



def plot_sideband_quality(
    qm,
    side="load",
    quantity="current",
    title="Sideband_Quality",
    threshold_db=10.0,
    save_dir=None,
    save=True,
    show=True,
    formats=("pdf",),
):
    """
    Plot sideband signal-to-background quality metrics.

    The four panels show:
        - p component from positive-sequence injection
        - n component from positive-sequence injection
        - p component from negative-sequence injection
        - n component from negative-sequence injection

    Points below the threshold are shown in red.
    """
    import os
    import re

    f_hz, metrics = _sideband_quality_metrics(
        qm,
        side=side,
        quantity=quantity,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.4, 5.8),
        sharex=True,
        sharey=True,
    )

    ylabel = "Signal-to-background ratio [dB]"

    _plot_sideband_quality_panel(
        axes[0, 0],
        f_hz,
        metrics["p_pos"],
        r"$p$ component, positive-sequence injection",
        threshold_db=threshold_db,
        ylabel=ylabel,
    )

    _plot_sideband_quality_panel(
        axes[0, 1],
        f_hz,
        metrics["n_pos"],
        r"$n$ component, positive-sequence injection",
        threshold_db=threshold_db,
    )

    _plot_sideband_quality_panel(
        axes[1, 0],
        f_hz,
        metrics["p_neg"],
        r"$p$ component, negative-sequence injection",
        threshold_db=threshold_db,
        ylabel=ylabel,
    )

    _plot_sideband_quality_panel(
        axes[1, 1],
        f_hz,
        metrics["n_neg"],
        r"$n$ component, negative-sequence injection",
        threshold_db=threshold_db,
    )

    axes[1, 0].set_xlabel("Frequency [Hz]", fontsize=10.5, labelpad=2)
    axes[1, 1].set_xlabel("Frequency [Hz]", fontsize=10.5, labelpad=2)

    fig.tight_layout()

    def _sanitize_filename(s):
        s = str(s).strip()

        if not s:
            s = "Sideband_Quality"

        return re.sub(r"[^\w\-_\. ]", "_", s).replace(" ", "_")

    if save:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        if save_dir is None:
            save_dir = os.path.join(base_dir, "Figures")

        os.makedirs(save_dir, exist_ok=True)

        filename_base = _sanitize_filename(title)

        for fmt in formats:
            save_path = os.path.join(save_dir, f"{filename_base}.{fmt}")

            fig.savefig(
                save_path,
                format=fmt,
                bbox_inches="tight",
                pad_inches=0.01,
            )

            print(f"Saved screening figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes



# ============================================================
# Plotting
# ============================================================


def compute_passivity_index_from_Z(f, Z):
    """
    Frequency-by-frequency passivity indicator for a 2x2 impedance matrix:
    minimum eigenvalue of the Hermitian part H = (Z + Z^H)/2.

    Positive -> passive-like at that frequency
    Negative -> non-passive-like at that frequency
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)

    N = len(f)
    pidx = np.full(N, np.nan, dtype=float)

    for k in range(N):
        Hk = 0.5 * (Z[k] + Z[k].conj().T)
        ev = np.linalg.eigvalsh(Hk)   # Hermitian => real eigenvalues
        pidx[k] = np.min(ev.real)

    return pidx



def plot_passivity_index_overlay_3(
    f_hz,
    Zmat1,
    Zmat2,
    Zmat3=None,
    names=("Case 1", "Case 2", "Case 3"),
    colors=("0.10", "#56B4E9", "#CC3311"),
    title="PassivityComparison",
    ylabel=r'$\lambda_{\min}\!\left(\frac{Z+Z^H}{2}\right)$',
    ylim=None,
    critical_freqs=None,
    marker=None,
    ms=4.0,
    lw=1.6,
    save_dir=None,
):
    """
    Overlay plot of passivity index for two or three 2x2 impedance matrices.

    - No visible title on the figure
    - Saves to BASE_DIR/Figures by default
    - Supports 2 or 3 cases
    """

    import re
    import os

    f_hz = np.asarray(f_hz, dtype=float)
    Zmat1 = np.asarray(Zmat1, dtype=complex)
    Zmat2 = np.asarray(Zmat2, dtype=complex)
    has_third = Zmat3 is not None
    if has_third:
        Zmat3 = np.asarray(Zmat3, dtype=complex)

    ncases = 3 if has_third else 2
    names = list(names[:ncases])
    colors = list(colors[:ncases])

    pidx1 = compute_passivity_index_from_Z(f_hz, Zmat1)
    pidx2 = compute_passivity_index_from_Z(f_hz, Zmat2)
    if has_third:
        pidx3 = compute_passivity_index_from_Z(f_hz, Zmat3)

    fig, ax = plt.subplots(figsize=(6.45, 4.4))

    # Main curves
    ax.semilogx(
        f_hz, pidx1,
        marker=marker, ms=ms, lw=lw, color=colors[0], label=names[0]
    )
    ax.semilogx(
        f_hz, pidx2,
        marker=marker, ms=ms, lw=lw, color=colors[1], label=names[1]
    )
    if has_third:
        ax.semilogx(
            f_hz, pidx3,
            marker=marker, ms=ms, lw=lw, color=colors[2], label=names[2]
        )

    # Zero line
    ax.axhline(0.0, color="0.25", ls="--", lw=1.0, zorder=0)

    # Optional critical frequencies
    if critical_freqs is not None:
        for f in critical_freqs:
            ax.axvline(f, color="0.45", lw=0.9, ls="--", alpha=0.8, zorder=0)

    # Style
    ax.grid(True, which="major", alpha=0.22, color="0.78", lw=0.8)
    ax.grid(True, which="minor", alpha=0.10, color="0.88", lw=0.45)

    ax.set_xlabel("Frequency [Hz]", fontsize=11, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=2)

    ax.tick_params(axis="both", which="major", labelsize=9.2, pad=2)
    ax.tick_params(axis="both", which="minor", labelsize=8.5, pad=2)

    for spine in ax.spines.values():
        spine.set_color("0.35")

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Frameless legend at top
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=ncases,
        frameon=False,
        fontsize=8.9,
        handlelength=2.0,
        columnspacing=1.35,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    ax.add_artist(leg)

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.88,
        bottom=0.16,
    )

    def _sanitize_filename(s):
        s = s.strip()
        if not s:
            s = "PassivityComparison"
        return re.sub(r"[^\w\-_\. ]", "_", s).replace(" ", "_")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if save_dir is None:
        save_dir = os.path.join(BASE_DIR, "Figures")
    os.makedirs(save_dir, exist_ok=True)

    filename = _sanitize_filename(title) + ".pdf"
    save_path = os.path.join(save_dir, filename)

    fig.canvas.draw()
    fig.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        bbox_extra_artists=(leg,),
        pad_inches=0.01,
    )

    plt.show()
    print(f"Saved figure to: {save_path}")

    return fig, ax



def plot_bode_matrix_masked(
    f_hz, Zmat, masks, title="", phase_wrapped=True,
    color_keep="tab:blue", color_reject="0.75",
    phase_ylim=(-190, 190),
    mag_ylim=None,
    labels=None):
    """
    8-panel Bode plot with accepted points connected and rejected points shown in gray.

    Parameters
    ----------
    f_hz : ndarray, shape (N,)
    Zmat : ndarray, shape (N,2,2)
    masks : dict
        Keys: (0,0), (0,1), (1,0), (1,1)
    title : str
    phase_wrapped : bool
    color_keep, color_reject : str
    phase_ylim : tuple or None
    mag_ylim : tuple or None
        Applied to all magnitude panels in this figure.
    labels : dict or None
        Example for pn:
            {"11": "pp", "12": "pn", "21": "np", "22": "nn"}
        Example for dq:
            {"11": "dd", "12": "dq", "21": "qd", "22": "qq"}
    """
    f_hz = np.asarray(f_hz, float)
    Zmat = np.asarray(Zmat, complex)

    if labels is None:
        labels = {"11": "pp", "12": "pn", "21": "np", "22": "nn"}

    Z11 = Zmat[:, 0, 0]
    Z12 = Zmat[:, 0, 1]
    Z21 = Zmat[:, 1, 0]
    Z22 = Zmat[:, 1, 1]

    fig, axes = plt.subplots(4, 2, figsize=(12.5, 13.5), sharex=True)
    fig.suptitle(title, fontsize=17, y=0.995)

    def _plot_mag(ax, z, mask, lab, ylabel=False):
        mag, _ = mag_phase(z, phase_wrapped=phase_wrapped)
        bad = ~mask

        if np.any(bad):
            ax.loglog(
                f_hz[bad], mag[bad],
                linestyle="None", marker="o", ms=4.0,
                color=color_reject, zorder=1
            )

        if np.any(mask):
            ax.loglog(
                f_hz[mask], mag[mask],
                "-", marker="o", ms=4.0, lw=1.8,
                color=color_keep, zorder=2
            )

        ax.set_title(lab)
        if ylabel:
            ax.set_ylabel("Magnitude [Ω]")
        if mag_ylim is not None:
            ax.set_ylim(*mag_ylim)
        ax.grid(True, which="both", alpha=0.35)

    def _plot_ph(ax, z, mask, lab, ylabel=False):
        _, ph = mag_phase(z, phase_wrapped=phase_wrapped)
        bad = ~mask

        if np.any(bad):
            ax.semilogx(
                f_hz[bad], ph[bad],
                linestyle="None", marker="o", ms=4.0,
                color=color_reject, zorder=1
            )

        if np.any(mask):
            ax.semilogx(
                f_hz[mask], ph[mask],
                "-", marker="o", ms=4.0, lw=1.8,
                color=color_keep, zorder=2
            )

        ax.set_title(lab)
        if ylabel:
            ax.set_ylabel("Phase [deg]")
        if phase_ylim is not None:
            ax.set_ylim(*phase_ylim)
        ax.grid(True, which="both", alpha=0.35)

    _plot_mag(axes[0, 0], Z11, masks[(0, 0)], rf"$|Z_{{{labels['11']}}}|$", ylabel=True)
    _plot_mag(axes[0, 1], Z12, masks[(0, 1)], rf"$|Z_{{{labels['12']}}}|$")
    _plot_ph (axes[1, 0], Z11, masks[(0, 0)], rf"$\angle Z_{{{labels['11']}}}$", ylabel=True)
    _plot_ph (axes[1, 1], Z12, masks[(0, 1)], rf"$\angle Z_{{{labels['12']}}}$")

    _plot_mag(axes[2, 0], Z21, masks[(1, 0)], rf"$|Z_{{{labels['21']}}}|$", ylabel=True)
    _plot_mag(axes[2, 1], Z22, masks[(1, 1)], rf"$|Z_{{{labels['22']}}}|$")
    _plot_ph (axes[3, 0], Z21, masks[(1, 0)], rf"$\angle Z_{{{labels['21']}}}$", ylabel=True)
    _plot_ph (axes[3, 1], Z22, masks[(1, 1)], rf"$\angle Z_{{{labels['22']}}}$")

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")

    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    plt.show()



def plot_bode_matrix_overlay_3(
    f_hz,
    Zmat1,
    Zmat2=None,
    Zmat3=None,
    names=("Case 1", "Case 2", "Case 3"),
    colors=("0.10", "#56B4E9", "#CC3311"),
    title="overlay_3_impedances",
    phase_wrapped=True,
    phase_ylim=(-180, 180),
    mag_ylim=None,
    labels=None,
    critical_freqs=None,
    save_dir=".",
):

    import re

    f_hz = np.asarray(f_hz, float)

    def _get(seq, i, default):
        return seq[i] if i < len(seq) else default

    cases = [
        {
            "Z": np.asarray(Zmat1, complex),
            "name": _get(names, 0, "Case 1"),
            "color": _get(colors, 0, "0.10"),
            "lw": 2.2,
        }
    ]

    if Zmat2 is not None:
        cases.append({
            "Z": np.asarray(Zmat2, complex),
            "name": _get(names, 1, "Case 2"),
            "color": _get(colors, 1, "#56B4E9"),
            "lw": 2.3,
        })

    if Zmat3 is not None:
        cases.append({
            "Z": np.asarray(Zmat3, complex),
            "name": _get(names, 2, "Case 3"),
            "color": _get(colors, 2, "#CC3311"),
            "lw": 2.3,
        })

    if labels is None:
        labels = {"11": "pp", "12": "pn", "21": "np", "22": "nn"}

    fig, axes = plt.subplots(4, 2, figsize=(13.0, 12.0), sharex=True)

    fig.patch.set_facecolor("white")
    for axrow in axes:
        for ax in axrow:
            ax.set_facecolor("white")

    grid_major = dict(which="major", alpha=0.22, color="0.78", lw=0.8)
    grid_minor_mag = dict(which="minor", alpha=0.10, color="0.88", lw=0.45)

    subplot_title_fs = 15
    axis_label_fs = 13
    tick_major_fs = 12
    tick_minor_fs = 10
    legend_fs = 12

    def _add_common_style(ax, phase=False):
        ax.grid(True, **grid_major)

        if not phase:
            ax.grid(True, **grid_minor_mag)

        ax.tick_params(axis="both", which="major", labelsize=tick_major_fs)
        ax.tick_params(axis="both", which="minor", labelsize=tick_minor_fs)

        for spine in ax.spines.values():
            spine.set_color("0.35")

        if critical_freqs is not None:
            for ff in critical_freqs:
                ax.axvline(ff, color="0.45", lw=0.9, ls="--", alpha=0.8, zorder=0)

        if phase and phase_ylim is not None:
            ax.set_ylim(*phase_ylim)
            ax.set_yticks(np.arange(-180, 181, 90))

    def _plot_mag(ax, i, j, lab, ylabel=False):
        for case in cases:
            mag, _ = mag_phase(case["Z"][:, i, j], phase_wrapped=phase_wrapped)
            ax.loglog(
                f_hz,
                mag,
                "-",
                lw=case["lw"],
                color=case["color"],
                label=case["name"],
            )

        ax.set_title(lab, fontsize=subplot_title_fs, pad=6)

        if ylabel:
            ax.set_ylabel(r"Magnitude [$\Omega$]", fontsize=axis_label_fs)

        if mag_ylim is not None:
            ax.set_ylim(*mag_ylim)

        _add_common_style(ax, phase=False)

    def _plot_ph(ax, i, j, lab, ylabel=False):
        for case in cases:
            _, ph = mag_phase(case["Z"][:, i, j], phase_wrapped=phase_wrapped)
            ax.semilogx(
                f_hz,
                ph,
                "-",
                lw=case["lw"],
                color=case["color"],
                label=case["name"],
            )

        ax.set_title(lab, fontsize=subplot_title_fs, pad=6)

        if ylabel:
            ax.set_ylabel("Phase [deg]", fontsize=axis_label_fs)

        _add_common_style(ax, phase=True)

    _plot_mag(axes[0, 0], 0, 0, rf"$|Z_{{{labels['11']}}}|$", ylabel=True)
    _plot_mag(axes[0, 1], 0, 1, rf"$|Z_{{{labels['12']}}}|$")

    _plot_ph(axes[1, 0], 0, 0, rf"$\angle Z_{{{labels['11']}}}$", ylabel=True)
    _plot_ph(axes[1, 1], 0, 1, rf"$\angle Z_{{{labels['12']}}}$")

    _plot_mag(axes[2, 0], 1, 0, rf"$|Z_{{{labels['21']}}}|$", ylabel=True)
    _plot_mag(axes[2, 1], 1, 1, rf"$|Z_{{{labels['22']}}}|$")

    _plot_ph(axes[3, 0], 1, 0, rf"$\angle Z_{{{labels['21']}}}$", ylabel=True)
    _plot_ph(axes[3, 1], 1, 1, rf"$\angle Z_{{{labels['22']}}}$")

    axes[3, 0].set_xlabel("Frequency [Hz]", fontsize=axis_label_fs)
    axes[3, 1].set_xlabel("Frequency [Hz]", fontsize=axis_label_fs)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(cases),
        frameon=False,
        bbox_to_anchor=(0.5, 0.992),
        fontsize=legend_fs,
        handlelength=2.4,
        columnspacing=1.8,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.975])

    def _sanitize_filename(s):
        s = s.strip()
        if not s:
            s = "impedance_plot"
        return re.sub(r"[^\w\-_\. ]", "_", s).replace(" ", "_")

    filename = _sanitize_filename(title) + ".pdf"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()

    print(f"Saved figure to: {save_path}")



def plot_passivity_index_overlay_3(
    f_hz,
    Zmat1,
    Zmat2=None,
    Zmat3=None,
    names=("Case 1", "Case 2", "Case 3"),
    colors=("0.10", "#56B4E9", "#CC3311"),
    title="PassivityComparison",
    ylabel=r'$\rho_Z$ [$\Omega$]',
    ylim=None,
    ylim_mode="full",              # "full", "robust", or "zero_focus"
    robust_percentiles=(1, 99),
    critical_freqs=None,
    critical_labels=None,
    shade_nonpassive=True,
    marker=None,
    ms=3.5,
    lw=1.8,
    # Inset options
    inset_zero=True,
    inset_xlim=None,
    inset_ylim=(-500, 1000),
    inset_loc="upper center",
    inset_width="60%",
    inset_height="38%",
    # Save/show options
    save_dir=None,
    save=True,
    show=True,
    formats=("pdf",),
):

    import os
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------
    f_hz = np.asarray(f_hz, dtype=float)

    def _get(seq, i, default):
        return seq[i] if i < len(seq) else default

    cases = [
        {
            "Z": np.asarray(Zmat1, dtype=complex),
            "name": _get(names, 0, "Case 1"),
            "color": _get(colors, 0, "0.10"),
        }
    ]

    if Zmat2 is not None:
        cases.append(
            {
                "Z": np.asarray(Zmat2, dtype=complex),
                "name": _get(names, 1, "Case 2"),
                "color": _get(colors, 1, "#56B4E9"),
            }
        )

    if Zmat3 is not None:
        cases.append(
            {
                "Z": np.asarray(Zmat3, dtype=complex),
                "name": _get(names, 2, "Case 3"),
                "color": _get(colors, 2, "#CC3311"),
            }
        )

    ncases = len(cases)

    # Cleaner inset defaults for single-case/base-case plots
    if ncases == 1 and inset_zero:
        if inset_loc == "upper center":
            inset_loc = "upper right"
        if inset_width == "60%":
            inset_width = "43%"
        if inset_height == "38%":
            inset_height = "30%"

    fmask = np.isfinite(f_hz) & (f_hz > 0)
    f_plot = f_hz[fmask]

    if f_plot.size == 0:
        raise ValueError("No valid positive frequency points found.")

    # ------------------------------------------------------------
    # Compute passivity indices
    # ------------------------------------------------------------
    pidx_all = []

    for case in cases:
        pidx = compute_passivity_index_from_Z(f_hz, case["Z"])
        pidx_all.append(np.asarray(pidx, dtype=float))

    y_all_list = []

    for pidx in pidx_all:
        y = pidx[fmask]
        y = y[np.isfinite(y)]

        if y.size > 0:
            y_all_list.append(y)

    if not y_all_list:
        raise ValueError("No finite passivity-index values found.")

    y_all = np.concatenate(y_all_list)

    # ------------------------------------------------------------
    # Main-axis y-limits
    # ------------------------------------------------------------
    if ylim is None:
        if ylim_mode == "full":
            y_min = np.nanmin(y_all)
            y_max = np.nanmax(y_all)

        elif ylim_mode == "robust":
            y_min, y_max = np.nanpercentile(y_all, robust_percentiles)
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)

        elif ylim_mode == "zero_focus":
            y_min_data = np.nanmin(y_all)

            if y_min_data < 0:
                neg_mag = abs(y_min_data)
                y_min = 1.15 * y_min_data

                y_pos = y_all[y_all > 0]

                if y_pos.size > 0:
                    y_pos_ref = np.nanpercentile(y_pos, 90)
                    y_max = min(y_pos_ref, 4.0 * neg_mag)
                    y_max = max(y_max, 0.25 * neg_mag)
                else:
                    y_max = 0.25 * neg_mag
            else:
                y_min = 0.0
                y_max = np.nanpercentile(y_all, 98)

        else:
            raise ValueError("ylim_mode must be 'full', 'robust', or 'zero_focus'.")

        yr = y_max - y_min

        if yr <= 0 or not np.isfinite(yr):
            yr = max(abs(y_max), 1.0)

        pad = 0.08 * yr
        ylim = (y_min - pad, y_max + pad)

    # ------------------------------------------------------------
    # Main plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.45, 4.4))

    if shade_nonpassive and ylim is not None and ylim[0] < 0:
        ax.axhspan(ylim[0], 0.0, color="0.94", zorder=-5)

    for case, pidx in zip(cases, pidx_all):
        y = pidx[fmask]
        valid = np.isfinite(f_plot) & np.isfinite(y)

        ax.semilogx(
            f_plot[valid],
            y[valid],
            marker=marker,
            ms=ms,
            lw=lw,
            color=case["color"],
            label=case["name"],
        )

    ax.axhline(0.0, color="0.20", ls="--", lw=1.0, zorder=1)

    # ------------------------------------------------------------
    # Critical frequency lines
    # ------------------------------------------------------------
    if critical_freqs is not None:
        if critical_labels is None:
            critical_labels = [None] * len(critical_freqs)

        for fc, lab in zip(critical_freqs, critical_labels):
            ax.axvline(
                fc,
                color="0.45",
                lw=0.9,
                ls="--",
                alpha=0.85,
                zorder=0,
            )

            if lab is not None:
                ax.text(
                    fc,
                    ylim[1] - 0.04 * (ylim[1] - ylim[0]),
                    lab,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7.5,
                    color="0.40",
                )

    # ------------------------------------------------------------
    # Main-axis style
    # ------------------------------------------------------------
    ax.set_ylim(*ylim)

    ax.grid(True, which="major", alpha=0.24, color="0.78", lw=0.8)
    ax.grid(True, which="minor", alpha=0.10, color="0.88", lw=0.45)

    ax.set_xlabel("Frequency [Hz]", fontsize=11, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=2)

    ax.tick_params(axis="both", which="major", labelsize=9.2, pad=2)
    ax.tick_params(axis="both", which="minor", labelsize=8.5, pad=2)

    for spine in ax.spines.values():
        spine.set_color("0.35")

    # ------------------------------------------------------------
    # Optional inset around zero/passivity boundary
    # ------------------------------------------------------------
    if inset_zero:
        if inset_xlim is None:
            inset_xlim = (np.nanmin(f_plot), np.nanmax(f_plot))

        axins = inset_axes(
            ax,
            width=inset_width,
            height=inset_height,
            loc=inset_loc,
            borderpad=1.0,
        )

        if shade_nonpassive and inset_ylim is not None and inset_ylim[0] < 0:
            axins.axhspan(inset_ylim[0], 0.0, color="0.94", zorder=-5)

        for case, pidx in zip(cases, pidx_all):
            y = pidx[fmask]
            valid = np.isfinite(f_plot) & np.isfinite(y)

            axins.semilogx(
                f_plot[valid],
                y[valid],
                marker=marker,
                ms=max(1.5, 0.65 * ms),
                lw=0.75 * lw,
                color=case["color"],
            )

        axins.axhline(0.0, color="0.20", ls="--", lw=0.9, zorder=1)

        if critical_freqs is not None:
            for fc in critical_freqs:
                axins.axvline(
                    fc,
                    color="0.45",
                    lw=0.8,
                    ls="--",
                    alpha=0.75,
                    zorder=0,
                )

        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)

        axins.grid(True, which="major", alpha=0.25, color="0.78", lw=0.7)
        axins.grid(True, which="minor", alpha=0.10, color="0.88", lw=0.4)

        axins.tick_params(axis="both", which="major", labelsize=7.0, pad=1)
        axins.tick_params(axis="both", which="minor", labelsize=6.0, pad=1)

        for spine in axins.spines.values():
            spine.set_color("0.35")

    # ------------------------------------------------------------
    # Legend and layout
    # ------------------------------------------------------------
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=ncases,
        frameon=False,
        fontsize=8.9,
        handlelength=2.0,
        columnspacing=1.35,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.88,
        bottom=0.16,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    def _sanitize_filename(s):
        s = str(s).strip()

        if not s:
            s = "PassivityComparison"

        return re.sub(r"[^\w\-_\. ]", "_", s).replace(" ", "_")

    if save:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        if save_dir is None:
            save_dir = os.path.join(base_dir, "Figures")

        os.makedirs(save_dir, exist_ok=True)

        filename_base = _sanitize_filename(title)

        for fmt in formats:
            save_path = os.path.join(save_dir, f"{filename_base}.{fmt}")

            save_kwargs = dict(
                format=fmt,
                bbox_inches="tight",
                bbox_extra_artists=(leg,),
                pad_inches=0.01,
            )

            if fmt.lower() in ("png", "jpg", "jpeg"):
                save_kwargs["dpi"] = 300

            fig.savefig(save_path, **save_kwargs)
            print(f"Saved figure to: {save_path}")

    if show:
        plt.show()

    return fig, ax



def plot_min_distance_overlay_3(
    f_hz,
    d1_case1, d2_case1,
    d1_case2, d2_case2,
    d1_case3, d2_case3,
    names=("Case 1", "Case 2", "Case 3"),
    colors=("tab:blue", "tab:orange", "tab:green"),
    title="Nyquist margin comparison",
    ylim=None
):
    """
    Overlay |1+lambda_1| and |1+lambda_2| for three cases.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=16, y=0.98)

    # lambda 1
    axes[0].semilogx(f_hz, d1_case1, "o-", ms=4, lw=1.6, color=colors[0], label=names[0])
    axes[0].semilogx(f_hz, d1_case2, "o-", ms=4, lw=1.6, color=colors[1], label=names[1])
    axes[0].semilogx(f_hz, d1_case3, "o-", ms=4, lw=1.6, color=colors[2], label=names[2])
    axes[0].set_title(r"$|1+\lambda_1|$")
    axes[0].set_ylabel("Distance to -1")
    axes[0].grid(True, which="major", alpha=0.35)
    axes[0].grid(False, which="minor")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].legend(loc="best")

    # lambda 2
    axes[1].semilogx(f_hz, d2_case1, "o-", ms=4, lw=1.6, color=colors[0], label=names[0])
    axes[1].semilogx(f_hz, d2_case2, "o-", ms=4, lw=1.6, color=colors[1], label=names[1])
    axes[1].semilogx(f_hz, d2_case3, "o-", ms=4, lw=1.6, color=colors[2], label=names[2])
    axes[1].set_title(r"$|1+\lambda_2|$")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Distance to -1")
    axes[1].grid(True, which="major", alpha=0.35)
    axes[1].grid(False, which="minor")
    if ylim is not None:
        axes[1].set_ylim(*ylim)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_nyquist_eigs(f_hz, ZL, ZS, title="Nyquist of loop-gain eigenvalues", show_freq_markers=True):
    """
    Nyquist plot of loop-gain eigenvalues.

    Returns
    -------
    eigs : ndarray, shape (N,2)
    info : dict
        Margin-like information for the two branches.
    """
    f_hz = np.asarray(f_hz, float)
    eigs = compute_loop_eigs_from_impedances(ZL, ZS)

    lam1 = eigs[:, 0]
    lam2 = eigs[:, 1]

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.set_title(title, fontsize=15)

    f_log = np.log10(f_hz)
    u = (f_log - f_log.min()) / (f_log.max() - f_log.min() + 1e-300)
    u_sharp = np.clip(u**0.55, 0.0, 1.0)

    cmap1 = plt.cm.viridis
    cmap2 = plt.cm.plasma

    def _add_gradient_line(lam, cmap, lw=2.8, zorder=2):
        pts = np.column_stack([lam.real, lam.imag])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, linewidths=lw, zorder=zorder)
        lc.set_array(u_sharp[:-1])
        lc.set_clim(0.0, 1.0)
        ax.add_collection(lc)

    _add_gradient_line(lam1, cmap1)
    _add_gradient_line(lam2, cmap2)

    ax.plot(lam1.real, lam1.imag, linestyle='None', marker='x', ms=4.5, mew=1.1,
            color='k', label=r'$\lambda_1$ (measured)', zorder=4)
    ax.plot(lam2.real, lam2.imag, linestyle='None', marker='o', ms=4.5, mew=1.1,
            mfc='none', mec='k', color='k', label=r'$\lambda_2$ (measured)', zorder=4)

    ax.plot(lam1.real[0],  lam1.imag[0],  '^', ms=8, mfc='k',    mec='k', zorder=6)
    ax.plot(lam1.real[-1], lam1.imag[-1], 's', ms=7, mfc='k',    mec='k', zorder=6)
    ax.plot(lam2.real[0],  lam2.imag[0],  '^', ms=8, mfc='none', mec='k', mew=1.6, zorder=6)
    ax.plot(lam2.real[-1], lam2.imag[-1], 's', ms=7, mfc='none', mec='k', mew=1.6, zorder=6)

    ax.plot(-1, 0, 'r+', ms=14, mew=2.0, zorder=7)
    ax.annotate(r'$-1$', xy=(-1, 0), xytext=(8, 8), textcoords='offset points', color='r')

    locus_xy = np.vstack([
        np.column_stack([lam1.real, lam1.imag]),
        np.column_stack([lam2.real, lam2.imag]),
    ])

    candidates = [
        (16, 14), (16, -18), (-16, 14), (-16, -18),
        (28, 22), (28, -26), (-28, 22), (-28, -26),
        (40, 30), (40, -34), (-40, 30), (-40, -34),
    ]

    def _best_offset_for_label(xy_point):
        x0, y0 = xy_point
        disp0 = ax.transData.transform((x0, y0))

        best = candidates[0]
        best_score = -np.inf

        for dx_pt, dy_pt in candidates:
            dx_px = dx_pt * fig.dpi / 72.0
            dy_px = dy_pt * fig.dpi / 72.0
            disp = disp0 + np.array([dx_px, dy_px])
            data_xy = ax.transData.inverted().transform(disp)
            d = np.sqrt((locus_xy[:, 0] - data_xy[0])**2 + (locus_xy[:, 1] - data_xy[1])**2)
            score = float(np.min(d))
            if score > best_score:
                best_score = score
                best = (dx_pt, dy_pt)

        return best

    def _annotate_smart(text, xy, color='k', fontsize=9):
        off = _best_offset_for_label(xy)
        ax.annotate(
            text, xy=xy,
            xytext=off, textcoords="offset points",
            fontsize=fontsize, color=color, zorder=9,
            arrowprops=dict(arrowstyle='-', lw=0.8, color='0.35', alpha=0.9),
            bbox=dict(facecolor='white', alpha=0.78, edgecolor='none', pad=1.2)
        )

    _annotate_smart(f"{f_hz[0]:.0f} Hz",  (lam1.real[0],  lam1.imag[0]))
    _annotate_smart(f"{f_hz[-1]:.0f} Hz", (lam1.real[-1], lam1.imag[-1]))
    _annotate_smart(f"{f_hz[0]:.0f} Hz",  (lam2.real[0],  lam2.imag[0]))
    _annotate_smart(f"{f_hz[-1]:.0f} Hz", (lam2.real[-1], lam2.imag[-1]))

    def _unit_circle_crossings(lam):
        out = []
        for i in range(len(lam) - 1):
            z0, z1 = lam[i], lam[i + 1]
            dz = z1 - z0
            a = (dz.real * dz.real + dz.imag * dz.imag)
            if a < 1e-30:
                continue
            b = 2.0 * (z0.real * dz.real + z0.imag * dz.imag)
            c = (z0.real * z0.real + z0.imag * z0.imag) - 1.0
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            sd = np.sqrt(disc)
            for t in [(-b + sd) / (2 * a), (-b - sd) / (2 * a)]:
                if np.isfinite(t) and 0.0 <= t <= 1.0:
                    zc = z0 + t * (z1 - z0)
                    fc = f_hz[i] + t * (f_hz[i + 1] - f_hz[i])
                    out.append((i, float(t), zc, float(fc)))
        return out

    def _margins_for_branch(lam):
        d = np.abs(lam + 1)
        i_min = int(np.argmin(d))
        min_dist = float(d[i_min])
        f_min = float(f_hz[i_min])

        ang_err = np.array([angle_error_to_neg_real_deg(z) for z in lam])
        mask = (lam.real < 0)
        i_gm = int(np.argmin(np.where(mask, ang_err, np.inf))) if np.any(mask) else int(np.argmin(ang_err))
        gm = float(1.0 / (abs(lam[i_gm]) + 1e-300))
        gm_ang_err = float(ang_err[i_gm])
        f_gm = float(f_hz[i_gm])

        crosses = _unit_circle_crossings(lam)
        if crosses:
            crosses_neg = [c for c in crosses if c[2].real < 0]
            if crosses_neg:
                crosses = crosses_neg
            best = None
            for (_, _, zc, fc) in crosses:
                pm = angle_error_to_neg_real_deg(zc)
                if (best is None) or (pm < best["pm_deg"]):
                    best = {"pm_deg": float(pm), "f_pm": float(fc), "interp": True, "resid": 0.0}
            pm_info = best
        else:
            resid = np.abs(np.abs(lam) - 1.0)
            i_pm = int(np.argmin(np.where(mask, resid, np.inf))) if np.any(mask) else int(np.argmin(resid))
            pm_info = {
                "pm_deg": float(angle_error_to_neg_real_deg(lam[i_pm])),
                "f_pm": float(f_hz[i_pm]),
                "interp": False,
                "resid": float(resid[i_pm]),
            }

        return {
            "i_min": i_min,
            "min_dist": min_dist,
            "f_min": f_min,
            "i_gm": i_gm,
            "gm": gm,
            "gm_angle_err_deg": gm_ang_err,
            "f_gm": f_gm,
            "pm_deg": pm_info["pm_deg"],
            "f_pm": pm_info["f_pm"],
            "pm_interp": pm_info["interp"],
            "pm_resid": pm_info["resid"],
        }

    m1 = _margins_for_branch(lam1)
    m2 = _margins_for_branch(lam2)

    def _stability_marker(lam, m, name, side=+1):
        i = m["i_min"]
        pt = lam[i]

        ax.plot([pt.real, -1.0], [pt.imag, 0.0], color='crimson', lw=3.0,
                alpha=0.25, solid_capstyle='round', zorder=3)
        ax.plot(pt.real, pt.imag, 'o', mfc='none', mec='crimson', mew=2.0, ms=9, zorder=8)

        alpha = 0.18
        x_txt = pt.real + alpha * (-1.0 - pt.real)
        y_txt = pt.imag + alpha * (0.0 - pt.imag)

        vx = (-1.0 - pt.real)
        vy = (0.0 - pt.imag)
        nrm = np.hypot(vx, vy) + 1e-300
        nx, ny = -vy / nrm, vx / nrm
        nudge = 0.006 * nrm
        x_txt += side * nudge * nx
        y_txt += side * nudge * ny

        ax.text(
            x_txt, y_txt,
            f"min |{name}+1| = {m['min_dist']:.3g} @ {m['f_min']:.0f} Hz",
            color='crimson', fontsize=9, zorder=9,
            bbox=dict(facecolor='white', alpha=0.78, edgecolor='none', pad=1.2)
        )

    _stability_marker(lam1, m1, "λ1", side=+1)
    _stability_marker(lam2, m2, "λ2", side=-1)

    if show_freq_markers:
        targets = np.geomspace(1.0, 1000.0, 14)
        idxs = np.array([int(np.argmin(np.abs(f_hz - t))) for t in targets], dtype=int)
        idxs = np.unique(idxs)
        idxs = [i for i in idxs if i not in (0, len(f_hz) - 1)]
        idxs = idxs[:10]

        for idx in idxs:
            c1 = cmap1(u_sharp[idx])
            c2 = cmap2(u_sharp[idx])
            ax.plot(lam1.real[idx], lam1.imag[idx], marker='o', ms=6.2, mfc=c1, mec='k', mew=0.7, zorder=7)
            ax.plot(lam2.real[idx], lam2.imag[idx], marker='D', ms=5.7, mfc=c2, mec='k', mew=0.7, zorder=7)
            _annotate_smart(f"{f_hz[idx]:.0f} Hz", (lam1.real[idx], lam1.imag[idx]))
            _annotate_smart(f"{f_hz[idx]:.0f} Hz", (lam2.real[idx], lam2.imag[idx]))

    ax.axhline(0, color='0.55', lw=0.8)
    ax.axvline(0, color='0.55', lw=0.8)
    ax.set_xlabel(r"Re{$\lambda$}")
    ax.set_ylabel(r"Im{$\lambda$}")
    ax.grid(True, alpha=0.35)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best', framealpha=0.95)

    plt.tight_layout()
    plt.show()

    return eigs, {"lam1": m1, "lam2": m2}



def plot_bode_matrix_smooth(f_hz, Zmat, title="", phase_wrapped=True):
    """
    8-panel Bode plot with visually smooth interpolated curves.
    Measured points are shown as markers.
    """
    f_hz = np.asarray(f_hz, float)
    idx = np.argsort(f_hz)
    f_hz = f_hz[idx]
    Zmat = Zmat[idx]

    f_s = np.logspace(np.log10(f_hz[0]), np.log10(f_hz[-1]), 1200)
    x = np.log10(f_hz)
    xs = np.log10(f_s)

    try:
        from scipy.interpolate import PchipInterpolator
        use_pchip = True
    except Exception:
        use_pchip = False

    def _interp_logmag(z):
        mag, _ = mag_phase(z, phase_wrapped=phase_wrapped)
        mag = np.maximum(mag, 1e-300)
        y = np.log10(mag)
        if use_pchip:
            ys = PchipInterpolator(x, y)(xs)
        else:
            ys = np.interp(xs, x, y)
            w = 21
            k = np.ones(w) / w
            ys = np.convolve(ys, k, mode="same")
        return 10**ys

    def _interp_phase(z):
        _, ph_deg = mag_phase(z, phase_wrapped=phase_wrapped)
        if use_pchip:
            phs_deg = PchipInterpolator(x, ph_deg)(xs)
        else:
            phs_deg = np.interp(xs, x, ph_deg)
            w = 21
            k = np.ones(w) / w
            phs_deg = np.convolve(phs_deg, k, mode="same")
        return phs_deg

    def _plot_mag(ax, z, lab, ylabel=False):
        mag, _ = mag_phase(z, phase_wrapped=phase_wrapped)
        ax.loglog(f_s, _interp_logmag(z), '-', lw=1.8)
        ax.loglog(f_hz, mag, 'o', ms=1.7)
        ax.set_title(lab)
        if ylabel:
            ax.set_ylabel("Magnitude [Ω]")
        ax.grid(True, which="both", alpha=0.35)

    def _plot_ph(ax, z, lab, ylabel=False):
        _, ph_deg = mag_phase(z, phase_wrapped=phase_wrapped)
        ax.semilogx(f_s, _interp_phase(z), '-', lw=1.8)
        ax.semilogx(f_hz, ph_deg, 'o', ms=1.7)
        ax.set_title(lab)
        if ylabel:
            ax.set_ylabel("Phase [deg]")
        ax.grid(True, which="both", alpha=0.35)

    Zpp = Zmat[:, 0, 0]
    Zpn = Zmat[:, 0, 1]
    Znp = Zmat[:, 1, 0]
    Znn = Zmat[:, 1, 1]

    fig, axes = plt.subplots(4, 2, figsize=(12.5, 13.5), sharex=True)
    fig.suptitle(title, fontsize=17, y=0.995)

    _plot_mag(axes[0, 0], Zpp, r"$|Z_{pp}|$", ylabel=True)
    _plot_mag(axes[0, 1], Zpn, r"$|Z_{pn}|$")
    _plot_ph (axes[1, 0], Zpp, r"$\angle Z_{pp}$", ylabel=True)
    _plot_ph (axes[1, 1], Zpn, r"$\angle Z_{pn}$")

    _plot_mag(axes[2, 0], Znp, r"$|Z_{np}|$", ylabel=True)
    _plot_mag(axes[2, 1], Znn, r"$|Z_{nn}|$")
    _plot_ph (axes[3, 0], Znp, r"$\angle Z_{np}$", ylabel=True)
    _plot_ph (axes[3, 1], Znn, r"$\angle Z_{nn}$")

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")

    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    plt.show()



def summarize_eig_margins(f_hz, eigs):
    """
    Print distance-to-(-1), GM-like, and PM-like summaries for the two branches.
    """
    for k in [0, 1]:
        lam = eigs[:, k]

        d = np.abs(lam + 1)
        i_min = int(np.argmin(d))
        dmin = float(d[i_min])
        f_dmin = float(f_hz[i_min])

        mask = lam.real < 0
        ang_err = np.array([angle_error_to_neg_real_deg(z) for z in lam])
        ang_err_eff = np.where(mask, ang_err, np.inf)
        i_gm = int(np.argmin(ang_err_eff)) if np.any(mask) else int(np.argmin(ang_err))
        gm = 1.0 / (abs(lam[i_gm]) + 1e-300)
        f_gm = float(f_hz[i_gm])
        gm_ang_err = float(ang_err[i_gm])

        mag = np.abs(lam)
        resid = np.abs(mag - 1)
        resid_eff = np.where(mask, resid, np.inf) if np.any(mask) else resid
        i_pm = int(np.argmin(resid_eff))
        pm_like = angle_error_to_neg_real_deg(lam[i_pm])
        f_pm = float(f_hz[i_pm])
        pm_resid = float(resid[i_pm])

        print(f"\nλ{k+1}:")
        print(f"  min |λ+1| = {dmin:.4g} @ {f_dmin:.2f} Hz")
        print(f"  GM~ = {gm:.4g} @ {f_gm:.2f} Hz  (angle error to -180°: {gm_ang_err:.2f}°)")
        print(f"  PM~ = {pm_like:.2f}° @ {f_pm:.2f} Hz  (||λ|-1| residual: {pm_resid:.4g})")



def plot_error_summary(err, savepath=None, cmap=plt.cm.viridis,
                       annotate_low_peak=True, low_peak_fmax=10.0):
    f = np.asarray(err["f_meas"], float)
    fro = np.asarray(err["fro_rel_pct"], float)

    mask = np.isfinite(f) & np.isfinite(fro) & (f > 0)
    f2, fro2 = f[mask], fro[mask]

    # Sort for clean line + correct peak selection
    order = np.argsort(f2)
    f2, fro2 = f2[order], fro2[order]

    kmax = int(np.argmax(fro2))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # --- Gradient line ---
    pts = np.column_stack([f2, fro2])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(np.log10(f2.min()), np.log10(f2.max()))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.0, zorder=2)
    lc.set_array(np.log10(f2[:-1]))
    lc.set_antialiased(True)
    ax.add_collection(lc)

    # --- Black markers for all points ---
    ax.plot(f2, fro2, linestyle="None", marker="o", color="k", ms=2.8, zorder=3)

    ax.autoscale()

    # Start y-axis at 0
    y0, y1 = ax.get_ylim()
    ax.set_ylim(0.0, y1 * 1.12)   # keep a bit of headroom for annotations
    ax.margins(x=0.06, y=0.10)

    # -------- Max annotation: place text below, arrow up --------
    ax.plot(f2[kmax], fro2[kmax], marker="o", ms=7, mfc="none", mec="k", mew=1.2, zorder=4)
    ax.annotate(
        f"max {fro2[kmax]:.2f}% @ {f2[kmax]:.1f} Hz",
        xy=(f2[kmax], fro2[kmax]),
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="k"),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        zorder=5,
        annotation_clip=False,
    )

    # -------- Low-frequency peak (e.g., around 3 Hz) --------
    if annotate_low_peak:
        m = f2 <= low_peak_fmax
        if np.any(m):
            ilow = np.where(m)[0][int(np.argmax(fro2[m]))]
            ax.plot(f2[ilow], fro2[ilow], marker="o", ms=6.5, mfc="none", mec="k", mew=1.1, zorder=4)
            ax.annotate(
                f"{f2[ilow]:.1f} Hz peak\n{fro2[ilow]:.2f}%",
                xy=(f2[ilow], fro2[ilow]),
                xytext=(10, 18),
                textcoords="offset points",
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="k"),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5,
                annotation_clip=False,
            )

    # Axes + grid
    ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(False, which="minor")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Frobenius relative error [%]")

    fig.tight_layout()

    plt.show()
    return fig, ax



def show_chart_grid(
    charts,
    figsize=(14.5, 16),
    mid_gap=120,
    savepath=None,
    show=True,
):
    cell_w = max(im.width for im in charts)
    cell_h = max(im.height for im in charts)
    cols, rows = 2, 4

    total_w = cell_w * cols
    total_h = cell_h * rows + mid_gap

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    for idx, im in enumerate(charts):
        r = idx // cols
        c = idx % cols

        x0 = c * cell_w + (cell_w - im.width) // 2
        y_shift = 0 if r < 2 else mid_gap
        y0 = r * cell_h + y_shift + (cell_h - im.height) // 2

        grid.paste(im, (x0, y0))

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.005, 0.02, 0.99, 0.96])
    ax.imshow(grid)
    ax.axis("off")

    handles = [
        Line2D([0], [0], color='tab:blue', lw=1.8, label='Analytical'),
        Line2D([0], [0], color='black', marker='x', linestyle='none',
               markersize=6, markeredgewidth=1.2, label='Measured')
    ]

    fig.legend(
        handles=handles,
        loc='center',
        bbox_to_anchor=(0.51, 0.49),
        ncol=2,
        frameon=False,
        fontsize=13
    )

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved figure to: {savepath}")

    plt.show()
    return fig, ax



def chart_to_image(x, y, xscale='log', yscale='log',
                   xlabel='', ylabel='', title='',
                   markers=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7.8, 3.6))

    line_kwargs = dict(color='tab:blue', linewidth=1.8)

    if xscale == 'log' and yscale == 'log':
        ax.loglog(x, y, **line_kwargs)
    elif xscale == 'log':
        ax.semilogx(x, y, **line_kwargs)
    elif yscale == 'log':
        ax.semilogy(x, y, **line_kwargs)
    else:
        ax.plot(x, y, **line_kwargs)

    if markers is not None:
        for m in markers:
            ax.plot(
                m['x'], m['y'],
                m.get('fmt', 'x'),
                linestyle='none',
                markersize=6,
                markeredgewidth=1.2,
                color=m.get('color', 'black')
            )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, which="both", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', labelsize=11)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.88)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)



def plot_nyquist_eigs_compare(
    f_hz,
    ZL_cases,
    ZS_cases,
    names=("CASE1", "CASE2", "CASE3"),
    colors=("#56B4E9", "0.10", "#CC3311"),
    title="SAVE_FILE_AS",
    arrow_freqs=(3, 170.0, 190, 270, 350.0),
    xlim=None,
    ylim=None,
):
    """
    Plot Nyquist eigenloci for dq MIMO cases.

    If one case is given:
        - lambda_1 and lambda_2 are plotted in the same figure.

    If two or three cases are given:
        - one figure is created for lambda_1
        - one figure is created for lambda_2
    """

    f_hz = np.asarray(f_hz, dtype=float)
    ZL_cases = list(ZL_cases)
    ZS_cases = list(ZS_cases)
    names = list(names)
    input_colors = list(colors)

    if not (len(ZL_cases) == len(ZS_cases) == len(names)):
        raise ValueError("ZL_cases, ZS_cases, and names must all have the same length.")

    ncases = len(names)

    if ncases not in (1, 2, 3):
        raise ValueError("This function supports 1, 2, or 3 cases.")

    if len(input_colors) < ncases:
        raise ValueError("colors must contain at least as many entries as there are cases.")

    if f_hz.ndim != 1 or len(f_hz) < 2:
        raise ValueError("f_hz must be a 1D array with at least 2 points.")

    colors = input_colors[:ncases]

    lambda_colors = (
        input_colors[0],
        input_colors[1] if len(input_colors) > 1 else "#CC3311",
    )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _angle_error_to_neg_real_deg(z):
        ang = np.degrees(np.angle(z))
        return min(abs(ang - 180.0), abs(ang + 180.0), abs(ang - 540.0))

    def _unit_circle_crossings(lam):
        out = []

        for i in range(len(lam) - 1):
            z0, z1 = lam[i], lam[i + 1]
            dz = z1 - z0

            a = dz.real * dz.real + dz.imag * dz.imag
            if a < 1e-30:
                continue

            b = 2.0 * (z0.real * dz.real + z0.imag * dz.imag)
            c = (z0.real * z0.real + z0.imag * z0.imag) - 1.0
            disc = b * b - 4.0 * a * c

            if disc < 0:
                continue

            sd = np.sqrt(disc)

            for t in ((-b + sd) / (2.0 * a), (-b - sd) / (2.0 * a)):
                if np.isfinite(t) and 0.0 <= t <= 1.0:
                    zc = z0 + t * dz
                    fc = f_hz[i] + t * (f_hz[i + 1] - f_hz[i])
                    out.append((i, float(t), zc, float(fc)))

        dedup = []
        for item in out:
            zc = item[2]
            if not any(abs(zc - old[2]) < 1e-9 for old in dedup):
                dedup.append(item)

        return dedup

    def _neg_real_axis_crossings(lam):
        out = []
        tol = 1e-14

        for i in range(len(lam) - 1):
            z0, z1 = lam[i], lam[i + 1]
            y0, y1 = z0.imag, z1.imag
            dy = y1 - y0

            if abs(dy) < tol:
                if abs(y0) < tol:
                    zc = 0.5 * (z0 + z1)
                    if zc.real < 0:
                        fc = 0.5 * (f_hz[i] + f_hz[i + 1])
                        out.append((i, 0.5, zc, float(fc)))
                continue

            t = -y0 / dy
            if 0.0 <= t <= 1.0:
                zc = z0 + t * (z1 - z0)
                if zc.real < 0:
                    fc = f_hz[i] + t * (f_hz[i + 1] - f_hz[i])
                    out.append((i, float(t), zc, float(fc)))

        dedup = []
        for item in out:
            zc = item[2]
            if not any(abs(zc - old[2]) < 1e-9 for old in dedup):
                dedup.append(item)

        return dedup

    def _worst_pm_info(lam):
        crosses = _unit_circle_crossings(lam)
        if not crosses:
            return None

        left = [c for c in crosses if c[2].real < 0]
        use = left if left else crosses

        best = None

        for _, _, zc, fc in use:
            pm_deg = _angle_error_to_neg_real_deg(zc)
            cand = {
                "point": zc,
                "freq": float(fc),
                "pm_deg": float(pm_deg),
            }

            if best is None or cand["pm_deg"] < best["pm_deg"]:
                best = cand

        return best

    def _worst_gm_info(lam):
        crosses = _neg_real_axis_crossings(lam)
        if not crosses:
            return None

        best = None

        for _, _, zc, fc in crosses:
            mag = abs(zc)
            gm_db = 20.0 * np.log10(1.0 / max(mag, 1e-300))
            cand = {
                "point": zc,
                "freq": float(fc),
                "gm_db": float(gm_db),
            }

            if best is None or cand["gm_db"] < best["gm_db"]:
                best = cand

        return best

    def _closest_to_minus_one_info(lam):
        d = np.abs(lam + 1.0)
        i = int(np.argmin(d))

        return {
            "point": lam[i],
            "freq": float(f_hz[i]),
            "dist": float(d[i]),
        }

    def _draw_arrow(ax, lam, freq, color):
        idx = int(np.argmin(np.abs(f_hz - freq)))

        if idx <= 0:
            i0, i1 = 0, 1
        elif idx >= len(lam) - 1:
            i0, i1 = len(lam) - 2, len(lam) - 1
        else:
            i0, i1 = idx - 1, idx + 1

        p0 = (lam[i0].real, lam[i0].imag)
        p1 = (lam[i1].real, lam[i1].imag)

        arrow = FancyArrowPatch(
            p0,
            p1,
            arrowstyle='-|>',
            mutation_scale=10,
            lw=1.3,
            color=color,
            alpha=0.95,
            zorder=5,
            shrinkA=0,
            shrinkB=0,
        )

        ax.add_patch(arrow)

    def _format_axes(ax):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.20)
        ax.tick_params(axis='both', labelsize=9.2, pad=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def _general_handles():
        return [
            Line2D(
                [0], [0],
                marker='x',
                ms=5.8,
                mew=1.35,
                color='red',
                linestyle='None',
                label='Critical point -1',
            ),
            Line2D(
                [0], [0],
                color='0.78',
                lw=0.85,
                ls='--',
                label='Unit circle',
            ),
            Line2D(
                [0, 1], [0, 0],
                color='0.25',
                lw=1.35,
                marker='>',
                markersize=6.5,
                markevery=[1],
                label='Increasing frequency',
            ),
        ]

    def _save_figure(fig, title, suffix, extra_artists=()):
        if not title:
            return

        out_dir = os.path.join(BASE_DIR, "Figures")
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(out_dir, f"{title}_{suffix}.pdf")

        fig.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            bbox_extra_artists=extra_artists,
            pad_inches=0.01,
        )

        print(f"Saved figure to: {save_path}")

    # -------------------------------------------------------------------------
    # Compute eigenloci
    # -------------------------------------------------------------------------
    eigs_cases = []

    for ZL, ZS in zip(ZL_cases, ZS_cases):
        eigs = compute_loop_eigs_from_impedances(ZL, ZS)
        eigs_cases.append(np.asarray(eigs))

    # -------------------------------------------------------------------------
    # Shared limits
    # -------------------------------------------------------------------------
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    uc_x = np.cos(theta)
    uc_y = np.sin(theta)

    if xlim is None or ylim is None:
        all_re = []
        all_im = []

        for eigs in eigs_cases:
            all_re.extend(eigs[:, 0].real)
            all_re.extend(eigs[:, 1].real)
            all_im.extend(eigs[:, 0].imag)
            all_im.extend(eigs[:, 1].imag)

        all_re.extend(uc_x)
        all_im.extend(uc_y)
        all_re.append(-1.0)
        all_im.append(0.0)

        all_re = np.asarray(all_re)
        all_im = np.asarray(all_im)

        if xlim is None:
            xmin, xmax = np.min(all_re), np.max(all_re)
            dx = xmax - xmin
            padx = max(0.04 * dx, 0.04)
            xlim = (xmin - padx, xmax + padx)

        if ylim is None:
            ymin, ymax = np.min(all_im), np.max(all_im)
            dy = ymax - ymin
            pady = max(0.04 * dy, 0.04)
            ylim = (ymin - pady, ymax + pady)

    # -------------------------------------------------------------------------
    # One case: plot lambda_1 and lambda_2 in the same figure
    # -------------------------------------------------------------------------
    if ncases == 1:
        fig, ax = plt.subplots(figsize=(6.45, 5.75))

        ax.plot(uc_x, uc_y, ls='--', lw=0.85, color='0.78', zorder=0)
        ax.axhline(0.0, color='0.83', lw=0.75, zorder=0)
        ax.axvline(0.0, color='0.83', lw=0.75, zorder=0)

        ax.plot(
            -1.0, 0.0,
            marker='x',
            ms=5.8,
            mew=1.35,
            color='red',
            linestyle='None',
            zorder=6,
        )

        eigs = eigs_cases[0]

        branch_handles = []
        metric_handles = []

        for branch_idx, color in zip((0, 1), lambda_colors):
            lam = eigs[:, branch_idx]
            lam_label = rf"$\lambda_{branch_idx + 1}$"

            ax.plot(lam.real, lam.imag, color=color, lw=1.45, zorder=2)

            for af in arrow_freqs:
                if f_hz[0] <= af <= f_hz[-1]:
                    _draw_arrow(ax, lam, af, color)

            branch_handles.append(
                Line2D([0], [0], color=color, lw=1.45, label=lam_label)
            )

            gm = _worst_gm_info(lam)
            if gm is not None:
                p = gm["point"]
                ax.plot(
                    p.real, p.imag,
                    marker='s',
                    ms=5.5,
                    mfc='white',
                    mec=color,
                    mew=1.35,
                    linestyle='None',
                    zorder=6,
                )
                metric_handles.append(
                    Line2D(
                        [0], [0],
                        marker='s',
                        ms=5.5,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label=rf"$\lambda_{branch_idx + 1}$: GM = {gm['gm_db']:.1f} dB @ {gm['freq']:.0f} Hz",
                    )
                )
            else:
                metric_handles.append(
                    Line2D(
                        [0], [0],
                        marker='s',
                        ms=5.5,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label=rf"$\lambda_{branch_idx + 1}$: GM = n/a",
                    )
                )

            pm = _worst_pm_info(lam)
            if pm is not None:
                p = pm["point"]
                ax.plot(
                    p.real, p.imag,
                    marker='o',
                    ms=5.6,
                    mfc='white',
                    mec=color,
                    mew=1.35,
                    linestyle='None',
                    zorder=6,
                )
                metric_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        ms=5.6,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label=rf"$\lambda_{branch_idx + 1}$: PM = {pm['pm_deg']:.1f}° @ {pm['freq']:.0f} Hz",
                    )
                )
            else:
                metric_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        ms=5.6,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label=rf"$\lambda_{branch_idx + 1}$: PM = n/a",
                    )
                )

            md = _closest_to_minus_one_info(lam)
            p = md["point"]

            ax.plot(
                p.real, p.imag,
                marker='D',
                ms=5.4,
                mfc='white',
                mec=color,
                mew=1.35,
                linestyle='None',
                zorder=6,
            )

            metric_handles.append(
                Line2D(
                    [0], [0],
                    marker='D',
                    ms=5.4,
                    mfc='white',
                    mec=color,
                    mew=1.35,
                    linestyle='None',
                    label=rf"$\lambda_{branch_idx + 1}$: min$|1+\lambda|$ = {md['dist']:.3g} @ {md['freq']:.0f} Hz",
                )
            )

        _format_axes(ax)

        ax.set_xlabel(r"$\mathrm{Re}\{\lambda\}$", fontsize=11, labelpad=2)
        ax.set_ylabel(r"$\mathrm{Im}\{\lambda\}$", fontsize=11, labelpad=2)

        leg_branches = ax.legend(
            handles=branch_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.005),
            ncol=2,
            frameon=False,
            fontsize=8.9,
            handlelength=2.0,
            columnspacing=1.35,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        ax.add_artist(leg_branches)

        leg_markers = ax.legend(
            handles=metric_handles + _general_handles(),
            loc='upper center',
            bbox_to_anchor=(0.5, -0.095),
            ncol=2,
            frameon=False,
            fontsize=7.9,
            handlelength=1.6,
            columnspacing=1.15,
            handletextpad=0.40,
            labelspacing=0.18,
            borderaxespad=0.0,
        )

        fig.canvas.draw()

        _save_figure(
            fig,
            title,
            "lambda12",
            extra_artists=(leg_branches, leg_markers),
        )

        plt.show()
        return

    # -------------------------------------------------------------------------
    # Two or three cases: one figure per eigenvalue branch
    # -------------------------------------------------------------------------
    for panel_idx in range(2):
        fig, ax = plt.subplots(figsize=(6.45, 5.75))

        ax.plot(uc_x, uc_y, ls='--', lw=0.85, color='0.78', zorder=0)
        ax.axhline(0.0, color='0.83', lw=0.75, zorder=0)
        ax.axvline(0.0, color='0.83', lw=0.75, zorder=0)

        ax.plot(
            -1.0, 0.0,
            marker='x',
            ms=5.8,
            mew=1.35,
            color='red',
            linestyle='None',
            zorder=6,
        )

        case_handles = []
        gm_handles = []
        pm_handles = []
        md_handles = []

        for name, color, eigs in zip(names, colors, eigs_cases):
            lam = eigs[:, panel_idx]

            ax.plot(lam.real, lam.imag, color=color, lw=1.4, zorder=2)

            for af in arrow_freqs:
                if f_hz[0] <= af <= f_hz[-1]:
                    _draw_arrow(ax, lam, af, color)

            case_handles.append(
                Line2D([0], [0], color=color, lw=1.4, label=name)
            )

            gm = _worst_gm_info(lam)
            if gm is not None:
                p = gm["point"]
                ax.plot(
                    p.real, p.imag,
                    marker='s',
                    ms=5.5,
                    mfc='white',
                    mec=color,
                    mew=1.35,
                    linestyle='None',
                    zorder=6,
                )
                gm_handles.append(
                    Line2D(
                        [0], [0],
                        marker='s',
                        ms=5.5,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label=f"GM = {gm['gm_db']:.1f} dB @ {gm['freq']:.0f} Hz",
                    )
                )
            else:
                gm_handles.append(
                    Line2D(
                        [0], [0],
                        marker='s',
                        ms=5.5,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label="GM = n/a",
                    )
                )

            pm = _worst_pm_info(lam)
            if pm is not None:
                p = pm["point"]
                ax.plot(
                    p.real, p.imag,
                    marker='o',
                    ms=5.6,
                    mfc='white',
                    mec=color,
                    mew=1.35,
                    linestyle='None',
                    zorder=6,
                )
                pm_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        ms=5.6,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label=f"PM = {pm['pm_deg']:.1f}° @ {pm['freq']:.0f} Hz",
                    )
                )
            else:
                pm_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        ms=5.6,
                        mfc='white',
                        mec=color,
                        mew=1.35,
                        linestyle='None',
                        label="PM = n/a",
                    )
                )

            md = _closest_to_minus_one_info(lam)
            p = md["point"]

            ax.plot(
                p.real, p.imag,
                marker='D',
                ms=5.4,
                mfc='white',
                mec=color,
                mew=1.35,
                linestyle='None',
                zorder=6,
            )

            md_handles.append(
                Line2D(
                    [0], [0],
                    marker='D',
                    ms=5.4,
                    mfc='white',
                    mec=color,
                    mew=1.35,
                    linestyle='None',
                    label=rf"min$|1+\lambda|$ = {md['dist']:.3g} @ {md['freq']:.0f} Hz",
                )
            )

        _format_axes(ax)

        if panel_idx == 0:
            ax.set_xlabel(r"$\mathrm{Re}\{\lambda_1\}$", fontsize=11, labelpad=2)
            ax.set_ylabel(r"$\mathrm{Im}\{\lambda_1\}$", fontsize=11, labelpad=2)
        else:
            ax.set_xlabel(r"$\mathrm{Re}\{\lambda_2\}$", fontsize=11, labelpad=2)
            ax.set_ylabel(r"$\mathrm{Im}\{\lambda_2\}$", fontsize=11, labelpad=2)

        blank_handle = Line2D([], [], linestyle='None', label='')

        gm_cols = gm_handles + [blank_handle] * (3 - ncases)
        pm_cols = pm_handles + [blank_handle] * (3 - ncases)
        md_cols = md_handles + [blank_handle] * (3 - ncases)

        marker_handles = []
        for gm_h, pm_h, md_h, gen_h in zip(gm_cols, pm_cols, md_cols, _general_handles()):
            marker_handles.extend([gm_h, pm_h, md_h, gen_h])

        leg_cases = ax.legend(
            handles=case_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.005),
            ncol=ncases,
            frameon=False,
            fontsize=8.9,
            handlelength=2.0,
            columnspacing=1.35,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        ax.add_artist(leg_cases)

        leg_markers = ax.legend(
            handles=marker_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.095),
            ncol=3,
            frameon=False,
            fontsize=7.9,
            handlelength=1.6,
            columnspacing=1.15,
            handletextpad=0.40,
            labelspacing=0.18,
            borderaxespad=0.0,
        )

        fig.canvas.draw()

        lam_name = f"lam{panel_idx + 1}"

        _save_figure(
            fig,
            title,
            lam_name,
            extra_artists=(leg_cases, leg_markers),
        )

        plt.show()
        


# ============================================================
# CoF correction for diagonal dq impedance plots
# ============================================================

def _phase_diff_deg(phi_a_deg, phi_b_deg):
    """
    Return the smallest signed phase difference phi_a - phi_b [deg],
    wrapped to [-180, 180).
    """
    dphi = np.asarray(phi_a_deg, dtype=float) - np.asarray(phi_b_deg, dtype=float)
    return (dphi + 180.0) % 360.0 - 180.0



def _find_mag_crossings(f_hz, z1, z2):
    """
    Find magnitude crossings between |z1| and |z2|.

    Interpolation is done in:
        x = log10(f)
        y = log10(|z1|) - log10(|z2|)

    Returns
    -------
    crossings : list of dict
        Each dictionary contains:
            f_cross     : crossing frequency [Hz]
            mag_cross   : magnitude at crossing [ohm]
            ph1_cross   : unwrapped phase of z1 at crossing [deg]
            ph2_cross   : unwrapped phase of z2 at crossing [deg]
            dphi_cross  : wrapped phase difference ph2 - ph1 [deg]
    """
    f_hz = np.asarray(f_hz, dtype=float)
    z1 = np.asarray(z1, dtype=complex)
    z2 = np.asarray(z2, dtype=complex)

    valid = (
        np.isfinite(f_hz) & (f_hz > 0.0) &
        np.isfinite(z1.real) & np.isfinite(z1.imag) &
        np.isfinite(z2.real) & np.isfinite(z2.imag) &
        (np.abs(z1) > 0.0) &
        (np.abs(z2) > 0.0)
    )

    f = f_hz[valid]
    z1 = z1[valid]
    z2 = z2[valid]

    if len(f) < 2:
        return []

    x = np.log10(f)

    mag1, ph1_unwrapped = mag_phase(z1, phase_wrapped=False)
    mag2, ph2_unwrapped = mag_phase(z2, phase_wrapped=False)

    y1 = np.log10(mag1)
    y2 = np.log10(mag2)
    dy = y1 - y2

    crossings = []

    for k in range(len(f) - 1):
        d0 = dy[k]
        d1 = dy[k + 1]

        if np.isclose(d0, 0.0, atol=1e-12):
            t = 0.0
        elif d0 * d1 < 0.0:
            t = -d0 / (d1 - d0)
        else:
            continue

        x_cross = x[k] + t * (x[k + 1] - x[k])
        f_cross = 10.0**x_cross

        y_cross = y1[k] + t * (y1[k + 1] - y1[k])
        mag_cross = 10.0**y_cross

        ph1_cross = ph1_unwrapped[k] + t * (ph1_unwrapped[k + 1] - ph1_unwrapped[k])
        ph2_cross = ph2_unwrapped[k] + t * (ph2_unwrapped[k + 1] - ph2_unwrapped[k])

        crossings.append({
            "f_cross": float(f_cross),
            "mag_cross": float(mag_cross),
            "ph1_cross": float(ph1_cross),
            "ph2_cross": float(ph2_cross),
            "dphi_cross": float(_phase_diff_deg(ph2_cross, ph1_cross)),
        })

    # Remove duplicates that can occur when a crossing lands exactly on a
    # sampled frequency point.
    if len(crossings) <= 1:
        return crossings

    out = [crossings[0]]
    for c in crossings[1:]:
        if abs(np.log10(c["f_cross"]) - np.log10(out[-1]["f_cross"])) > 1e-6:
            out.append(c)

    return out



def compute_cof(ZL_dq, ZS_dq, swap_branches=False, eps=1e-18):
    """
    Compute correction factors for diagonal dq impedance analysis.

    The full MIMO loop gain is defined as:
        L = Z_source * inv(Z_load)

    The raw diagonal loop gains are:
        Ldd_raw = Zs_dd / Zl_dd
        Lqq_raw = Zs_qq / Zl_qq

    The correction factors are chosen so that:
        CoF_dd * Ldd_raw = lambda_1
        CoF_qq * Lqq_raw = lambda_2

    The correction is applied to the source-side diagonal impedances:
        Zs_dd_corr = CoF_dd * Zs_dd
        Zs_qq_corr = CoF_qq * Zs_qq

    Parameters
    ----------
    ZL_dq : ndarray, shape (N, 2, 2)
        Load-side dq impedance matrices.
    ZS_dq : ndarray, shape (N, 2, 2)
        Source-side dq impedance matrices.
    swap_branches : bool
        If True, lambda_1 and lambda_2 are swapped before the correction
        factors are formed.
    eps : float
        Small threshold used to avoid division by near-zero values.

    Returns
    -------
    out : dict
        Contains loop-gain eigenvalues, raw diagonal loop gains, correction
        factors, and corrected source-side dq impedance matrix.
    """
    ZL_dq = np.asarray(ZL_dq, dtype=complex)
    ZS_dq = np.asarray(ZS_dq, dtype=complex)

    eigs = compute_loop_eigs_from_impedances(ZL_dq, ZS_dq)

    lam1 = eigs[:, 0].copy()
    lam2 = eigs[:, 1].copy()

    if swap_branches:
        lam1, lam2 = lam2, lam1

    Zl_dd = ZL_dq[:, 0, 0]
    Zl_qq = ZL_dq[:, 1, 1]
    Zs_dd = ZS_dq[:, 0, 0]
    Zs_qq = ZS_dq[:, 1, 1]

    Ldd_raw = np.full_like(Zs_dd, np.nan + 1j*np.nan)
    Lqq_raw = np.full_like(Zs_qq, np.nan + 1j*np.nan)

    ok_dd = np.abs(Zl_dd) > eps
    ok_qq = np.abs(Zl_qq) > eps

    Ldd_raw[ok_dd] = Zs_dd[ok_dd] / Zl_dd[ok_dd]
    Lqq_raw[ok_qq] = Zs_qq[ok_qq] / Zl_qq[ok_qq]

    CoF_dd = np.full_like(Zs_dd, np.nan + 1j*np.nan)
    CoF_qq = np.full_like(Zs_qq, np.nan + 1j*np.nan)

    ok_Ldd = np.abs(Ldd_raw) > eps
    ok_Lqq = np.abs(Lqq_raw) > eps

    CoF_dd[ok_Ldd] = lam1[ok_Ldd] / Ldd_raw[ok_Ldd]
    CoF_qq[ok_Lqq] = lam2[ok_Lqq] / Lqq_raw[ok_Lqq]

    ZS_dq_corr = ZS_dq.copy()
    ZS_dq_corr[:, 0, 0] = CoF_dd * ZS_dq[:, 0, 0]
    ZS_dq_corr[:, 1, 1] = CoF_qq * ZS_dq[:, 1, 1]

    return {
        "eigs": eigs,
        "lam_dd": lam1,
        "lam_qq": lam2,
        "Ldd_raw": Ldd_raw,
        "Lqq_raw": Lqq_raw,
        "CoF_dd": CoF_dd,
        "CoF_qq": CoF_qq,
        "ZS_dq_corr": ZS_dq_corr,
    }



def plot_cof_diagonal(
    f_hz,
    ZL_dq,
    ZS_dq,
    ZS_dq_corr,
    masks_load=None,
    masks_source=None,
    title="CoF_Diagonal_Impedance",
    phase_wrapped=True,
    mag_ylim=None,
    phase_ylim=(-180, 180),
    label_source="Source",
    label_source_corr="CoF-corrected source",
    label_load="Load",
    color_source="0.55",
    color_source_corr="#CC3311",
    color_load="0.10",
    color_cross="#0072B2",
    critical_freqs=None,
    annotate=False,
    save_dir=None,
    save=True,
    show=True,
    formats=("pdf",),
    cross_marker_style="X",
    cross_marker_size=6.0,
    cross_marker_edgewidth=1.2,
    cross_marker_facecolor="#0072B2",
    cross_line_alpha=0.35,
):
    """
    Plot diagonal dq impedances with CoF-corrected source diagonals.

    The plot compares:
        - raw source impedance
        - CoF-corrected source impedance
        - load impedance

    Only the diagonal dq elements are shown:
        - Zdd
        - Zqq

    Magnitude crossings and phase differences are computed between:
        CoF-corrected source impedance and load impedance.

    Returns
    -------
    crossing_info : dict
        crossing_info["dd"] and crossing_info["qq"] contain lists of
        crossing dictionaries.
    """
    import os
    import re
    from matplotlib.lines import Line2D

    f_hz = np.asarray(f_hz, dtype=float)
    ZL_dq = np.asarray(ZL_dq, dtype=complex)
    ZS_dq = np.asarray(ZS_dq, dtype=complex)
    ZS_dq_corr = np.asarray(ZS_dq_corr, dtype=complex)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.0, 7.2),
        sharex="col"
    )

    fig.patch.set_facecolor("white")

    for axrow in axes:
        for ax in axrow:
            ax.set_facecolor("white")

    grid_major = dict(which="major", alpha=0.22, color="0.78", lw=0.8)
    grid_minor_mag = dict(which="minor", alpha=0.10, color="0.88", lw=0.45)

    subplot_title_fs = 15
    axis_label_fs = 13
    tick_major_fs = 12
    tick_minor_fs = 10
    legend_fs = 12

    lw_raw = 1.4
    lw_main = 2.2

    panels = [
        {"idx": (0, 0), "name": "dd", "axm": axes[0, 0], "axp": axes[1, 0]},
        {"idx": (1, 1), "name": "qq", "axm": axes[0, 1], "axp": axes[1, 1]},
    ]

    crossing_info = {}

    def _style_axis(ax, phase=False):
        ax.grid(True, **grid_major)

        if not phase:
            ax.grid(True, **grid_minor_mag)

        ax.tick_params(axis="both", which="major", labelsize=tick_major_fs)
        ax.tick_params(axis="both", which="minor", labelsize=tick_minor_fs)

        for spine in ax.spines.values():
            spine.set_color("0.35")

        if critical_freqs is not None:
            for fc in critical_freqs:
                ax.axvline(
                    fc,
                    color="0.45",
                    lw=0.9,
                    ls="--",
                    alpha=0.75,
                    zorder=0,
                )

    for panel in panels:
        i, j = panel["idx"]
        name = panel["name"]
        axm = panel["axm"]
        axp = panel["axp"]

        zL = ZL_dq[:, i, j]
        zS = ZS_dq[:, i, j]
        zSc = ZS_dq_corr[:, i, j]

        if masks_load is None:
            mL = np.ones(len(f_hz), dtype=bool)
        else:
            mL = np.asarray(masks_load[(i, j)], dtype=bool)

        if masks_source is None:
            mS = np.ones(len(f_hz), dtype=bool)
        else:
            mS = np.asarray(masks_source[(i, j)], dtype=bool)

        mask = (
            mL & mS &
            np.isfinite(f_hz) & (f_hz > 0.0) &
            np.isfinite(zL.real) & np.isfinite(zL.imag) &
            np.isfinite(zS.real) & np.isfinite(zS.imag) &
            np.isfinite(zSc.real) & np.isfinite(zSc.imag)
        )

        f = f_hz[mask]
        zL = zL[mask]
        zS = zS[mask]
        zSc = zSc[mask]

        if len(f) == 0:
            crossing_info[name] = []
            continue

        magL, phL = mag_phase(zL, phase_wrapped=phase_wrapped)
        magS, phS = mag_phase(zS, phase_wrapped=phase_wrapped)
        magSc, phSc = mag_phase(zSc, phase_wrapped=phase_wrapped)

        axm.loglog(
            f, magS,
            "--",
            lw=lw_raw,
            color=color_source,
            alpha=0.95,
            label=label_source,
        )

        axm.loglog(
            f, magSc,
            "-",
            lw=lw_main,
            color=color_source_corr,
            label=label_source_corr,
        )

        axm.loglog(
            f, magL,
            "-",
            lw=lw_main,
            color=color_load,
            label=label_load,
        )

        axm.set_title(rf"$|Z_{{{name}}}|$", fontsize=subplot_title_fs, pad=6)
        axm.set_ylabel(r"Magnitude [$\Omega$]", fontsize=axis_label_fs)

        if mag_ylim is not None:
            axm.set_ylim(*mag_ylim)

        _style_axis(axm, phase=False)

        axp.semilogx(
            f, phS,
            "--",
            lw=lw_raw,
            color=color_source,
            alpha=0.95,
        )

        axp.semilogx(
            f, phSc,
            "-",
            lw=lw_main,
            color=color_source_corr,
        )

        axp.semilogx(
            f, phL,
            "-",
            lw=lw_main,
            color=color_load,
        )

        axp.set_title(rf"$\angle Z_{{{name}}}$", fontsize=subplot_title_fs, pad=6)
        axp.set_ylabel("Phase [deg]", fontsize=axis_label_fs)
        axp.set_xlabel("Frequency [Hz]", fontsize=axis_label_fs)

        if phase_ylim is not None:
            axp.set_ylim(*phase_ylim)
            axp.set_yticks(np.arange(-180, 181, 90))

        _style_axis(axp, phase=True)

        crossings = _find_mag_crossings(f, zL, zSc)
        crossing_info[name] = crossings

        for n, c in enumerate(crossings):
            fc = c["f_cross"]
            mc = c["mag_cross"]
            dphi = c["dphi_cross"]

            axm.plot(
                fc,
                mc,
                marker=cross_marker_style,
                ms=cross_marker_size,
                mfc=cross_marker_facecolor,
                mec=color_cross,
                mew=cross_marker_edgewidth,
                linestyle="None",
                zorder=8,
            )

            for ax in (axm, axp):
                ax.axvline(
                    fc,
                    color=color_cross,
                    ls="--",
                    lw=0.85,
                    alpha=cross_line_alpha,
                    zorder=1,
                )

            if annotate:
                txt = rf"{fc:.1f} Hz" + "\n" + rf"$\Delta\phi={dphi:+.0f}^\circ$"

                axm.annotate(
                    txt,
                    xy=(fc, mc),
                    xytext=(8, 10 if n % 2 == 0 else -28),
                    textcoords="offset points",
                    ha="left",
                    va="bottom" if n % 2 == 0 else "top",
                    color=color_cross,
                    fontsize=8.8,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        fc="white",
                        ec="none",
                        alpha=0.82,
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color_cross,
                        lw=0.9,
                    ),
                    zorder=8,
                )

    legend_handles = [
        Line2D(
            [0], [0],
            color=color_source,
            lw=lw_raw,
            ls="--",
            label=label_source,
        ),
        Line2D(
            [0], [0],
            color=color_source_corr,
            lw=lw_main,
            ls="-",
            label=label_source_corr,
        ),
        Line2D(
            [0], [0],
            color=color_load,
            lw=lw_main,
            ls="-",
            label=label_load,
        ),
        Line2D(
            [0], [0],
            color=color_cross,
            lw=0,
            marker=cross_marker_style,
            mfc=cross_marker_facecolor,
            mec=color_cross,
            mew=cross_marker_edgewidth,
            ms=cross_marker_size,
            label="Magnitude crossing",
        ),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.992),
        fontsize=legend_fs,
        handlelength=2.4,
        columnspacing=1.8,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _sanitize_filename(s):
        s = str(s).strip()
        if not s:
            s = "CoF_Diagonal_Impedance"
        return re.sub(r"[^\w\-_\. ]", "_", s).replace(" ", "_")

    if save:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        if save_dir is None:
            save_dir = os.path.join(base_dir, "Figures")

        os.makedirs(save_dir, exist_ok=True)

        filename_base = _sanitize_filename(title)

        for fmt in formats:
            save_path = os.path.join(save_dir, f"{filename_base}.{fmt}")

            save_kwargs = dict(
                format=fmt,
                bbox_inches="tight",
                bbox_extra_artists=(leg,),
                pad_inches=0.01,
            )

            if fmt.lower() in ("png", "jpg", "jpeg"):
                save_kwargs["dpi"] = 300

            fig.savefig(save_path, **save_kwargs)
            print(f"Saved figure to: {save_path}")

    if show:
        plt.show()

    return crossing_info

