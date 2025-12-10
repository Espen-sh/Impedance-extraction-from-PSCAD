# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 19:13:49 2025

@author: Espen
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# ---- Parameters ----
f1 = 50.0
f_min = 0.1
f_max = 2000.0
n = 1000
f = np.logspace(np.log10(f_min), np.log10(f_max), n)
w = 2*np.pi*f
w1 = 2*np.pi*f1


def Z_dq_seriesRLC(s, R=10, L=10e-3, C=100e-6):
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


def mag_phase(Z):
    """
    Convert a complex quantity to magnitude and phase.

    Parameters
    ----------
    Z : complex or array_like of complex
        Input impedance or eigenvalue(s).

    Returns
    -------
    mag : ndarray
        Magnitude |Z|.
    phase_deg : ndarray
        Phase of Z in degrees, wrapped to the interval [-180, 180).
    """
    mag = np.abs(Z)
    phase_deg = np.angle(Z, deg=True)
    phase_deg = (phase_deg + 180) % 360 - 180 # wrap to [-180, 180)
    return mag, phase_deg


def Z_pn_from_Zdq(Zdq):
    """
    Transform a dq impedance matrix to the modified sequence (p–n) domain.

    The transformation is
        Z_pn = A_Z · Z_dq · A_Z^{-1}

    where A_Z is the dq → (p, n) transformation matrix.

    Parameters
    ----------
    Zdq : (2, 2) complex ndarray
        dq-domain impedance matrix.

    Returns
    -------
    Z_pn : (2, 2) complex ndarray
        Impedance matrix in the modified sequence domain.
    """
    AZ = (1/np.sqrt(2)) * np.array([[1,  1j],
                                    [1, -1j]],
        dtype=complex)
    AZ_inv = (1/np.sqrt(2)) * np.array([[1,   1],
                                        [-1j, 1j]],
        dtype=complex)
    return AZ @ Zdq @ AZ_inv


def load_pscad_segment(path, t_start, t_end, skiprows=0):
    """
    Load a time window of load-side measurements from a PSCAD .out file.

    The function extracts the three-phase voltages and currents at the load bus,
    scales them by the base value (here by 1000), and shifts the time vector so that the window
    starts at t = 0.

    Parameters
    ----------
    path : str
        Path to the PSCAD .out file.
    t_start, t_end : float
        Start and end time [s] of the window to extract.
    skiprows : int, optional
        Number of header lines to skip in the file.

    Returns
    -------
    t_seg : (N,) ndarray
        Time vector in the selected window, re-referenced to start at 0.
    Va_seg, Vb_seg, Vc_seg : (N,) ndarray
        Phase voltages at the load bus.
    Ia_seg, Ib_seg, Ic_seg : (N,) ndarray
        Phase currents at the load bus.
    """
    data = np.loadtxt(path, skiprows=skiprows)

    t  = data[:, 0]
    Va = data[:, 1] * 1000; Ia = data[:, 2] * 1000
    Vb = data[:, 4] * 1000; Ib = data[:, 5] * 1000
    Ic = data[:, 8] * 1000; Vc = data[:, 9] * 1000

    time_window = (t >= t_start) & (t <= t_end)
    t_seg  = t[time_window] - t[time_window][0]   # re-reference time to 0
    Va_seg = Va[time_window]; Ia_seg = Ia[time_window]
    Vb_seg = Vb[time_window]; Ib_seg = Ib[time_window]
    Vc_seg = Vc[time_window]; Ic_seg = Ic[time_window]

    return t_seg, Va_seg, Vb_seg, Vc_seg, Ia_seg, Ib_seg, Ic_seg


def load_pscad_segment_with_source(path, t_start, t_end, skiprows=0):
    """
    Load a time window of load- and source-side measurements from PSCAD.

    This function returns:
      - Load bus phase voltages and currents.
      - ΔV across the source impedance for each phase.

    Here ΔV is assumed to be provided directly by PSCAD (Vas, Vbs, Vcs),
    representing the voltage drop across the source impedance.

    Parameters
    ----------
    path : str
        Path to the PSCAD .out file.
    t_start, t_end : float
        Start and end time [s] of the window to extract.
    skiprows : int, optional
        Number of header lines to skip in the file.

    Returns
    -------
    t_seg : (N,) ndarray
        Time vector in the selected window, re-referenced to start at 0.
    Va_seg, Vb_seg, Vc_seg : (N,) ndarray
        Load bus phase voltages.
    Ia_seg, Ib_seg, Ic_seg : (N,) ndarray
        Phase currents (same current through source and load in series).
    dVa_seg, dVb_seg, dVc_seg : (N,) ndarray
        Phase voltage drops across the source impedance.
    """
    data = np.loadtxt(path, skiprows=skiprows)

    t  = data[:, 0]
    Va = data[:, 1] * 1000; Ia = data[:, 2] * 1000
    Vas = data[:, 3] * 1000; Vbs = data[:, 6] * 1000; Vcs = data[:, 7] * 1000
    Ic = data[:, 8] * 1000; Vc = data[:, 9] * 1000
    Ib = data[:, 5] * 1000; Vb = data[:, 4] * 1000

    time_window = (t >= t_start) & (t <= t_end)
    t_seg  = t[time_window] - t[time_window][0]

    Va_seg = Va[time_window]; Ia_seg = Ia[time_window]
    Vb_seg = Vb[time_window]; Ib_seg = Ib[time_window]
    Vc_seg = Vc[time_window]; Ic_seg = Ic[time_window]

    # voltage drop across source impedance (ΔV from PSCAD)
    dVa_seg = Vas[time_window]
    dVb_seg = Vbs[time_window]
    dVc_seg = Vcs[time_window]

    return (t_seg,
            Va_seg, Vb_seg, Vc_seg,
            Ia_seg, Ib_seg, Ic_seg,
            dVa_seg, dVb_seg, dVc_seg)


def fft_three_phase(t, a, b, c):
    """
    Compute FFT of three phase signals over a given time window.

    Assuming the time window contains an integer
    number of periods for the tones of interest.

    Parameters
    ----------
    t : (N,) ndarray
        Time vector.
    a, b, c : (N,) ndarray
        Phase waveforms.

    Returns
    -------
    freqs : (N,) ndarray
        FFT frequency axis [Hz].
    A_f, B_f, C_f : (N,) complex ndarray
        Complex FFT spectra of the three phases.
    """
    N  = len(t)
    dt = t[1] - t[0]
    window = np.ones(N)     # integer-number-of-periods

    A_f = np.fft.fft(a*window) / N
    B_f = np.fft.fft(b*window) / N
    C_f = np.fft.fft(c*window) / N
    freqs = np.fft.fftfreq(N, d=dt)   # Hz

    return freqs, A_f, B_f, C_f


def sym_components(Va, Vb, Vc):
    """
    Compute symmetrical components (0, positive, negative) from phase quantities.

    Parameters
    ----------
    Va, Vb, Vc : complex ndarray
        Phase-domain phasors or FFT spectra.

    Returns
    -------
    V0, Vp, Vn : complex ndarray
        Zero-, positive-, and negative-sequence components.
    """
    a = np.exp(1j * 2*np.pi/3)
    T = (1/3) * np.array([[1, 1,    1   ],
                          [1, a,    a**2],
                          [1, a**2, a   ]], dtype=complex)
    stacked = np.vstack([Va, Vb, Vc])
    V0, Vp, Vn = T @ stacked
    return V0, Vp, Vn


def find_bin(freqs, f_target):
    """
    Return index of the FFT bin closest to a given target frequency.

    Parameters
    ----------
    freqs : ndarray
        FFT frequency axis [Hz].
    f_target : float
        Target frequency [Hz].

    Returns
    -------
    idx : int
        Index of the closest FFT bin.
    """
    return np.argmin(np.abs(freqs - f_target))


def compute_aligned_spectra(path, t_start, t_end, f1, theta1, skiprows=0):
    """
    Compute dq-aligned positive/negative sequence spectra for load and source.

    For a perturbed PSCAD run that contains both bus voltages/currents and the
    source voltage drop ΔV, this function:

      1. Loads a time window [t_start, t_end].
      2. Computes FFT of load bus voltages/currents and ΔV.
      3. Transforms each to symmetrical components.
      4. Rotates positive sequence by -θ1 and negative sequence by +θ1 so that
         they are aligned with the dq reference frame used in the analytical model.

    Parameters
    ----------
    path : str
        Path to perturbed PSCAD .out file (pos- or neg-sequence injection).
    t_start, t_end : float
        Start and end time [s] of the analysis window.
    f1 : float
        Fundamental frequency [Hz].
    theta1 : float
        Reference angle (from unperturbed run), in radians.
    skiprows : int, optional
        Number of header lines to skip.

    Returns
    -------
    spectra : dict
        Dictionary with keys:
          "freqs"   : FFT frequency axis
          "VpL_al", "VnL_al", "IpL_al", "InL_al" (load side)
          "VpS_al", "VnS_al", "IpS_al", "InS_al" (source side, ΔV)
    """
    (t, Va, Vb, Vc,
     Ia, Ib, Ic,
     dVa, dVb, dVc) = load_pscad_segment_with_source(
         path, t_start, t_end, skiprows=skiprows
    )

    # FFT bus voltages & currents (load)
    freqs, Va_f, Vb_f, Vc_f = fft_three_phase(t, Va, Vb, Vc)
    _,     Ia_f, Ib_f, Ic_f = fft_three_phase(t, Ia, Ib, Ic)

    # FFT source voltage drop & currents (source)
    freqs2, dVa_f, dVb_f, dVc_f = fft_three_phase(t, dVa, dVb, dVc)
    # (freqs2 should equal freqs; same dt and window)

    # symmetrical components: load
    V0L_f, VpL_f, VnL_f = sym_components(Va_f, Vb_f, Vc_f)
    I0L_f, IpL_f, InL_f = sym_components(Ia_f, Ib_f, Ic_f)

    # symmetrical components: source (ΔV)
    V0S_f, VpS_f, VnS_f = sym_components(dVa_f, dVb_f, dVc_f)
    I0S_f, IpS_f, InS_f = sym_components(Ia_f, Ib_f, Ic_f)  # same currents

    # align to dq reference (same θ1 as before)
    rot_p = np.exp(-1j * theta1)
    rot_n = np.exp(+1j * theta1)

    VpL_al = VpL_f * rot_p
    IpL_al = IpL_f * rot_p
    VnL_al = VnL_f * rot_n
    InL_al = InL_f * rot_n

    VpS_al = VpS_f * rot_p
    IpS_al = IpS_f * rot_p
    VnS_al = VnS_f * rot_n
    InS_al = InS_f * rot_n

    return {
        "freqs": freqs,
        # load side
        "VpL_al": VpL_al, "VnL_al": VnL_al,
        "IpL_al": IpL_al, "InL_al": InL_al,
        # source side
        "VpS_al": VpS_al, "VnS_al": VnS_al,
        "IpS_al": IpS_al, "InS_al": InS_al,
    }


def extract_sidebands_from_spectra(spectra, f1, f_sweep_list):
    """
    Extract positive- and negative-sequence sidebands for a list of sweep frequencies.

    For each sweep frequency f, this function returns:
      - Vp_pos(f1 + f), Ip_pos(f1 + f)
      - Vn_neg(f1 - f), In_neg(f1 - f)

    with correct handling of the sign/conjugation for negative frequencies.

    Parameters
    ----------
    spectra : dict
        Dictionary with keys "freqs", "Vp_al", "Vn_al", "Ip_al", "In_al"
        (typically produced by compute_aligned_spectra).
    f1 : float
        Fundamental frequency [Hz].
    f_sweep_list : array_like
        List/array of injected modulation frequencies f [Hz].

    Returns
    -------
    result : dict
        Keys:
          "f_sweep" : array of sweep frequencies
          "Vp_pos"  : Vp at f1 + f
          "Ip_pos"  : Ip at f1 + f
          "Vn_neg"  : Vn at f1 - f (properly conjugated if needed)
          "In_neg"  : In at f1 - f (properly conjugated if needed)
    """
    freqs = spectra["freqs"]
    Vp_al = spectra["Vp_al"]
    Vn_al = spectra["Vn_al"]
    Ip_al = spectra["Ip_al"]
    In_al = spectra["In_al"]

    f_sweep_list = np.atleast_1d(f_sweep_list)
    Nf = len(f_sweep_list)

    Vp_pos = np.zeros(Nf, dtype=complex)
    Ip_pos = np.zeros(Nf, dtype=complex)
    Vn_neg = np.zeros(Nf, dtype=complex)
    In_neg = np.zeros(Nf, dtype=complex)

    for i, f in enumerate(f_sweep_list):
        # sideband frequencies in the dq-domain
        f_pos = f1 + f      # always positive
        f_neg = f - f1      # can be negative or positive

        k_pos = find_bin(freqs, f_pos)
        k_neg = find_bin(freqs, abs(f_neg))

        Vp_pos[i] = Vp_al[k_pos]
        Ip_pos[i] = Ip_al[k_pos]

        if f < f1:
            # negative frequency – negative sequence → need conjugation
            Vn_neg[i] = np.conj(Vn_al[k_neg])
            In_neg[i] = np.conj(In_al[k_neg])
        else:
            # positive frequency – negative sequence → no conjugation
            Vn_neg[i] = Vn_al[k_neg]
            In_neg[i] = In_al[k_neg]

    return {
        "f_sweep": f_sweep_list,
        "Vp_pos": Vp_pos,
        "Ip_pos": Ip_pos,
        "Vn_neg": Vn_neg,
        "In_neg": In_neg,
    }


def compute_theta1_from_unperturbed(path, t_start, t_end, f1, skiprows=0):
    """
    Compute the reference angle θ1 from an unperturbed PSCAD run.

    θ1 is defined as the angle of the positive-sequence fundamental voltage at
    the terminal bus:
        θ1 = angle{ Vp(f1) }

    This angle is later used to align all perturbed simulations with the dq
    reference frame.

    Parameters
    ----------
    path : str
        Path to the unperturbed PSCAD .out file.
    t_start, t_end : float
        Start and end time [s] of the analysis window.
    f1 : float
        Fundamental frequency [Hz].
    skiprows : int, optional
        Number of header lines to skip in the file.

    Returns
    -------
    theta1 : float
        Reference angle in radians.
    """
    # Extract time window
    t, Va, Vb, Vc, Ia, Ib, Ic = load_pscad_segment(
        path, t_start, t_end, skiprows=skiprows
    )

    # FFT of phase voltages
    freqs, Va_f, Vb_f, Vc_f = fft_three_phase(t, Va, Vb, Vc)

    # Symmetrical components for all frequency bins
    V0_f, Vp_f, Vn_f = sym_components(Va_f, Vb_f, Vc_f)

    # Locate positive-sequence fundamental
    k1 = find_bin(freqs, f1)
    Vp1 = Vp_f[k1]

    theta1 = np.angle(Vp1)  # radians
    return theta1


def Zpn_load_and_source_from_two_files_multi(file_pos_inj, file_neg_inj,
                                             t_start, t_end, f1,
                                             f_sweep_list, theta1,
                                             skiprows=0):
    """
    Identify load and source modified-sequence impedance matrices from two simulations.

    Using one PSCAD simulation with pure positive-sequence injection and one with pure
    negative-sequence injection, this function reconstructs the 2×2 modified
    sequence impedance matrices for both the load and the source branch for all
    sweep frequencies.

    Parameters
    ----------
    file_pos_inj : str
        PSCAD .out file with positive-sequence multi-tone injection.
    file_neg_inj : str
        PSCAD .out file with negative-sequence multi-tone injection.
    t_start, t_end : float
        Start and end time [s] of the common analysis window.
    f1 : float
        Fundamental frequency [Hz].
    f_sweep_list : array_like
        List of frequencies f [Hz] used in the multi-tone injection.
    theta1 : float
        Reference angle from unperturbed run, in radians.
    skiprows : int, optional
        Number of header lines to skip in the PSCAD files.

    Returns
    -------
    ZL_all : (Nf, 2, 2) complex ndarray
        Modified-sequence impedance matrices of the load at each sweep frequency.
    ZS_all : (Nf, 2, 2) complex ndarray
        Modified-sequence impedance matrices of the source at each sweep frequency.
    """
    # aligned spectra (both load & source) from each file
    spec_pos = compute_aligned_spectra(
        file_pos_inj, t_start, t_end, f1, theta1, skiprows)
    spec_neg = compute_aligned_spectra(
        file_neg_inj, t_start, t_end, f1, theta1, skiprows)

    # sidebands for load
    sb_pos_L = extract_sidebands_from_spectra(
        {
            "freqs": spec_pos["freqs"],
            "Vp_al": spec_pos["VpL_al"],
            "Vn_al": spec_pos["VnL_al"],
            "Ip_al": spec_pos["IpL_al"],
            "In_al": spec_pos["InL_al"],
        },
        f1, f_sweep_list
    )

    sb_neg_L = extract_sidebands_from_spectra(
        {
            "freqs": spec_neg["freqs"],
            "Vp_al": spec_neg["VpL_al"],
            "Vn_al": spec_neg["VnL_al"],
            "Ip_al": spec_neg["IpL_al"],
            "In_al": spec_neg["InL_al"],
        },
        f1, f_sweep_list
    )

    # 3) sidebands for source
    sb_pos_S = extract_sidebands_from_spectra(
        {
            "freqs": spec_pos["freqs"],
            "Vp_al": spec_pos["VpS_al"],
            "Vn_al": spec_pos["VnS_al"],
            "Ip_al": spec_pos["IpS_al"],
            "In_al": spec_pos["InS_al"],
        },
        f1, f_sweep_list
    )

    sb_neg_S = extract_sidebands_from_spectra(
        {
            "freqs": spec_neg["freqs"],
            "Vp_al": spec_neg["VpS_al"],
            "Vn_al": spec_neg["VnS_al"],
            "Ip_al": spec_neg["IpS_al"],
            "In_al": spec_neg["InS_al"],
        },
        f1, f_sweep_list
    )

    # assemble Z matrices for all sweep frequencies
    f_sweep_list = np.atleast_1d(f_sweep_list)
    Nf = len(f_sweep_list)

    ZL_all = np.zeros((Nf, 2, 2), dtype=complex)
    ZS_all = np.zeros((Nf, 2, 2), dtype=complex)

    for i in range(Nf):
        # --- load ---
        Vp1L, Ip1L = sb_pos_L["Vp_pos"][i], sb_pos_L["Ip_pos"][i]
        Vn1L, In1L = sb_pos_L["Vn_neg"][i], sb_pos_L["In_neg"][i]

        Vp2L, Ip2L = sb_neg_L["Vp_pos"][i], sb_neg_L["Ip_pos"][i]
        Vn2L, In2L = sb_neg_L["Vn_neg"][i], sb_neg_L["In_neg"][i]

        V_mat_L = np.array([[Vp1L, Vp2L],
                            [Vn1L, Vn2L]], dtype=complex)
        I_mat_L = np.array([[Ip1L, Ip2L],
                            [In1L, In2L]], dtype=complex)

        ZL_all[i, :, :] = V_mat_L @ np.linalg.inv(I_mat_L)

        # --- source ---
        Vp1S, Ip1S = sb_pos_S["Vp_pos"][i], sb_pos_S["Ip_pos"][i]
        Vn1S, In1S = sb_pos_S["Vn_neg"][i], sb_pos_S["In_neg"][i]

        Vp2S, Ip2S = sb_neg_S["Vp_pos"][i], sb_neg_S["Ip_pos"][i]
        Vn2S, In2S = sb_neg_S["Vn_neg"][i], sb_neg_S["In_neg"][i]

        V_mat_S = np.array([[Vp1S, Vp2S],
                            [Vn1S, Vn2S]], dtype=complex)
        I_mat_S = np.array([[Ip1S, Ip2S],
                            [In1S, In2S]], dtype=complex)

        ZS_all[i, :, :] = V_mat_S @ np.linalg.inv(I_mat_S)

    return ZL_all, ZS_all


def chart_to_image(x, y, xscale='log', yscale='log',
                   xlabel='', ylabel='', title='',
                   markers=None):
    """
    Create a Matplotlib chart and return it as a PIL image.

    The function plots a single analytic curve (x, y) with optional simulation
    markers overlaid, and returns the rendered figure as a PNG image in memory.

    Parameters
    ----------
    x, y : array_like
        Data for the analytic curve.
    xscale, yscale : {'log', 'linear'}
        Axis scaling for x and y.
    xlabel, ylabel, title : str
        Axis labels and plot title.
    markers : list of dict, optional
        Each dict can have keys:
          'x' : x positions of markers
          'y' : y positions of markers
          'fmt' : Matplotlib marker format (e.g. 'x', 'o')
          'color' : marker color

    Returns
    -------
    img : PIL.Image
        Rendered plot as an image object.
    """
    fig, ax = plt.subplots(figsize=(6.2, 3.2))  # Figure size

    line_kwargs = dict(color='tab:blue', linewidth=1.6)  # analytic line style

    if xscale == 'log' and yscale == 'log':
        ax.loglog(x, y, **line_kwargs)
    elif xscale == 'log':
        ax.semilogx(x, y, **line_kwargs)
    elif yscale == 'log':
        ax.semilogy(x, y, **line_kwargs)
    else:
        ax.plot(x, y, **line_kwargs)

    # add markers from simulation
    if markers is not None:
        for m in markers:
            ax.plot(
                m['x'], m['y'],
                m.get('fmt', 'x'),
                linestyle='none',
                markersize=7,
                markeredgewidth=1.4,
                color=m.get('color', 'black')
            )

    ax.grid(True, which="both", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.88)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def window_periodicity_score_norm(t, signals, base_period=1.0):
    """
    Compute a normalized periodicity score for a time window.

    The score compares the first and last segments of length `base_period`
    within the window, for each signal, and returns the mean of:

        RMS(x_last - x_first) / RMS(x_first)

    over all channels. A smaller score indicates a more periodic (steady-state)
    window.

    Parameters
    ----------
    t : (N,) ndarray
        Time vector for the window.
    signals : list of array_like
        List/tuple of signals, e.g. [Va, Vb, Vc, Ia, Ib, Ic] over the same t.
    base_period : float
        Length of the segment [s] used for comparison. For a multi-tone with
        integer Hz spacing, 1.0 s is convenient.

    Returns
    -------
    score : float
        Mean normalized periodicity score across all signals.
    """
    t = np.asarray(t)
    dt = t[1] - t[0]
    N_period = int(round(base_period / dt))

    if len(t) < 2 * N_period:
        raise ValueError("Window too short for two base_period segments.")

    start0 = 0
    end0   = start0 + N_period
    end1   = len(t)
    start1 = end1 - N_period

    scores = []
    for x in signals:
        x = np.asarray(x)
        x0 = x[start0:end0]
        x1 = x[start1:end1]
        diff_rms = np.sqrt(np.mean((x1 - x0)**2))
        x_rms    = np.sqrt(np.mean(x0**2)) + 1e-12  # avoid div-by-zero
        scores.append(diff_rms / x_rms)

    return float(np.mean(scores))


def find_best_common_window(file_pos, file_neg,
                            T=4.0, base_period=1.0,
                            skiprows=0, step=0.1):
    """
    Find the most periodic common time window for the two perturbed simulations.

    This function scans start times t_start and evaluates a window
    [t_start, t_start + T] for both the positive- and negative-sequence
    injection simulations. For each window it computes a normalized
    periodicity score (using `window_periodicity_score_norm`) across the six
    load-side signals:

        Va, Vb, Vc, Ia, Ib, Ic

    The window with the lowest average score (over both files) is returned.

    Parameters
    ----------
    file_pos : str
        PSCAD .out file with positive-sequence injection.
    file_neg : str
        PSCAD .out file with negative-sequence injection.
    T : float
        Window length [s] used for FFT and impedance extraction.
    base_period : float
        Comparison period [s] used inside the periodicity metric.
    skiprows : int, optional
        Number of header lines to skip in the PSCAD files.
    step : float
        Step size [s] when scanning candidate t_start values.

    Returns
    -------
    best_t_start : float
        Start time [s] of the best common window.
    best_t_end : float
        End time [s] of the best common window (best_t_start + T).
    best_score : float
        Periodicity score of the selected window.
    """
    # Load full time series once from each perturbed file
    t_p, Va_p, Vb_p, Vc_p, Ia_p, Ib_p, Ic_p = load_pscad_segment(
        file_pos, 0.0, 1e9, skiprows=skiprows)
    t_n, Va_n, Vb_n, Vc_n, Ia_n, Ib_n, Ic_n = load_pscad_segment(
        file_neg, 0.0, 1e9, skiprows=skiprows)

    # assumes same time vector in both perturbed files
    t = t_p
    dt = t[1] - t[0]
    t_min = float(t[0])
    t_max = float(t[-1])

    last_start = t_max - T - dt
    t_start_candidates = np.arange(t_min, last_start + 0.5*step, step)

    best_score = np.inf
    best_t_start = None

    # choose an effective comparison period that fits twice in the window
    # e.g. for T=1 s and base_period=1 s -> effective_bp = 0.5 s
    effective_bp = min(base_period, T / 2.0 - 1e-9)
    if effective_bp <= 0:
        raise ValueError("Window T too small.")

    for t0 in t_start_candidates:
        t1 = t0 + T

        mask = (t >= t0) & (t <= t1)
        if np.count_nonzero(mask) < 10:
            continue

        t_win = t[mask]

        try:
            score_p = window_periodicity_score_norm(
                t_win,
                [Va_p[mask], Vb_p[mask], Vc_p[mask],
                 Ia_p[mask], Ib_p[mask], Ic_p[mask]],
                base_period=effective_bp
            )
            score_n = window_periodicity_score_norm(
                t_win,
                [Va_n[mask], Vb_n[mask], Vc_n[mask],
                 Ia_n[mask], Ib_n[mask], Ic_n[mask]],
                base_period=effective_bp
            )
        except ValueError:
            continue

        score = 0.5 * (score_p + score_n)

        if score < best_score:
            best_score = score
            best_t_start = t0

    if best_t_start is None:
        raise RuntimeError("No valid window found.")

    return best_t_start, best_t_start + T, best_score