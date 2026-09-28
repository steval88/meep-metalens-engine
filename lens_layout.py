#!/usr/bin/env python3
"""
lens_layout.py - map a metalens phase profile onto a unit-cell LUT and check it.

Steps
  1. Target: standard hyperbolic profile of an on-axis lens (focus in air)
         phi_t(r) = -(2 pi / lambda) * ( sqrt(r^2 + f^2) - f ),   f = N * D
     (phi_t(0) = 0, decreasing outwards), plus an optional global offset phi0
     (default 0). Wrapped phases are reported in (0, 2 pi], so the lens centre
     is at 2 pi and the phase decreases outwards (converging lens).
     Same sign convention as the LUT (more optical path -> larger phase, Meep's
     exp(-i w t)): the centre needs the most delay.
  2. LUT: T(r_p) and unwrapped phase(r_p) from lut.csv (written by lut_sweep.py),
     linearly interpolated onto candidate pillar radii (--r_step, e.g. the
     fabrication grid). Complex transmission t(r_p) = sqrt(T) exp(i phase).
  3. Mapping: every lattice site (square lattice, period P from the LUT, one
     site at the lens centre) gets the pillar radius minimising
         | t(r_p) - exp(i (phi_t + phi0)) |
     i.e. best phase match with a preference for high transmission. The global
     offset phi0 does not change the focus. Default --phi0 0 is the textbook
     profile; --phi0 auto instead picks the offset that maximises the overlap
     |< t exp(-i(phi_t + phi0)) >|^2 with this LUT (reported, for comparison).
  4. Check: scalar angular-spectrum propagation of the designed field (one
     complex value per site, zero outside the aperture) to the focal region,
     compared with an ideal lens (unit amplitude, exact phase) of the same
     aperture. Reports focal-spot FWHM, Strehl ratio and focusing efficiency.

Approximations (state them when quoting numbers): local periodic approximation
(each pillar behaves as in an infinite array of identical pillars), scalar
diffraction, normal incidence, single wavelength.

Output: layouts/<YYYYmmdd-HHMMSS>[_tag]/
  config.json, summary.txt, sites.csv (x, y, pillar radius, target/realised
  phase, T for every site), radial.png, layout_map.png, focus.png

Example
  python lens_layout.py --lut runs/20260928-144928_H1000_dr12p5/results/res050/H1000nm/lut.csv \
                        --D 100 --N 3 --tag D100_f3
"""
import argparse
import datetime
import json
import os
import re

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lut", required=True, help="lut.csv from lut_sweep.py")
    ap.add_argument("--D", type=float, default=100.0, help="lens diameter (um)")
    ap.add_argument("--N", type=float, default=3.0, help="f-number, f = N * D")
    ap.add_argument("--wl", type=float, default=None, help="wavelength (um); default: from LUT")
    ap.add_argument("--r_step", type=float, default=0.001,
                    help="pillar-radius grid (um) for the choice, e.g. fabrication step")
    ap.add_argument("--root", default="layouts", help="parent folder for outputs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--phi0", default="0",
                    help="global phase offset (rad) added to the standard profile "
                         "(default 0), or 'auto' to optimise it for the LUT")
    ap.add_argument("--no_propagate", action="store_true", help="skip the focusing check")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
def read_lut(path):
    head = [l for l in open(path) if l.startswith("#")]
    meta = {}
    for key in ("wl", "P", "H"):
        m = re.search(rf"\b{key}=([0-9.eE+-]+)", head[0])
        if m:
            meta[key] = float(m.group(1))
    d = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    r_um = d[:, 0] * 1e-3
    T = d[:, 1]
    ph = d[:, 6]                       # unwrapped phase relative to r_min
    hit = d[:, 8].astype(bool) if d.shape[1] > 8 else np.zeros(len(r_um), bool)
    order = np.argsort(r_um)
    return meta, r_um[order], T[order], ph[order], hit[order]


def wrap_2pi(x):
    """Wrap into (0, 2 pi]: 0 -> 2 pi, so a converging lens starts at 2 pi and decreases."""
    return 2 * np.pi - np.mod(-np.asarray(x), 2 * np.pi)


def target_slope(rho, f, wl):
    """d phi_t / d r (rad/um) of the hyperbolic profile."""
    k = 2 * np.pi / wl
    return -k * rho / np.sqrt(rho**2 + f**2)


def target_phase(rho, R, f, wl):
    k = 2 * np.pi / wl
    return -k * (np.sqrt(rho**2 + f**2) - f)


def main():
    a = parse_args()
    meta, r_lut, T_lut, ph_lut, hit = read_lut(a.lut)
    wl = a.wl or meta.get("wl")
    P = meta.get("P")
    if wl is None or P is None:
        raise SystemExit("could not read wl / P from the LUT header; pass --wl")
    R = a.D / 2
    f = a.N * a.D
    NA = np.sin(np.arctan(R / f))

    # ---- candidate pillars (interpolated LUT) -----------------------------
    r_c = np.arange(r_lut[0], r_lut[-1] + 1e-9, a.r_step)
    T_c = np.interp(r_c, r_lut, T_lut)
    ph_c = np.interp(r_c, r_lut, ph_lut)
    t_c = np.sqrt(T_c) * np.exp(1j * ph_c)
    span = ph_lut.max() - ph_lut.min()

    # best candidate for every phase value (0.05 deg bins) -> fast offset search
    nb = 7200
    theta = (np.arange(nb) + 0.5) * 2 * np.pi / nb
    best = np.argmin(np.abs(t_c[None, :] - np.exp(1j * theta)[:, None]), axis=1)

    # ---- lattice sites inside the aperture --------------------------------
    n = int(np.floor(R / P))
    i = np.arange(-n, n + 1)
    X, Y = np.meshgrid(i * P, i * P, indexing="xy")
    rho = np.hypot(X, Y)
    inside = rho <= R + 1e-9
    xs, ys, rhos = X[inside], Y[inside], rho[inside]
    phi_t = target_phase(rhos, R, f, wl)

    def assign(phi0):
        b = np.floor(np.mod(phi_t + phi0, 2 * np.pi) / (2 * np.pi) * nb).astype(int) % nb
        return best[b]

    def overlap(idx, phi0):
        return abs(np.mean(t_c[idx] * np.exp(-1j * (phi_t + phi0)))) ** 2

    if str(a.phi0).lower() == "auto":
        offsets = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        scores = [overlap(assign(p0), p0) for p0 in offsets]
        phi0 = offsets[int(np.argmax(scores))]
    else:
        phi0 = float(a.phi0) % (2 * np.pi)
    idx = assign(phi0)
    r_site, T_site, t_site = r_c[idx], T_c[idx], t_c[idx]
    err = np.angle(t_site * np.exp(-1j * (phi_t + phi0)))       # realised - target, wrapped
    ovl = overlap(idx, phi0)

    # ---- output folder ------------------------------------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(a.root, stamp + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(out, exist_ok=True)
    json.dump({"args": vars(a), "wl_um": wl, "P_um": P, "f_um": f, "NA": NA,
               "lut_meta": meta, "phi0_rad": float(phi0),
               "created": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    np.savetxt(os.path.join(out, "sites.csv"),
               np.column_stack([xs, ys, rhos, r_site * 1e3, wrap_2pi(phi_t + phi0),
                                wrap_2pi(np.angle(t_site)), err, T_site]),
               delimiter=",", fmt="%.6f",
               header=(f"D={a.D} um, N={a.N}, f={f} um, wl={wl} um, P={P} um, phi0={phi0:.6f} rad, "
                       f"LUT={a.lut}\n"
                       "x_um,y_um,rho_um,pillar_radius_nm,target_phase_rad(0,2pi],realised_phase_rad(0,2pi],"
                       "phase_error_rad,T"))

    # ---- focusing check (scalar angular spectrum) ---------------------------
    focus = None
    if not a.no_propagate:
        focus = propagate(xs, ys, t_site, phi_t + phi0, P, wl, f, R, NA)

    # ---- summary -----------------------------------------------------------
    lines = [
        f"lens: D = {a.D} um, f/{a.N:g} -> f = {f:.1f} um, NA = {NA:.4f}, lambda = {wl*1e3:.0f} nm",
        f"lattice: P = {P*1e3:.0f} nm, {xs.size} sites inside the aperture "
        f"({rhos[np.abs(ys) < 1e-9].size // 2 + 1} along one radius)",
        f"LUT: {a.lut}",
        f"     radii {r_lut[0]*1e3:.1f}-{r_lut[-1]*1e3:.1f} nm, {r_lut.size} simulated points, "
        f"phase span {span:.3f} rad ({span/(2*np.pi):.2f} x 2pi), "
        f"{int(hit.sum())} point(s) flagged hit_max_time",
        f"     candidate radii every {a.r_step*1e3:.1f} nm (linear interpolation between simulated points)",
        f"target: phi(r) = -(2pi/lambda)(sqrt(r^2+f^2) - f) + phi0, "
        f"phi0 = {phi0:.3f} rad = {phi0/np.pi:.3f} pi "
        f"({'optimised for this LUT' if str(a.phi0).lower() == 'auto' else 'fixed by --phi0'})",
        f"phase range phi(0) - phi(R) = {abs(target_phase(R, R, f, wl)):.3f} rad "
        f"= {abs(target_phase(R, R, f, wl))/(2*np.pi):.3f} x 2pi "
        f"({int(np.floor(abs(target_phase(R, R, f, wl))/(2*np.pi)))} full zones)",
        f"slope dphi/dr: {target_slope(R/2, f, wl):.4f} rad/um at r = R/2, "
        f"{target_slope(R, f, wl):.4f} rad/um at r = R "
        f"(= {abs(target_slope(R, f, wl))*P:.3f} rad per period, "
        f"{2*np.pi/(abs(target_slope(R, f, wl))*P):.1f} sites per 2pi zone at the rim)",
        f"phase error: RMS {np.sqrt(np.mean(err**2)):.3f} rad, max |err| {np.max(np.abs(err)):.3f} rad",
        f"transmission over sites: mean {T_site.mean():.4f}, min {T_site.min():.4f}",
        f"overlap with ideal unit-amplitude lens |<t e^-i phi_t>|^2 = {ovl:.4f}",
        f"pillar radii used: {r_site.min()*1e3:.1f}-{r_site.max()*1e3:.1f} nm "
        f"({np.unique(np.round(r_site*1e3, 3)).size} distinct values)",
    ]
    if focus:
        lines += [
            f"focusing check (scalar angular spectrum, local periodic approximation):",
            f"  axial intensity peak at z = {focus['z_peak']:.1f} um (design f = {f:.1f} um)",
            f"  focal spot FWHM {focus['fwhm']:.2f} um (ideal lens {focus['fwhm_ideal']:.2f} um, "
            f"0.514 lambda/NA = {0.514*wl/NA:.2f} um)",
            f"  Strehl ratio (peak / ideal peak) {focus['strehl']:.3f}",
            f"  focusing efficiency (power within r = 3 x ideal FWHM / power incident on aperture) "
            f"{focus['eff']:.3f} (ideal lens {focus['eff_ideal']:.3f})",
        ]
    txt = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    print(txt)

    plots(out, a, wl, P, f, R, xs, ys, rhos, phi_t, phi0, t_site, r_site, T_site, err,
          r_lut, T_lut, ph_lut, focus)
    print(f"[lens_layout] written to {out}")


# --------------------------------------------------------------------------- #
def propagate(xs, ys, t_site, phi_ideal, P, wl, f, R, NA):
    """Scalar angular-spectrum propagation on the lattice grid (dx = P)."""
    n_ap = int(np.ceil(2 * R / P)) + 1
    Ngrid = int(2 ** np.ceil(np.log2(4 * n_ap)))            # >= 4x aperture: no wrap-around
    c = Ngrid // 2
    ix = np.round(xs / P).astype(int) + c
    iy = np.round(ys / P).astype(int) + c
    U = np.zeros((Ngrid, Ngrid), complex)
    U0 = np.zeros((Ngrid, Ngrid), complex)
    U[iy, ix] = t_site
    U0[iy, ix] = np.exp(1j * phi_ideal)
    k = 2 * np.pi / wl
    fx = np.fft.fftfreq(Ngrid, d=P)
    KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fx, indexing="xy")
    kz2 = k**2 - KX**2 - KY**2
    prop = kz2 > 0
    kz = np.sqrt(np.where(prop, kz2, 0.0))
    A, A0 = np.fft.fft2(U), np.fft.fft2(U0)
    # Meep convention exp(-i w t): forward propagation multiplies by exp(+i kz z)
    field = lambda Aspec, z: np.fft.ifft2(Aspec * np.where(prop, np.exp(1j * kz * z), 0))

    zs = np.linspace(0.5 * f, 1.5 * f, 81)
    Iax = np.array([abs(field(A, z)[c, c]) ** 2 for z in zs])
    Iax0 = np.array([abs(field(A0, z)[c, c]) ** 2 for z in zs])
    z_peak = zs[int(np.argmax(Iax))]

    Uf, Uf0 = field(A, f), field(A0, f)
    I, I0 = abs(Uf) ** 2, abs(Uf0) ** 2
    x = (np.arange(Ngrid) - c) * P

    def fwhm(prof):
        # linear interpolation of the half-maximum crossing on both sides of the peak
        p = prof / prof.max()
        j = int(np.argmax(p))
        lo = j
        while lo > 0 and p[lo] > 0.5:
            lo -= 1
        hi = j
        while hi < len(p) - 1 and p[hi] > 0.5:
            hi += 1
        xl = x[lo] + (0.5 - p[lo]) * (x[lo + 1] - x[lo]) / (p[lo + 1] - p[lo])
        xr = x[hi - 1] + (0.5 - p[hi - 1]) * (x[hi] - x[hi - 1]) / (p[hi] - p[hi - 1])
        return xr - xl

    w, w0 = fwhm(I[c, :]), fwhm(I0[c, :])
    XX, YY = np.meshgrid(x, x, indexing="xy")
    disk = np.hypot(XX, YY) <= 3 * w0
    P_in = xs.size                                           # |incident|^2 = 1 per site
    return {
        "z": zs, "Iax": Iax, "Iax0": Iax0, "z_peak": z_peak,
        "x": x, "Ix": I[c, :], "Ix0": I0[c, :],
        "fwhm": w, "fwhm_ideal": w0,
        "strehl": I.max() / I0.max(),
        "eff": I[disk].sum() / P_in, "eff_ideal": I0[disk].sum() / P_in,
    }


def plots(out, a, wl, P, f, R, xs, ys, rhos, phi_t, phi0, t_site, r_site, T_site, err,
          r_lut, T_lut, ph_lut, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter, AutoMinorLocator

    BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#6b6b6b"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

    def grid(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", color="#d6d6d6", lw=0.8)
        ax.grid(True, which="minor", color="#eeeeee", lw=0.5)

    def pi_fmt(v, _):
        n = int(round(v / (np.pi / 2)))
        return {0: "0", 1: "π/2", 2: "π", 3: "3π/2", 4: "2π"}.get(n, f"{n}π/2")

    title = (f"D = {a.D:g} µm, f/{a.N:g} (f = {f:.0f} µm), λ = {wl*1e3:.0f} nm, "
             f"P = {P*1e3:.0f} nm")

    # ---- radial cut (sites on the +x axis) ----------------------------------
    sel = (np.abs(ys) < 1e-9) & (xs >= -1e-9)
    o = np.argsort(xs[sel])
    xr = xs[sel][o]
    fig, ax = plt.subplots(3, 2, figsize=(12, 11.5), dpi=150)
    rr = np.linspace(0, R, 2000)
    phi_rr = target_phase(rr, R, f, wl) + phi0
    rng = abs(target_phase(R, R, f, wl))
    # (row 0) the two quantities that define the lens: phase range and slope
    ax[0, 0].plot(rr, phi_rr, c=BLUE, lw=2)
    ax[0, 0].set(xlabel="radial position (µm)", ylabel="φ(r) (rad)",
                 title=f"Target phase, unwrapped\nφ(0) − φ(R) = {rng:.2f} rad "
                       f"({rng/(2*np.pi):.2f} × 2π)")
    sl = target_slope(rr, f, wl)
    ax[0, 1].plot(rr, sl, c=BLUE, lw=2)
    ax[0, 1].set(xlabel="radial position (µm)", ylabel="dφ/dr (rad/µm)",
                 title=f"Target slope: {sl[-1]:.3f} rad/µm at the rim\n"
                       f"({abs(sl[-1])*P:.3f} rad per {P*1e3:.0f} nm period)")
    # (row 1) wrapped phase at the sites and chosen pillar radius
    ax[1, 0].plot(rr, wrap_2pi(phi_rr), c=MUTED, lw=1, label="target")
    ax[1, 0].plot(xr, wrap_2pi(np.angle(t_site[sel][o])), "o", c=BLUE, ms=3.5,
                  label="realised (LUT)")
    ax[1, 0].set(xlabel="radial position (µm)", ylabel="phase, wrapped to (0, 2π]",
                 ylim=(0, 2 * np.pi + 0.05), title="Phase at the lattice sites")
    ax[1, 0].yaxis.set_major_locator(MultipleLocator(np.pi / 2))
    ax[1, 0].yaxis.set_major_formatter(FuncFormatter(pi_fmt))
    ax[1, 0].legend(frameon=False, fontsize=8, loc="lower left")
    ax[1, 1].plot(xr, r_site[sel][o] * 1e3, "o-", c=BLUE, ms=3.5, lw=1)
    ax[1, 1].set(xlabel="radial position (µm)", ylabel="pillar radius (nm)",
                 title="Pillar radius")
    # (row 2) transmission and phase error
    ax[2, 0].plot(xr, T_site[sel][o], "o-", c=BLUE, ms=3.5, lw=1)
    ax[2, 0].set(xlabel="radial position (µm)", ylabel="T", ylim=(0, 1.05),
                 title="Transmission of the chosen pillar")
    ax[2, 1].hist(err, bins=60, color=BLUE)
    ax[2, 1].set(xlabel="phase error, realised − target (rad)", ylabel="number of sites",
                 title=f"Phase error, all {err.size} sites (RMS {np.sqrt(np.mean(err**2)):.3f} rad)")
    for x in ax.flat:
        grid(x)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "radial.png"))
    plt.close(fig)

    # ---- 2-D map of pillar radii --------------------------------------------
    n = int(np.floor(R / P))
    M = np.full((2 * n + 1, 2 * n + 1), np.nan)
    M[np.round(ys / P).astype(int) + n, np.round(xs / P).astype(int) + n] = r_site * 1e3
    fig, axm = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = axm.imshow(M, origin="lower", cmap="Blues", extent=[-(n + .5) * P, (n + .5) * P] * 2)
    fig.colorbar(im, ax=axm, label="pillar radius (nm)")
    axm.set(xlabel="x (µm)", ylabel="y (µm)", title="Pillar radius map")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "layout_map.png"))
    plt.close(fig)

    # ---- focusing check -------------------------------------------------------
    if focus:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
        norm = focus["Ix0"].max()
        win = np.abs(focus["x"]) <= 8 * focus["fwhm_ideal"]
        ax[0].plot(focus["x"][win], focus["Ix0"][win] / norm, "--", c=MUTED, lw=1.5,
                   label="ideal lens")
        ax[0].plot(focus["x"][win], focus["Ix"][win] / norm, "-", c=BLUE, lw=2,
                   label=f"LUT design (Strehl {focus['strehl']:.2f})")
        ax[0].set(xlabel="x at z = f (µm)", ylabel="intensity / ideal peak",
                  title=f"Focal-plane cut, FWHM {focus['fwhm']:.2f} µm "
                        f"(ideal {focus['fwhm_ideal']:.2f} µm)")
        ax[0].legend(frameon=False, fontsize=8)
        nz = focus["Iax0"].max()
        ax[1].plot(focus["z"], focus["Iax0"] / nz, "--", c=MUTED, lw=1.5, label="ideal lens")
        ax[1].plot(focus["z"], focus["Iax"] / nz, "-", c=BLUE, lw=2, label="LUT design")
        ax[1].axvline(f, c=MUTED, lw=0.8, ls=":")
        ax[1].set(xlabel="z (µm)", ylabel="on-axis intensity / ideal peak",
                  title=f"On-axis intensity (peak at z = {focus['z_peak']:.0f} µm)")
        ax[1].legend(frameon=False, fontsize=8)
        for x in ax:
            grid(x)
        fig.suptitle(title + " — scalar angular-spectrum check")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "focus.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
