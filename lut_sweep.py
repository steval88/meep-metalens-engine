#!/usr/bin/env python3
"""
lut_sweep.py - Meep unit-cell look-up tables (LUTs) for nanocylinder metasurfaces.

Nested sweeps:   resolution  x  pillar height H  x  pillar radius r
  * several --res values   -> mesh-convergence study
  * several --heights      -> height sweep
  * radii from --rmin/--rmax/--dr (or an explicit --radii list)

Physics (unchanged from lut_cylinders.py)
  Square lattice, period P, Bloch-periodic x/y with k = 0 (normal incidence),
  PML along z. One cylinder (n_pil, height H) on a semi-infinite substrate
  (n_sub), air above. Ex plane wave launched inside the substrate, single
  wavelength = DFT at f = 1/lambda of a Gaussian pulse.
  Units: um. Meep convention exp(-i w t): more optical path -> larger phase.

Quantities per simulation
  T      transmitted flux (air, 1 um above pillar top) / incident flux
  R      reflected flux (substrate, below the interface; incident fields
         subtracted with load_minus_flux_data) / incident flux
  R+T    energy-conservation check (lossless -> 1 up to discretisation error)
  T0     |mean Ex|^2-based zeroth-order transmission (cross-check of T)
  phase  arg(mean Ex / mean Ex_bare_substrate) at the transmission plane

Output layout (one time-stamped folder per execution)
  <root>/<YYYYmmdd-HHMMSS>[_tag]/
      config.json                       all parameters, Meep version, #procs
      logs/group<g>.log                 progress of each process group
      jobs/res<RRR>/H<hhhh>nm/norm.json incident + bare-substrate reference
      jobs/res<RRR>/H<hhhh>nm/r<rrrr>nm.json   one record per simulation
      results/res<RRR>/H<hhhh>nm/lut.csv, lut.png
      summary/heights_res<RRR>.csv/.png  (if >1 height)
      summary/convergence.csv/.png       (if >1 resolution)

Examples
  # single LUT
  mpirun -np 4 python lut_sweep.py --heights 1.0 --res 50
  # height sweep, 16 procs split into 8 independent groups of 2
  mpirun -np 16 python lut_sweep.py --heights 0.7 0.8 0.9 1.0 1.1 1.2 1.3 --groups 8
  # mesh convergence at H = 1 um
  mpirun -np 16 python lut_sweep.py --heights 1.0 --res 30 40 50 60 70 --groups 8
  # resume an interrupted run (finished simulations are skipped)
  mpirun -np 16 python lut_sweep.py --resume runs/20260928-101500 --groups 8
  # (re)build tables/plots only, no simulation
  python lut_sweep.py --resume runs/20260928-101500 --summarize_only
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import meep as mp

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ImportError:  # serial Meep build
    COMM = None


# --------------------------------------------------------------------------- #
# arguments / run folder
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wl", type=float, default=1.55, help="wavelength (um)")
    ap.add_argument("--period", type=float, default=0.70, help="lattice period (um)")
    ap.add_argument("--heights", type=float, nargs="+", default=[1.0], help="pillar heights (um)")
    ap.add_argument("--res", type=int, nargs="+", default=[50], help="resolution(s), px/um")
    ap.add_argument("--rmin", type=float, default=0.050)
    ap.add_argument("--rmax", type=float, default=0.275)
    ap.add_argument("--dr", type=float, default=0.025)
    ap.add_argument("--radii", type=float, nargs="+", default=None,
                    help="explicit radii (um); overrides rmin/rmax/dr")
    ap.add_argument("--n_sub", type=float, default=1.44)
    ap.add_argument("--n_pil", type=float, default=3.45)
    ap.add_argument("--dpml", type=float, default=1.0, help="PML thickness (um)")
    ap.add_argument("--tol", type=float, default=1e-8, help="DFT decay tolerance")
    ap.add_argument("--max_time", type=float, default=5000.0,
                    help="max run time after the source (Meep units). High-Q lattice "
                         "resonances near lambda can ring longer; such runs are "
                         "flagged hit_max_time and show R+T != 1")
    ap.add_argument("--no_sym", action="store_true", help="disable mirror symmetries")
    ap.add_argument("--groups", type=int, default=1,
                    help="split MPI processes into this many independent groups")
    ap.add_argument("--root", default="runs", help="parent folder for run folders")
    ap.add_argument("--tag", default="", help="suffix for the run-folder name")
    ap.add_argument("--resume", default=None, help="existing run folder to continue")
    ap.add_argument("--summarize_only", action="store_true",
                    help="only rebuild results/ and summary/ from jobs/")
    ap.add_argument("--quiet", action="store_true", help="suppress Meep's own output")
    return ap.parse_args()


SWEEP_KEYS = ["wl", "period", "heights", "res", "rmin", "rmax", "dr", "radii", "n_sub",
              "n_pil", "dpml", "tol", "max_time", "no_sym"]


def bcast(obj):
    """Broadcast from global rank 0 (all ranks must agree on paths)."""
    return COMM.bcast(obj, root=0) if COMM is not None and COMM.Get_size() > 1 else obj


def barrier():
    if COMM is not None and COMM.Get_size() > 1:
        COMM.Barrier()


def setup_run_dir(args):
    if args.resume:
        run_dir = args.resume.rstrip("/")
        with open(os.path.join(run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        for k in SWEEP_KEYS:              # physics/sweep come from the original run
            setattr(args, k, cfg["args"][k])
        return run_dir
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = stamp + (f"_{args.tag}" if args.tag else "")
    run_dir = bcast(os.path.join(args.root, name))
    if mp.am_really_master():
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        cfg = {
            "args": {k: getattr(args, k) for k in SWEEP_KEYS},
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "meep_version": mp.__version__,
            "nprocs": mp.count_processors(),
            "groups": args.groups,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)
    barrier()
    return run_dir


def radii_list(args):
    if args.radii:
        return [round(r, 6) for r in args.radii]
    return list(np.round(np.arange(args.rmin, args.rmax + 0.5 * args.dr, args.dr), 6))


def job_dir(run_dir, res, H):
    return os.path.join(run_dir, "jobs", f"res{res:03d}", f"H{round(H * 1000):04d}nm")


def job_file(run_dir, res, H, r):
    return os.path.join(job_dir(run_dir, res, H), f"r{r * 1000:07.2f}nm.json")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{mp.my_rank()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)          # atomic: no half-written records on crash


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
class UnitCell:
    """Geometry/layout for one (resolution, height) pair."""

    DSUB = 1.2   # substrate between bottom PML and pillar base (um)
    DAIR = 1.5   # air between pillar top and top PML (um)

    def __init__(self, args, res, H):
        self.a, self.res, self.H = args, res, H
        self.P = args.period
        self.fcen = 1.0 / args.wl
        self.sub = mp.Medium(index=args.n_sub)
        self.pil = mp.Medium(index=args.n_pil)
        d = args.dpml
        # the lateral period must be an integer number of pixels, otherwise Meep
        # silently rounds the cell (i.e. changes the period!)
        if abs(self.P * res - round(self.P * res)) > 1e-6:
            raise SystemExit(f"period {self.P} um x resolution {res} = {self.P * res} px "
                             "is not an integer; choose another --res")
        sz = d + self.DSUB + H + self.DAIR + d
        self.sz = np.ceil(sz * res - 1e-9) / res    # pad the air gap to whole pixels
        self.z_int = -0.5 * self.sz + d + self.DSUB   # substrate/air interface
        self.z_src = self.z_int - 0.8                 # source, in substrate
        self.z_ref = self.z_int - 0.4                 # reflection monitor, in substrate
        self.z_tra = self.z_int + H + 1.0             # transmission monitor, in air
        self.substrate = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, d + self.DSUB),
            center=mp.Vector3(0, 0, -0.5 * self.sz + 0.5 * (d + self.DSUB)),
            material=self.sub)

    def cylinder(self, r):
        return mp.Cylinder(radius=r, height=self.H, axis=mp.Vector3(0, 0, 1),
                           center=mp.Vector3(0, 0, self.z_int + 0.5 * self.H),
                           material=self.pil)

    def simulation(self, geometry, default_material=mp.air):
        P = self.P
        return mp.Simulation(
            cell_size=mp.Vector3(P, P, self.sz),
            resolution=self.res,
            boundary_layers=[mp.PML(self.a.dpml, direction=mp.Z)],
            # Bloch-periodic x, y with k = 0 (normal incidence). Without k_point
            # the lateral walls would be perfect electric conductors.
            k_point=mp.Vector3(0, 0, 0),
            geometry=geometry,
            default_material=default_material,
            sources=[mp.Source(mp.GaussianSource(self.fcen, fwidth=0.2 * self.fcen),
                               component=mp.Ex,
                               center=mp.Vector3(0, 0, self.z_src),
                               size=mp.Vector3(P, P, 0))],
            # Ex plane wave + cylinder: odd in x, even in y about the cell centre
            symmetries=[] if self.a.no_sym else [mp.Mirror(mp.X, phase=-1),
                                                  mp.Mirror(mp.Y)],
        )

    def monitors(self, sim):
        P, f = self.P, self.fcen
        tra = sim.add_flux(f, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_tra),
                                                  size=mp.Vector3(P, P, 0)))
        ref = sim.add_flux(f, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, self.z_ref),
                                                  size=mp.Vector3(P, P, 0)))
        dft = sim.add_dft_fields([mp.Ex], f, 0, 1, center=mp.Vector3(0, 0, self.z_tra),
                                 size=mp.Vector3(P, P, 0))
        return tra, ref, dft

    def run(self, sim):
        t0 = time.time()
        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=self.a.tol, minimum_run_time=50, maximum_run_time=self.a.max_time))
        return sim.meep_time(), time.time() - t0

    def normalization(self):
        """Incident flux (homogeneous substrate) + bare-substrate reference."""
        # (a) substrate everywhere: incident power and incident fields at z_ref
        sim = self.simulation([], default_material=self.sub)
        tra, ref, _ = self.monitors(sim)
        t_a, w_a = self.run(sim)
        P_inc = mp.get_fluxes(tra)[0]
        P_inc_ref = mp.get_fluxes(ref)[0]
        inc_ref_data = sim.get_flux_data(ref)
        sim.reset_meep()
        # (b) bare substrate / air interface
        sim = self.simulation([self.substrate])
        tra, ref, dft = self.monitors(sim)
        sim.load_minus_flux_data(ref, inc_ref_data)
        t_b, w_b = self.run(sim)
        T_sub = mp.get_fluxes(tra)[0] / P_inc
        R_sub = -mp.get_fluxes(ref)[0] / P_inc
        E_sub = complex(np.mean(sim.get_dft_array(dft, mp.Ex, 0)))
        sim.reset_meep()
        n = self.a.n_sub
        T_fresnel = 1 - ((n - 1) / (n + 1)) ** 2
        rec = {
            "res": self.res, "H_um": self.H, "P_inc": P_inc,
            "P_inc_at_refl_plane": P_inc_ref,
            "T_sub": T_sub, "R_sub": R_sub, "RT_sub": R_sub + T_sub,
            "T_sub_fresnel_exact": T_fresnel, "T_sub_error": T_sub - T_fresnel,
            "E_sub": [E_sub.real, E_sub.imag],
            "t_sim": [t_a, t_b], "wall_s": w_a + w_b,
        }
        return rec, inc_ref_data

    def pillar(self, r, norm, inc_ref_data):
        sim = self.simulation([self.substrate, self.cylinder(r)])
        tra, ref, dft = self.monitors(sim)
        sim.load_minus_flux_data(ref, inc_ref_data)
        t_sim, wall = self.run(sim)
        P_inc = norm["P_inc"]
        T = mp.get_fluxes(tra)[0] / P_inc
        R = -mp.get_fluxes(ref)[0] / P_inc
        E = complex(np.mean(sim.get_dft_array(dft, mp.Ex, 0)))
        sim.reset_meep()
        E_sub = complex(*norm["E_sub"])
        return {
            "res": self.res, "H_um": self.H, "r_um": r,
            "T": T, "R": R, "RT": R + T,
            "T0": norm["T_sub"] * abs(E) ** 2 / abs(E_sub) ** 2,
            "phase_rad": float(np.angle(E / E_sub)),
            "E": [E.real, E.imag],
            "t_sim": t_sim, "wall_s": wall,
            "hit_max_time": bool(t_sim >= 0.98 * self.a.max_time),
        }


# --------------------------------------------------------------------------- #
# post-processing (global master only; reads jobs/*.json)
# --------------------------------------------------------------------------- #
def load_records(run_dir, res, H):
    d = job_dir(run_dir, res, H)
    if not os.path.isdir(d):
        return None, []
    norm_path = os.path.join(d, "norm.json")
    norm = json.load(open(norm_path)) if os.path.exists(norm_path) else None
    recs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))
            if f.startswith("r") and f.endswith(".json")]
    recs.sort(key=lambda x: x["r_um"])
    return norm, recs


def lut_table(recs):
    r = np.array([x["r_um"] for x in recs])
    T = np.array([x["T"] for x in recs])
    R = np.array([x["R"] for x in recs])
    T0 = np.array([x["T0"] for x in recs])
    ph = np.unwrap(np.array([x["phase_rad"] for x in recs]))
    return r, T, R, T0, ph


# categorical palette (validated for colour-vision deficiency, light surface);
# markers are the secondary encoding so no series relies on colour alone.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def series_style(i):
    """Fixed-order colour/marker for the i-th series (wraps after 8 with dashes)."""
    return dict(color=PALETTE[i % 8], marker=MARKERS[i % 8],
                ls="-" if i < 8 else "--")


def pi_label(v, _pos=None):
    """Axis label for a value given in units of pi: 0, π/2, π, 3π/2, 2π, ..."""
    n = int(round(2 * v))
    if n == 0:
        return "0"
    if n % 2 == 0:
        k = n // 2
        return "π" if k == 1 else ("−π" if k == -1 else f"{k}π")
    if n == 1:
        return "π/2"
    if n == -1:
        return "−π/2"
    return f"{n}π/2".replace("-", "−")


def fine_grid(ax, y_minor=True):
    """Major + minor grid (minor = 1/5 of the major spacing on each axis)."""
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if y_minor:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(True, which="major", color="#d6d6d6", lw=0.8)
    ax.grid(True, which="minor", color="#eeeeee", lw=0.5)


def summarize(args, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e3e3e3"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
    res_list, heights = sorted(args.res), list(args.heights)
    os.makedirs(os.path.join(run_dir, "summary"), exist_ok=True)
    luts = {}

    # ---- one LUT per (res, H) ---------------------------------------------
    for res in res_list:
        for H in heights:
            norm, recs = load_records(run_dir, res, H)
            if not recs or norm is None:
                continue
            r, T, R, T0, ph = lut_table(recs)
            luts[(res, H)] = (r, T, R, ph, norm)
            out = os.path.join(run_dir, "results", f"res{res:03d}", f"H{round(H*1000):04d}nm")
            os.makedirs(out, exist_ok=True)
            steps = np.abs(np.diff(ph))
            hdr = (f"wl={args.wl} um, P={args.period} um, H={H} um, n_sub={args.n_sub}, "
                   f"n_pil={args.n_pil}, res={res}/um\n"
                   f"bare substrate: T={norm['T_sub']:.6f} (Fresnel {norm['T_sub_fresnel_exact']:.6f}), "
                   f"R={norm['R_sub']:.6f}, R+T={norm['RT_sub']:.6f}\n"
                   "r_nm,T,R,R+T,T0_check,phase_rad_unwrapped,phase_rel_rmin_rad,phase_wrapped_rad,hit_max_time")
            np.savetxt(os.path.join(out, "lut.csv"),
                       np.column_stack([r * 1e3, T, R, R + T, T0, ph, ph - ph[0],
                                        np.mod(ph - ph[0], 2 * np.pi),
                                        [x["hit_max_time"] for x in recs]]),
                       delimiter=",", header=hdr, fmt="%.6f")

            title = (f"H = {H*1e3:.0f} nm, P = {args.period*1e3:.0f} nm, "
                     f"λ = {args.wl*1e3:.0f} nm, res = {res}/µm")
            blue = PALETTE[0]
            # (a) the LUT itself: transmission and phase
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
            ax[0].plot(r * 1e3, T, "o-", c=blue, lw=2, ms=5)
            ax[0].set(xlabel="radius (nm)", ylabel="T (fraction of incident power)",
                      ylim=(0, 1.05), title="Transmission")
            ax[1].plot(r * 1e3, ph - ph[0], "o-", c=blue, lw=2, ms=5)
            ax[1].axhline(2 * np.pi, ls="--", c=MUTED, lw=0.8)
            ax[1].set(xlabel="radius (nm)", ylabel="phase rel. to r_min (rad)",
                      title="Phase (unwrapped)")
            if steps.size and steps.max() > 2.5:
                ax[1].text(0.02, 0.95, "warning: |Δφ| > 2.5 rad between radii - refine dr",
                           transform=ax[1].transAxes, color="#b3261e", fontsize=8, va="top")
            for a in ax:
                fine_grid(a)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "lut.png"))
            plt.close(fig)

            # (a') phase in units of pi (same data, pi-labelled axis)
            fig, a = plt.subplots(figsize=(6.5, 4.5), dpi=150)
            a.plot(r * 1e3, (ph - ph[0]) / np.pi, "o-", c=blue, lw=2, ms=5)
            a.axhline(2, ls="--", c=MUTED, lw=0.8)
            top = max(2.0, np.ceil(2 * (ph - ph[0]).max() / np.pi) / 2)
            a.set_ylim(min(0, (ph - ph[0]).min() / np.pi) - 0.1, top + 0.1)
            a.yaxis.set_major_locator(MultipleLocator(0.5))
            a.yaxis.set_major_formatter(FuncFormatter(pi_label))
            a.set(xlabel="radius (nm)", ylabel="phase rel. to r_min",
                  title="Phase (unwrapped, multiples of π)")
            fine_grid(a, y_minor=False)
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "lut_phase_pi.png"))
            plt.close(fig)

            # (b) reflection and energy-conservation check, separate figure
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
            ax[0].plot(r * 1e3, R, "s-", c=PALETTE[1], lw=2, ms=5)
            ax[0].axhline(norm["R_sub"], ls="--", c=MUTED, lw=0.8)
            ax[0].set(xlabel="radius (nm)", ylabel="R (fraction of incident power)",
                      ylim=(0, max(0.1, 1.1 * R.max())), title="Reflection")
            ax[1].plot(r * 1e3, R + T - 1, "o-", c=MUTED, lw=1.5, ms=4)
            ax[1].axhline(0, c=MUTED, lw=0.8)
            flag = np.array([x["hit_max_time"] for x in recs], bool)
            if flag.any():
                ax[1].plot(r[flag] * 1e3, (R + T - 1)[flag], "x", c="#b3261e", ms=9,
                           label="hit max_time")
                ax[1].legend(frameon=False, fontsize=8)
            ax[1].set(xlabel="radius (nm)", ylabel="R + T − 1",
                      title="Energy conservation check (should be ≈ 0)")
            for a in ax:
                fine_grid(a)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "lut_reflection.png"))
            plt.close(fig)

    # ---- height sweep summary (per resolution) ----------------------------
    for res in res_list:
        Hs = [H for H in heights if (res, H) in luts]
        if len(Hs) < 2:
            continue
        rows = []
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
        figR, axR = plt.subplots(figsize=(6.5, 4.5), dpi=150)
        for i, H in enumerate(Hs):
            r, T, R, ph, norm = luts[(res, H)]
            span = ph.max() - ph.min()
            rows.append([H * 1e3, span, span / (2 * np.pi), T.min(), T.mean(),
                         np.max(np.abs(R + T - 1)), r[np.argmin(T)] * 1e3])
            st = series_style(i)
            lab = f"{H*1e3:.0f} nm"
            ax[0].plot(r * 1e3, T, lw=1.6, ms=5, label=lab, **st)
            ax[1].plot(r * 1e3, ph - ph[0], lw=1.6, ms=5, label=lab, **st)
            axR.plot(r * 1e3, R, lw=1.6, ms=5, label=lab, **st)
        ax[0].set(xlabel="radius (nm)", ylabel="T", ylim=(0, 1.05), title="Transmission")
        ax[1].axhline(2 * np.pi, ls="--", c=MUTED, lw=0.8)
        ax[1].set(xlabel="radius (nm)", ylabel="phase rel. to r_min (rad)",
                  title="Phase (unwrapped)")
        # one legend for both panels, outside the data area
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, title="H", frameon=False, fontsize=8, loc="center right")
        fig.suptitle(f"Height sweep, res = {res}/µm")
        fig.tight_layout(rect=(0, 0, 0.9, 1))
        fig.savefig(os.path.join(run_dir, "summary", f"heights_res{res:03d}.png"))
        plt.close(fig)
        axR.set(xlabel="radius (nm)", ylabel="R", ylim=(0, 1.05),
                title=f"Reflection, height sweep, res = {res}/µm")
        axR.legend(title="H", frameon=False, fontsize=8)
        figR.tight_layout()
        figR.savefig(os.path.join(run_dir, "summary", f"heights_reflection_res{res:03d}.png"))
        plt.close(figR)
        np.savetxt(os.path.join(run_dir, "summary", f"heights_res{res:03d}.csv"),
                   np.array(rows), delimiter=",", fmt="%.6f",
                   header="H_nm,phase_span_rad,phase_span_over_2pi,T_min,T_mean,max_abs_RT_minus_1,r_at_T_min_nm")

    # ---- convergence summary (per height) ---------------------------------
    if len(res_list) >= 2:
        rows = []
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
        for k, H in enumerate(heights):
            have = [res for res in res_list if (res, H) in luts]
            if len(have) < 2:
                continue
            st = series_style(k)
            rf = have[-1]
            r_f, T_f, _, ph_f, _ = luts[(rf, H)]
            dT, dph, rt, tsub = [], [], [], []
            for res in have:
                r, T, R, ph, norm = luts[(res, H)]
                common = np.intersect1d(np.round(r, 6), np.round(r_f, 6))
                i1 = np.isin(np.round(r, 6), common)
                i2 = np.isin(np.round(r_f, 6), common)
                d_ph = (ph[i1] - ph[i1][0]) - (ph_f[i2] - ph_f[i2][0])
                dT.append(np.max(np.abs(T[i1] - T_f[i2])))
                dph.append(np.max(np.abs(d_ph)))
                rt.append(np.max(np.abs(R + T - 1)))
                tsub.append(abs(norm["T_sub_error"]))
                rows.append([H * 1e3, res, dT[-1], dph[-1], rt[-1], tsub[-1]])
            ax[0].plot(have[:-1], dT[:-1], label=f"H={H*1e3:.0f} nm", **st)
            ax[1].plot(have[:-1], dph[:-1], **st)
            ax[2].plot(have, rt, label=f"max|R+T-1|, H={H*1e3:.0f}", **st)
            ax[2].plot(have, tsub, "s--", c=MUTED, label="|T_sub - Fresnel|")
        ax[0].set(xlabel="resolution (px/µm)", ylabel=f"max |T - T(res={res_list[-1]})|",
                  yscale="log", title="Transmission vs finest mesh")
        ax[1].set(xlabel="resolution (px/µm)", ylabel=f"max |φ - φ(res={res_list[-1]})| (rad)",
                  yscale="log", title="Phase vs finest mesh")
        ax[2].set(xlabel="resolution (px/µm)", ylabel="absolute error", yscale="log",
                  title="Energy conservation / exact benchmark")
        ax[0].legend(frameon=False, fontsize=7)
        ax[2].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "summary", "convergence.png"))
        plt.close(fig)
        np.savetxt(os.path.join(run_dir, "summary", "convergence.csv"), np.array(rows),
                   delimiter=",", fmt="%.6e",
                   header="H_nm,res,max_abs_dT_vs_finest,max_abs_dphase_vs_finest_rad,"
                          "max_abs_RT_minus_1,abs_Tsub_minus_Fresnel")

        # overlay of LUT curves for every resolution (one figure per height)
        for H in heights:
            have = [res for res in res_list if (res, H) in luts]
            if len(have) < 2:
                continue
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
            for i, res in enumerate(have):
                r, T, R, ph, _ = luts[(res, H)]
                st = series_style(i)
                ax[0].plot(r * 1e3, T, lw=1.5, ms=4, label=f"res {res}", **st)
                ax[1].plot(r * 1e3, ph - ph[0], lw=1.5, ms=4, **st)
            ax[0].set(xlabel="radius (nm)", ylabel="T", ylim=(0, 1.05), title="Transmission")
            ax[0].legend(frameon=False, fontsize=7, loc="lower left")
            ax[1].set(xlabel="radius (nm)", ylabel="phase rel. to r_min (rad)", title="Phase")
            fig.suptitle(f"Mesh convergence, H = {H*1e3:.0f} nm")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, "summary", f"convergence_H{round(H*1000):04d}nm.png"))
            plt.close(fig)
    print(f"[summary] written to {os.path.join(run_dir, 'results')} and "
          f"{os.path.join(run_dir, 'summary')}", flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
T_START = time.time()


def main():
    args = parse_args()
    if args.quiet:
        mp.verbosity(0)
    if mp.count_processors() > 1 and COMM is None:
        raise SystemExit("mpi4py is required for MPI runs: conda install -c conda-forge mpi4py")
    run_dir = setup_run_dir(args)

    if args.summarize_only:
        if mp.am_really_master():
            summarize(args, run_dir)
        return

    radii = radii_list(args)
    jobs = [(res, H, r) for res in args.res for H in args.heights for r in radii]
    # every rank decides from the same files -> identical job lists everywhere
    todo = [j for j in jobs if not os.path.exists(job_file(run_dir, *j))]
    # dynamic scheduling: groups claim jobs one at a time from a shared queue.
    # Largest radii first: the slow, resonant runs start early instead of
    # piling up at the end of one group's list.
    todo.sort(key=lambda j: (-j[2], j[0], j[1]))
    claim_dir = os.path.join(run_dir, "claims")
    if mp.am_really_master():
        os.makedirs(claim_dir, exist_ok=True)
        for f in os.listdir(claim_dir):          # stale claims of an interrupted run
            os.remove(os.path.join(claim_dir, f))
    barrier()

    G = args.groups
    nproc = mp.count_processors()
    if G > 1:
        if nproc % G:
            raise SystemExit(f"--groups {G} must divide the number of MPI processes ({nproc})")
        gid = mp.divide_parallel_processes(G)
    else:
        gid = 0

    log_path = os.path.join(run_dir, "logs", f"group{gid:02d}.log")

    def log(msg):
        if mp.am_master():                       # master of this group
            line = f"{datetime.datetime.now():%H:%M:%S} [g{gid}] {msg}"
            # Meep redirects sys.stdout to /dev/null on every rank except the
            # global master; sys.__stdout__ is the real terminal for all ranks.
            print(line, file=sys.__stdout__, flush=True)
            with open(log_path, "a") as fh:
                fh.write(line + "\n")

    def claim(job):
        """True if this group wins the job. The group master creates the claim
        file atomically (O_EXCL); after a group barrier every rank of the group
        reads the owner from the file, so the whole group agrees."""
        res, H, r = job
        path = os.path.join(claim_dir, f"res{res:03d}_H{round(H*1000):04d}_r{r*1000:07.2f}")
        if mp.am_master():
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(gid).encode())
                os.close(fd)
            except FileExistsError:
                pass
        mp.all_wait()                            # barrier within this group only
        try:
            with open(path) as fh:
                return fh.read().strip() == str(gid)
        except FileNotFoundError:
            return False

    log(f"{len(jobs)} jobs total, {len(todo)} to do, {G} groups x {nproc // G} procs, "
        f"dynamic queue (largest radius first)")
    cache = {}
    mine = []
    for (res, H, r) in todo:
        if not claim((res, H, r)):
            continue
        mine.append((res, H, r))
        cell = UnitCell(args, res, H)
        if (res, H) not in cache:
            norm, inc = cell.normalization()
            cache[(res, H)] = (norm, inc)
            if mp.am_master():
                write_json(os.path.join(job_dir(run_dir, res, H), "norm.json"), norm)
            log(f"norm res={res} H={H*1e3:.0f}nm  T_sub={norm['T_sub']:.5f} "
                f"(Fresnel {norm['T_sub_fresnel_exact']:.5f})  R+T={norm['RT_sub']:.5f}")
        norm, inc = cache[(res, H)]
        rec = cell.pillar(r, norm, inc)
        if mp.am_master():
            write_json(job_file(run_dir, res, H, r), rec)
        log(f"res={res} H={H*1e3:.0f}nm r={r*1e3:6.1f}nm  T={rec['T']:.5f} R={rec['R']:.5f} "
            f"R+T={rec['RT']:.5f} phase={rec['phase_rad']:+.4f}  "
            f"({rec['wall_s']:.0f}s{', MAX TIME HIT' if rec['hit_max_time'] else ''})")

    log(f"group finished its {len(mine)} simulations; waiting for the other groups")
    if G > 1:
        mp.begin_global_communications()
    barrier()
    if mp.am_really_master():
        el = time.time() - T_START
        print(f"{datetime.datetime.now():%H:%M:%S} all {G} groups finished; "
              f"wall time {el/60:.1f} min", file=sys.__stdout__, flush=True)
        summarize(args, run_dir)
    if G > 1:
        mp.end_global_communications()


if __name__ == "__main__":
    main()
