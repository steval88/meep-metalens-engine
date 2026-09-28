#!/usr/bin/env python3
"""
Meep LUT: transmission and phase vs nanocylinder radius.

Unit cell: square lattice (period P), one high-index cylinder (n_pil) of
height H standing on a semi-infinite substrate (n_sub), air above.
Normal-incidence x-polarised plane wave launched from inside the substrate.
Single wavelength (DFT at f = 1/lambda of a narrow Gaussian pulse).

Units: micrometres.  Meep time convention exp(-i*omega*t), so extra optical
path gives a POSITIVE phase increase.

Outputs (written by MPI rank 0 only):
  <out>.csv : r_nm, T, T_rel_substrate, phase_rad, phase_wrapped_rad, T0_check
  <out>.png : T(r) and phase(r)

Run:
  python lut_cylinders.py --height 1.0
  mpirun -np 4 python lut_cylinders.py --height 1.0
"""
import argparse

import numpy as np
import meep as mp

ap = argparse.ArgumentParser()
ap.add_argument("--wl", type=float, default=1.55, help="wavelength (um)")
ap.add_argument("--period", type=float, default=0.70, help="lattice period (um)")
ap.add_argument("--height", type=float, default=1.0, help="pillar height (um)")
ap.add_argument("--n_sub", type=float, default=1.44)
ap.add_argument("--n_pil", type=float, default=3.45)
ap.add_argument("--rmin", type=float, default=0.050)
ap.add_argument("--rmax", type=float, default=0.275)
ap.add_argument("--dr", type=float, default=0.025)
ap.add_argument("--res", type=int, default=50, help="pixels per um")
ap.add_argument("--no_sym", action="store_true", help="disable mirror symmetries")
ap.add_argument("--out", default=None, help="output file prefix")
args = ap.parse_args()

wl, P, H = args.wl, args.period, args.height
fcen = 1.0 / wl
df = 0.2 * fcen
radii = np.round(np.arange(args.rmin, args.rmax + 0.5 * args.dr, args.dr), 6)
out = args.out or f"lut_H{int(round(H * 1000))}nm_res{args.res}"

sub = mp.Medium(index=args.n_sub)
pil = mp.Medium(index=args.n_pil)

# ---- vertical layout (z) -------------------------------------------------
dpml = 1.0   # PML thickness (> lambda in substrate, 1.08 um)
dsub = 1.0   # substrate between bottom PML and pillar base
dair = 1.5   # air between pillar top and top PML
sz = dpml + dsub + H + dair + dpml
z_int = -0.5 * sz + dpml + dsub          # substrate/air interface = pillar base
z_src = z_int - 0.5 * dsub               # source plane, inside substrate
z_mon = z_int + H + 1.0                  # monitor plane, in air, 1 um above pillar top
# Evanescent (1,0) order in air decays as exp(-8.0 z[um]) for P=0.7, wl=1.55,
# so at 1 um the non-propagating orders are < 1e-3 in amplitude.

cell = mp.Vector3(P, P, sz)
pml = [mp.PML(dpml, direction=mp.Z)]
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(0, 0, z_src),
        size=mp.Vector3(P, P, 0),
    )
]
# Ex plane wave + cylinder: odd under x-mirror, even under y-mirror.
symmetries = [] if args.no_sym else [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y)]

substrate = mp.Block(
    size=mp.Vector3(mp.inf, mp.inf, dpml + dsub),
    center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * (dpml + dsub)),
    material=sub,
)
mon_center = mp.Vector3(0, 0, z_mon)
mon_size = mp.Vector3(P, P, 0)


def run(geometry, default_material=mp.air):
    """Return (flux, mean complex Ex over monitor plane) at fcen."""
    sim = mp.Simulation(
        cell_size=cell,
        resolution=args.res,
        boundary_layers=pml,
        geometry=geometry,
        default_material=default_material,
        sources=sources,
        symmetries=symmetries,
        k_point=mp.Vector3(),  # periodic in x,y, normal incidence
    )
    flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=mon_center, size=mon_size))
    dft = sim.add_dft_fields([mp.Ex], fcen, 0, 1, center=mon_center, size=mon_size)
    sim.run(
        until_after_sources=mp.stop_when_dft_decayed(
            tol=1e-8, minimum_run_time=50, maximum_run_time=2000
        )
    )
    f = mp.get_fluxes(flux)[0]
    ex = np.mean(sim.get_dft_array(dft, mp.Ex, 0))
    sim.reset_meep()
    return f, ex


# 1) incident power: homogeneous substrate everywhere
P_inc, _ = run([], default_material=sub)
# 2) bare substrate (no pillar): phase reference and Fresnel transmission
P_sub, E_sub = run([substrate])
T_sub = P_sub / P_inc

T, T0, phi = [], [], []
for r in radii:
    cyl = mp.Cylinder(
        radius=r,
        height=H,
        axis=mp.Vector3(0, 0, 1),
        center=mp.Vector3(0, 0, z_int + 0.5 * H),
        material=pil,
    )
    f, e = run([substrate, cyl])
    T.append(f / P_inc)
    T0.append(T_sub * abs(e) ** 2 / abs(E_sub) ** 2)  # zeroth-order check
    phi.append(np.angle(e / E_sub))
    if mp.am_master():
        print(f"LUT r={r*1e3:6.1f} nm  T={T[-1]:.4f}  phase={phi[-1]:+.4f} rad", flush=True)

T, T0 = np.array(T), np.array(T0)
phi_unw = np.unwrap(np.array(phi))
phi_wrap = np.mod(phi_unw - phi_unw[0], 2 * np.pi)

if mp.am_master():
    hdr = (
        f"wl={wl} um, P={P} um, H={H} um, n_sub={args.n_sub}, n_pil={args.n_pil}, "
        f"res={args.res}/um, T_bare_substrate={T_sub:.4f}\n"
        "r_nm,T,T_rel_substrate,phase_rad(unwrapped, vs bare substrate),"
        "phase_wrapped_rad(rel. to r_min),T0_check"
    )
    np.savetxt(
        out + ".csv",
        np.column_stack([radii * 1e3, T, T / T_sub, phi_unw, phi_wrap, T0]),
        delimiter=",", header=hdr, fmt="%.6f",
    )
    span = phi_unw.max() - phi_unw.min()
    print(f"LUT bare-substrate T = {T_sub:.4f}")
    print(f"LUT phase span = {span:.3f} rad ({span / (2*np.pi):.2f} x 2pi), "
          f"T min/mean = {T.min():.3f}/{T.mean():.3f}")
    print(f"LUT max |T - T0| = {np.max(np.abs(T - T0)):.2e}  (should be small)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax[0].plot(radii * 1e3, T, "o-")
    ax[0].axhline(T_sub, ls="--", c="gray", lw=1, label="bare substrate")
    ax[0].set(xlabel="radius (nm)", ylabel="transmission", ylim=(0, 1.02),
              title="Transmission")
    ax[0].legend(loc="lower left")
    ax[1].plot(radii * 1e3, phi_unw - phi_unw[0], "s-", c="C3")
    ax[1].set(xlabel="radius (nm)", ylabel="phase (rad)",
              title="Phase (rel. to r_min, unwrapped)")
    ax[1].axhline(2 * np.pi, ls="--", c="gray", lw=1)
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"Si-like pillars n={args.n_pil}, H={H*1e3:.0f} nm, "
                 f"P={P*1e3:.0f} nm, lambda={wl*1e3:.0f} nm, res={args.res}/um")
    fig.tight_layout()
    fig.savefig(out + ".png")
