#!/usr/bin/env python3
"""
Hyperbolic (aberration-free, on-axis) phase profile of a metalens.

  f       = N * D
  phi(r)  = (2*pi/lambda) * ( sqrt(r_max^2 + f^2) - sqrt(r^2 + f^2) )     [rad]

phi is written as a phase DELAY (>= 0, maximum at the centre, 0 at the rim),
the same sign convention as the Meep LUT (lut_cylinders.py: more optical path
-> larger phase). Wrapped to [0, 2*pi) and sampled at the meta-atom centres
r_j = j * P of a square lattice with period P.

Outputs: metalens_phase_profile.csv / .png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters (um) -------------------------------------------------------
D = 100.0            # lens diameter
N = 3.0              # f-number
wl = 1.55            # design wavelength
P = 0.70             # lattice period = LUT period

R = 0.5 * D
f = N * D
k = 2 * np.pi / wl
NA = np.sin(np.arctan(R / f))


def phase_delay(r):
    return k * (np.sqrt(R**2 + f**2) - np.sqrt(r**2 + f**2))


# dense curve and meta-atom samples
r = np.linspace(0, R, 5001)
phi = phase_delay(r)
rj = np.arange(0, R + 1e-9, P)
phij = phase_delay(rj)
phij_w = np.mod(phij, 2 * np.pi)

# ---- design figures of merit ----------------------------------------------
n_zones = phi[0] / (2 * np.pi)
dphi_cell_max = np.max(np.abs(np.diff(phij)))            # worst step between atoms
edge_zone = wl / NA                                      # local zone width at rim ~ lambda/sin(theta)
print(f"f = {f:.1f} um, NA = {NA:.4f}")
print(f"total phase delay (centre) = {phi[0]:.3f} rad = {n_zones:.2f} x 2pi")
print(f"meta-atoms along radius   = {rj.size} (P = {P} um)")
print(f"max phase step per cell   = {dphi_cell_max:.3f} rad "
      f"(-> {2*np.pi/dphi_cell_max:.1f} atoms per 2pi zone at the rim)")
print(f"zone width at rim ~ {edge_zone:.2f} um")
print(f"Nyquist: P = {P} um < lambda/(2 NA) = {wl/(2*NA):.2f} um -> "
      f"{'OK' if P < wl/(2*NA) else 'VIOLATED'}")
print(f"diffraction-limited FWHM ~ lambda/(2NA) = {wl/(2*NA):.2f} um")

np.savetxt(
    "metalens_phase_profile.csv",
    np.column_stack([rj, phij, phij_w]),
    delimiter=",", fmt="%.6f",
    header=(f"D={D} um, N={N}, f={f} um, lambda={wl} um, P={P} um, NA={NA:.4f}\n"
            "r_um,phase_delay_unwrapped_rad,phase_delay_wrapped_rad"),
)

# ---- plot ------------------------------------------------------------------
ink, muted, grid = "#1f1f1f", "#6b6b6b", "#e3e3e3"
c1 = "#2a6fdb"
plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

ax[0].plot(r, phi, c=c1, lw=2)
ax[0].set(xlabel="radius r (µm)", ylabel="phase delay (rad)",
          title="Unwrapped hyperbolic phase", xlim=(0, R), ylim=(0, None))
ax[0].grid(True, c=grid, lw=0.8)
for m in range(1, int(n_zones) + 1):
    ax[0].axhline(2 * np.pi * m, c=muted, lw=0.6, ls=":")

phi_w = np.mod(phi, 2 * np.pi)
phi_w[1:][np.abs(np.diff(phi_w)) > np.pi] = np.nan   # no vertical lines at 2pi wraps
ax[1].plot(r, phi_w, c=muted, lw=1, alpha=0.7, label="continuous")
ax[1].plot(rj, phij_w, "o", ms=3.5, c=c1, label=f"meta-atom sites (P = {P*1e3:.0f} nm)")
ax[1].set(xlabel="radius r (µm)", ylabel="phase delay mod 2π (rad)",
          title="Wrapped phase, sampled at lattice sites",
          xlim=(0, R), ylim=(0, 2 * np.pi))
ax[1].set_yticks(np.arange(0, 2 * np.pi + 0.01, np.pi / 2),
                 ["0", "π/2", "π", "3π/2", "2π"])
ax[1].grid(True, c=grid, lw=0.8)
ax[1].legend(loc="lower left", frameon=False, fontsize=8)

fig.suptitle(f"Metalens D = {D:.0f} µm, f/{N:g} (f = {f:.0f} µm, NA = {NA:.3f}), "
             f"λ = {wl*1e3:.0f} nm", color=ink)
fig.tight_layout()
fig.savefig("metalens_phase_profile.png")
