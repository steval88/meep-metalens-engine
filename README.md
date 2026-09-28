# meep-metalens-engine

Meep (FDTD) engine for unit-cell look-up tables (LUTs) of dielectric
nanocylinder metasurfaces, plus the target phase profile of a metalens.

| File | Purpose |
|---|---|
| `lut_sweep.py` | LUT engine: T, R, phase vs pillar radius; nested sweeps over resolution × height × radius; one time-stamped folder per execution |
| `metalens_phase_profile.py` | Hyperbolic phase profile of the lens, sampled on the lattice |
| `legacy/lut_cylinders.py` | First single-height version (kept for reproducibility of early results) |
| `environment.yml` | conda environment (Linux/WSL, conda-forge, MPI build of Meep) |

## Default design point
λ = 1550 nm, square lattice P = 700 nm, pillar n = 3.45, substrate n = 1.44,
radius 50–275 nm (step 25 nm), height 1.0 µm; normal-incidence x-polarised plane wave from the substrate.
Lens: D = 100 µm, f/3 (f = 300 µm, NA = 0.164).

## Setup (WSL Ubuntu)
```bash
conda env create -f environment.yml
conda activate meep
```

## Usage
```bash
# one LUT
mpirun -np 4 python lut_sweep.py --heights 1.0 --res 50 --quiet
# height sweep: N = number of physical cores, one independent simulation per process
mpirun -np N python lut_sweep.py --heights 0.7 0.8 0.9 1.0 1.1 1.2 1.3 --groups N --quiet --tag heights
# mesh convergence
mpirun -np N python lut_sweep.py --heights 1.0 --res 30 40 50 60 70 --groups N --quiet --tag conv
# resume / rebuild plots
mpirun -np N python lut_sweep.py --resume runs/<folder> --groups N
python lut_sweep.py --resume runs/<folder> --summarize_only
```
Output goes to `runs/<YYYYmmdd-HHMMSS>_<tag>/` (`config.json`, `jobs/`, `results/`, `summary/`, `logs/`).
`runs/` is git-ignored.

## Validation notes (sandbox, 2026-09-28)
* Bare substrate: T − T_Fresnel = 1.5e-4 at res 50; R + T = 1 within 1e-6 (res ≥ 40, non-resonant cases).
* Mesh convergence (H = 1 µm, r = 100/150 nm, vs res 80): |ΔT| ≤ 3e-4, |Δφ| ≤ 2e-3 rad at res 50.
* Mirror symmetries (odd x, even y) reproduce the full-cell result exactly.
* MPI `--groups` gives results identical to serial runs.

## Known pitfalls
* `P × res` must be an integer (use res = 30, 40, 50, …); otherwise Meep rounds the cell and changes the period. The script refuses such values.
* High-Q lattice resonances near λ (e.g. f = 0.6516, Q ≈ 1.5e4 at H = 1 µm, r ≈ 200 nm) ring for >10⁴ time units.
  Runs truncated at `--max_time` are flagged `hit_max_time`; check `R+T` in `lut.csv` before trusting T there.
* On WSL, use at most as many MPI processes as physical cores: oversubscription made each step about 20× slower.
