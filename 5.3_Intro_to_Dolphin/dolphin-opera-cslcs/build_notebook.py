"""
Build dolphin-three-sisters-cslcs.ipynb from a list of cells.

The notebook is the artifact students open. This script is the editable
source — edit cells here, rerun this script to regenerate the .ipynb.
Outputs are NOT auto-executed; run the notebook end-to-end (or
`jupyter nbconvert --execute`) to populate outputs.

All dolphin calls are inline in the notebook (no helper-module imports)
so students see exactly what's being asked of dolphin.

Usage:
    /Users/zmhoppinen/miniforge3/envs/thp/bin/python build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "dolphin-three-sisters-cslcs.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ===========================================================================
# Section 0 — Setup
# ===========================================================================

md(r"""
# 5.3 Intro to dolphin — Three Sisters volcano (OPERA CSLC)

*EarthScope 2026 ISCE+ course — Zach Hoppinen & Scott Staniewicz*

### What is dolphin?

[dolphin](https://github.com/isce-framework/dolphin) is an open-source
Python pipeline that takes a stack of coregistered SAR SLCs and
produces a displacement time series. It packages persistent-scatterer
detection, distributed-scatterer phase linking, interferogram
formation, unwrapping, and time-series inversion behind one CLI and a
Python API. It's also the processing chain NASA's OPERA project uses
for the DISP-S1 displacement product — see
[Staniewicz et al. 2025](https://arxiv.org/abs/2511.12051) for the
full algorithm description.

![dolphin pipeline](dolphin_workflow_figure3.png)

*Figure 3 from Staniewicz et al. 2025 — the dolphin pipeline. Each
stage in this notebook lines up with one block in this diagram.*

### Study area

South Sister volcano in central Oregon has been inflating at a few
mm/yr since the late 1990s. We'll use dolphin to recover that signal
from Sentinel-1 OPERA CSLCs.

We use track 115, ascending, with two adjacent bursts covering the
volcano: `T115-245676-IW2` and `T115-245677-IW2`. 105 acquisitions
2016-07 to 2024-06, each cropped to the AOI (~1 GB total).
**We deliberately filtered out winter acquisitions** when staging this
stack — at the dome's elevation, snow cover collapses C-band coherence
and contaminates the phase. Consider similar filtering for your own
data (or a different sensor) if you work over snowy terrain.

GPS validation comes from two PBO stations: HUSB on the NW flank of
the dome (UNR-NGL reports ~+4 mm/yr LOS uplift) and PMAR ~10 km south,
which we treat as the reference.

![Three Sisters AOI with GPS comparison](three_sisters_figure9.png)

*Figure 9 from
[Staniewicz et al. 2025](https://arxiv.org/abs/2511.12051). The paper
runs over a wider AOI; we work on a smaller two-burst subset that
covers the HUSB and PMAR GPS stations shown in the inset, mainly to
keep the time series inversion tractable. The bottom panel — dolphin
LOS (orange) vs HUSB GPS (blue) — is the kind of GPS-vs-dolphin
agreement to aim for.*

By the end you'll know what each stage of dolphin produces, how to
read its QA outputs, and which parameters matter.
""")

code(r"""
%matplotlib inline
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

HERE = Path.cwd()

DATA         = HERE / "three_sisters" / "data"
SLC_PER_BURST = DATA / "slc_tif"          # per-burst per-acquisition SLCs
SLC_STITCHED = DATA / "slc_stitched"      # both bursts merged per acquisition
WORK         = HERE / "three_sisters" / "notebook_work"
WORK.mkdir(parents=True, exist_ok=True)

BURSTS = ["T115-245676-IW2", "T115-245677-IW2"]
""")

code(r"""
import dolphin
print(f"dolphin {dolphin.__version__}")
""")


# ===========================================================================
# Section 1 — Input data
# ===========================================================================

md(r"""
## 0. Notebook prep

We start with the housekeeping: where the data comes from, how the
full pipeline runs in production, and a few imports. The three actual
processing steps (PS / phase linking, unwrap, inversion) are Sections
1, 2, and 3 below.

### 0.1 Input data

dolphin works on **any stack of coregistered SLCs**. As long as every
acquisition has been resampled onto the same pixel grid, dolphin doesn't
care where the data came from — OPERA CSLCs, ISCE2 `topsApp` outputs,
GAMMA, your own pipeline, etc. That flexibility is one of dolphin's
strengths: it focuses purely on phase estimation, leaving SAR processing
and coregistration to whatever upstream tool you prefer.

For this notebook we use **OPERA L2 CSLCs**: per-burst, per-acquisition
HDF5 files holding a geocoded complex SLC, produced from Sentinel-1 SAFE
products by JPL's OPERA project and hosted at ASF DAAC. Every CSLC for a
given burst is already coregistered to the same UTM grid.

### Get the staged data

We've pre-staged the 210 CSLCs (105 dates × 2 bursts), the per-epoch
cropped geotiffs, **the burst-stitched geotiffs** (one complex SLC
per acquisition, both bursts already merged onto one UTM grid under
`data/slc_stitched/`), and the GNSS data into a single tarball on S3.
Pull it down and extract once:
""")

code(r"""
# Placeholder - replace this URL with the actual S3 path once uploaded.
S3_URL = "s3://earthscope-course-data/three-sisters-cslc.tar.gz"

# !aws s3 cp $S3_URL three-sisters-cslc.tar.gz --no-sign-request
# !tar -xzf three-sisters-cslc.tar.gz -C three_sisters/
""")

md(r"""
### Or build the staged dataset yourself

If you'd rather pull the OPERA CSLCs from ASF and crop to the AOI
yourself (this is what the staged tarball was made from), there's a
one-file script in this repo. Walks ASF (needs Earthdata `~/.netrc`),
downloads each CSLC, writes per-burst cropped geotiffs into
`data/slc_tif/`, then stitches into one tif per acquisition under
`data/slc_stitched/`.
""")

code(r"""
# !python stage_cslcs.py \
#     --bursts T115-245676-IW2 T115-245677-IW2 \
#     --start 2016-07-01 --end 2024-07-01 \
#     --bbox 587950 4866900 609130 4890550 \
#     --out three_sisters/data/
""")

md(r"""
The OPERA CSLC filename encodes the burst and the acquisition timestamp:

```
OPERA_L2_CSLC-S1_T115-245676-IW2_20160722T141410Z_20240614T041653Z_S1A_VV_v1.1.h5
                 ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^
                 track-burst-IW   acquisition       OPERA L2 processing time
```

The two bursts here are adjacent in azimuth — burst 245676 covers the
South Sister dome, burst 245677 covers PMAR ~10 km south. The staging
step has already merged them onto one common grid, so dolphin can
process the pair as a single stack.
""")


# ===========================================================================
# Section 2 — Full pipeline as one command
# ===========================================================================

md(r"""
### 0.2 The full pipeline as two commands

In production dolphin is two commands. First you generate a config YAML
that points at your inputs and sets every knob you care about:

```bash
dolphin config --slc-files three_sisters/data/*.h5 --subdataset /data/VV \
    --strides 12 6  --max-bandwidth 3  --unwrap-method snaphu
```

`dolphin config` with no flags would write a sensible default YAML for
the given inputs. CLI flags let you override the most-tuned knobs up
front (above: strides, network bandwidth, unwrap method); anything not
exposed as a flag — or anything you want to change later — you just
hand-edit in the resulting `dolphin_config.yaml`.

Then you run the whole pipeline against that YAML:

```bash
dolphin run dolphin_config.yaml
```

`dolphin run` runs the full pipeline: PS, phase linking, ifgs, unwrap,
inversion — all in sequence. We're going to skip both commands and walk
through each stage by hand so the YAML isn't a black box.
""")

code(r"""
# The two commands we're NOT running. Uncomment to do the whole pipeline.
#
# 1) Generate a YAML pointing at our CSLCs (knobs via flags here, or
#    leave defaults and hand-edit dolphin_config.yaml afterwards):
# !dolphin config --slc-files three_sisters/data/*.h5 --subdataset /data/VV
#
# 2) Run the pipeline:
# !dolphin run dolphin_config.yaml
""")


# ===========================================================================
# Section 1 — Step 1: PS / PL
# ===========================================================================

md(r"""
## 1. Step 1 — PS detection and phase linking

**Why this step.** Raw SLCs are noisy. At any single pixel, thermal
noise and random changes in the minor scatterers inside the resolution
cell add a random phase on top of whatever deformation signal we want.

**What we want out of it.** One clean phase value per pixel per epoch
(here and below "epoch" just means one acquisition) — something we can
difference between dates to get displacement.

**How dolphin does this.** Classify every pixel by how stable its
scattering is. The stable ones (PS) already have clean phase; use them
directly. The noisier ones (DS) get improved by averaging the phase of
nearby pixels that share the same scattering behavior.

---

A **PS pixel** is one whose radar return is dominated by a single bright,
stable reflector — a rock face, a roof corner, a metal pole. The
satellite sees almost the same thing on every pass, so the measured
brightness stays about the same from one acquisition to the next, and
the phase tracks the physical motion of that one reflector. PS pixels
are usable as-is.

A **DS pixel** is the opposite case: many minor scatterers in the
resolution cell jointly produce the return — bare soil, gravel,
vegetation, snow patches. Between acquisitions those minor scatterers
shift (wind moves leaves, moisture changes, surface roughness varies),
so both the brightness and the per-pixel phase fluctuate. DS pixels
tend to be too noisy to use individually, but if you average the phase
of a *group* of nearby DS pixels that all share the same scattering
statistics, the noise averages down and you get back a usable phase.

The discriminator between the two cases is **amplitude dispersion**:

$$
D = \frac{\sigma_A}{\mu_A}
$$

where, at each pixel,

- $\mu_A$ = mean amplitude across the $N$ acquisitions (how bright the
  pixel is on average)
- $\sigma_A$ = standard deviation of amplitude across the $N$
  acquisitions (how much that brightness fluctuates between
  acquisitions)

A PS pixel — same brightness each pass — has small $\sigma_A$ relative
to $\mu_A$, so small $D$. dolphin's default threshold is $D < 0.42$ for
PS; everything else is treated as DS.

The same two statistics ($\mu_A$, $\sigma_A$) also drive the GLRT test
that decides *which* DS pixels are similar enough to average together.
So dolphin computes them up front and reuses them at both stages.
""")

md(r"""
### 1.1 Load the stitched SLC stack

The staging step (`stage_cslcs.py`) already merged the two per-burst
SLCs onto a common spatial grid for every acquisition (105 stitched
complex tifs under `data/slc_stitched/`). The merge is element-wise:
take whichever burst has valid (non-zero) data at each pixel; the two
bursts cover complementary halves of the AOI with a small overlap.

We just hand dolphin the stitched file list.
""")

code(r"""
slc_tifs = sorted(SLC_STITCHED.glob("*.slc.tif"))
print(f"{len(slc_tifs)} stitched SLC tifs")
""")

md(r"""
### 1.2 Build a virtual stack and run PS detection
""")

code(r"""
# dolphin.io.VRTStack:
#   wraps the per-epoch geotiffs as a single lazy-loaded virtual stack.
#   We hand this object to every dolphin function below instead of
#   passing 105 file paths.
from dolphin.io import VRTStack

vrt = VRTStack(
    file_list=slc_tifs,                 # one path per acquisition
    outfile=WORK / "stack.vrt",         # virtual stack written here
    fail_on_overwrite=False,            # overwrite if it already exists
)
print(f"VRT stack:  {vrt.shape}  dtype={vrt.dtype}")
""")

code(r"""
# dolphin.ps.create_ps:
#   one pass over the stack to compute per-pixel mu_A and sigma_A and
#   write three rasters:
#     amp_mean.tif        = mu_A   (mean amplitude per pixel)
#     amp_dispersion.tif  = D      (sigma_A / mu_A per pixel)
#     ps_mask.tif         = boolean mask, True where D < threshold
from dolphin.ps import create_ps

amp_mean = WORK / "amp_mean.tif"
amp_disp = WORK / "amp_dispersion.tif"
ps_mask  = WORK / "ps_mask.tif"

create_ps(
    reader=vrt,                          # input stack
    like_filename=slc_tifs[0],           # template raster: the outputs inherit
                                         # its CRS, geotransform, and pixel grid
                                         # (just one example file, not a glob)
    output_file=ps_mask,                 # boolean PS-mask output
    output_amp_mean_file=amp_mean,       # mu_A raster output
    output_amp_dispersion_file=amp_disp, # D = sigma_A / mu_A raster output
    amp_dispersion_threshold=0.42,       # dolphin's default PS cutoff
)
""")

md(r"""
### 1.3 Plotting the two input statistics

$\mu_A$ tells us how bright each pixel is on average — rocky and rough
terrain returns brightly, water and shadow are dark.

$\sigma_A$ tells us how much that brightness fluctuates between
acquisitions. We get it via $\sigma_A = D \cdot \mu_A$ since dolphin
writes $D$ directly.
""")

code(r"""
mu  = xr.open_dataarray(amp_mean).squeeze()
D   = xr.open_dataarray(amp_disp).squeeze()
sig = mu * D                                  # sigma_A = D * mu_A

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Left panel: mean amplitude, log-stretched so faint scatterers show.
np.log10(mu.where(mu > 0)).plot.imshow(
    ax=axes[0], cmap="gray", robust=True, add_labels=False)
axes[0].set_title(r"mean amplitude  $\mu_A$  (log$_{10}$)")

# Right panel: std amplitude, linear scale.
sig.plot.imshow(
    ax=axes[1], cmap="magma", robust=True, add_labels=False)
axes[1].set_title(r"std amplitude  $\sigma_A$")

plt.show()
""")

md(r"""
### 1.4 One pixel through time

A PS pixel holds about the same brightness across acquisitions. A DS
pixel fluctuates around the same average.
""")

code(r"""
# Stack every SLC's amplitude into one (n_epochs, ny, nx) array.
amp_stack = np.abs(
    np.stack([xr.open_dataarray(p).squeeze().values for p in slc_tifs])
)

# Restrict our PS/DS candidates to bright pixels so D isn't dominated by
# the noise floor.
D_bright = np.where(mu.values > np.nanpercentile(mu.values, 70),
                     D.values, np.nan)
ps_rc = np.unravel_index(np.nanargmin(D_bright), D_bright.shape)
ds_rc = np.unravel_index(np.nanargmax(D_bright), D_bright.shape)

a_ps = amp_stack[:, ps_rc[0], ps_rc[1]]
a_ds = amp_stack[:, ds_rc[0], ds_rc[1]]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(a_ps, "o-", label=fr"PS pixel  $D$={D.values[ps_rc]:.2f}")
ax.plot(a_ds, "s-", label=fr"DS pixel  $D$={D.values[ds_rc]:.2f}")
ax.set_xlabel("acquisition index")
ax.set_ylabel("amplitude")
ax.set_title("amplitude over time: PS is bright and steady, DS is dimmer and fluctuates")
ax.legend(); ax.grid(alpha=0.3)
plt.show()
""")

md(r"""
### 1.5 Picking PS pixels: threshold on $D$

The PS mask is just $D < \text{threshold}$, computed pixel-by-pixel.
Default threshold is 0.42 in dolphin.
""")

code(r"""
psm = xr.open_dataarray(ps_mask).squeeze()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

np.log10(mu.where(mu > 0)).plot.imshow(
    ax=axes[0], cmap="gray", robust=True, add_labels=False)
axes[0].set_title(r"$\mu_A$  (log$_{10}$)")

D.plot.imshow(ax=axes[1], cmap="viridis", vmin=0, vmax=1, add_labels=False)
axes[1].set_title(r"amplitude dispersion  $D$")

psm.plot.imshow(ax=axes[2], cmap="Greens", add_labels=False,
                add_colorbar=False)
axes[2].set_title(f"PS mask  ($D<0.42$):  "
                  f"{100 * float((psm > 0).mean()):.1f}% PS")

plt.show()
""")

md(r"""
### 1.6 Phase linking for DS pixels

DS pixels need neighborhood averaging — but you can't blindly average
rock with forest, you have to first pick neighbors that share the same
scattering statistics. The Generalized Likelihood Ratio Test (Parizzi &
Brcic 2011) does this using only $\mu_A$. For two pixels $p$ and $q$
it forms

$$
T_{pq} \;=\; 2 N \, \ln\!\left[
\frac{\mu_{A,p}^{2} + \mu_{A,q}^{2}}{2\,\mu_{A,p}\,\mu_{A,q}}
\right]
$$

where $N$ is the number of acquisitions. dolphin accepts $q$ as a
statistically homogeneous pixel (SHP) of $p$ when $T_{pq}$ is below a
$\chi^{2}$ critical value at significance level `shp_alpha` (dolphin's
default is 0.001). Tighter `shp_alpha` ⇒ stricter test ⇒ fewer
accepted SHPs. The looser you go, the closer the result is to a
plain boxcar multilook.

For each DS-classified pixel, dolphin sweeps a fixed search window
(`half_window`, e.g. 11 in azimuth × 5 in range for OPERA's asymmetric
pixel posting) and applies the test pairwise. Each DS pixel ends up
with its own SHP set. dolphin then groups those SHPs together and
estimates the most likely noise-mitigated phase from their combined noisy
observations, returning one cleaned phase per acquisition (the EMI
estimator — Ansari, De Zan, Bamler 2018 — does this by solving an
eigenvalue problem on the group's coherence matrix).
""")

code(r"""
# dolphin.workflows.sequential.run_wrapped_phase_sequential:
#   the actual phase-linking workflow. For each pixel it
#     1) finds neighbors via the GLRT test using mu_A and sigma_A,
#     2) builds the NxN sample coherence matrix on that SHP set,
#     3) inverts it (EMI by default) to get one phase per acquisition.
#   Writes per-epoch phase-linked .slc.tif files, plus a temporal
#   coherence raster, shp_counts, and phase_similarity.
from dolphin.workflows.sequential import run_wrapped_phase_sequential
from dolphin.shp import ShpMethod
import shutil

pl_dir = WORK / "linked_phase"
# Wipe any prior PL outputs - dolphin's "load existing" cache path
# trips on its own partial outputs from a previous cell run.
shutil.rmtree(pl_dir, ignore_errors=True)
pl_dir.mkdir(parents=True, exist_ok=True)

run_wrapped_phase_sequential(
    slc_vrt_stack=vrt,                  # the virtual stack we built above
    output_folder=pl_dir,               # where PL .slc.tifs and QA rasters land
    ministack_size=15,                  # acquisitions per sequential ministack
    half_window={"y": 11, "x": 5},      # SHP search window (azimuth, range)
    strides={"y": 6, "x": 3},           # spatial downsample factor of PL output
    ps_mask_file=ps_mask,               # PS mask from create_ps
    amp_mean_file=amp_mean,             # mu_A raster used by the GLRT test
    amp_dispersion_file=amp_disp,       # D = sigma_A / mu_A raster
    shp_method=ShpMethod.GLRT,          # how to pick SHPs
    shp_alpha=0.001,                    # dolphin's default; smaller = stricter
)

pl_slcs = sorted(pl_dir.glob("2*.slc.tif"))
temp_coh = next(pl_dir.glob("temporal_coherence_*.tif"))
print(f"{len(pl_slcs)} phase-linked SLCs, temp_coh raster = {temp_coh.name}")
""")

md(r"""
### 1.7 Temporal coherence — how good was the fit?

For each pixel, dolphin's temporal coherence measures how similar the
recovered noise-mitigated phase is to the observed noisy phases.
Values near 1 mean the estimator fit the observations well; values
below ~0.5 mean it didn't, and any downstream product over those
pixels is unreliable. We use this raster to mask the final velocity
in Section 3.

A quick terminology note: "coherence" gets used two different ways in
InSAR.

- The **spatial coherence** $\gamma$ of an interferogram pair is a
  per-pixel statistic over a small neighborhood, typically a 3×3 or
  5×5 boxcar. It estimates how phase-correlated two SLCs are at one
  pixel. (We use it as the per-pair quality input to snaphu in
  Section 2.)
- The **temporal coherence** plotted here is a stack-level statistic
  from phase linking — how well the joint estimator fit the
  observations across *all* acquisitions at one pixel.

By construction PS pixels have temporal coherence = 1.0 (their phase
is used directly without any estimator, so the "fit" is exact).
""")

code(r"""
tc = xr.open_dataarray(temp_coh).squeeze()

fig, ax = plt.subplots(figsize=(7, 5))
tc.plot.imshow(ax=ax, cmap="cividis", vmin=0, vmax=1, add_labels=False)
ax.set_title(f"temporal coherence  (mean = {float(tc.mean()):.2f},  "
             f"fraction > 0.5 = {100 * float((tc > 0.5).mean()):.1f}%)")
plt.show()
""")

md(r"""
What does "high vs low temp_coh" look like at one pixel? We could pull
two real pixels from the stack, but $2\pi$ wrapping makes the
comparison hard to read (the high-tc pixel's smooth phase trend wraps
around the same as the low-tc pixel's random walk). It's clearer with
**synthetic** time series:

- High tc → recovered phase tracks a small deformation signal with a
  tight noise envelope (the model fit the observations well).
- Low tc → recovered phase wanders ±$\pi$ pretty much at random (the
  model couldn't find a coherent signal).
""")

code(r"""
# Synthetic phase histories. Both panels show the SAME noisy observed
# phases (linear trend + noise) - what changes is the recovered PL
# fit. High-tc means PL locked onto the underlying signal; low-tc
# means it didn't.
rng    = np.random.default_rng(0)
N_PLOT = 20
t      = np.arange(N_PLOT)
truth  = 0.1 * t                                  # small linear trend

# Shared observed phases for both panels.
obs = truth + rng.normal(0, 0.6, N_PLOT)

# High-tc PL: estimate sits right on the underlying trend.
est_hi = truth + rng.normal(0, 0.05, N_PLOT)

# Low-tc PL: estimate is its own random walk, doesn't track the
# observations at all.
est_lo = rng.uniform(-np.pi, np.pi, N_PLOT)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                          constrained_layout=True)

for ax, obs, est, label in [
    (axes[0], obs, est_hi, "high temporal coherence  (tc ~ 0.95)"),
    (axes[1], obs, est_lo, "low temporal coherence   (tc ~ 0.05)"),
]:
    ax.plot(obs, "o", color="gray", alpha=0.6, ms=6, label="observed")
    ax.plot(est, "s-", color="C0", lw=1.6, ms=5,
            label="phase-linked estimate")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_ylabel("phase  (rad)")
    ax.set_title(label)
    ax.legend(loc="best"); ax.grid(alpha=0.3)

axes[-1].set_xlabel("acquisition index")
plt.show()
""")

md(r"""
### 1.8 Raw single-look ifg vs phase-linked ifg

Same idea but spatial instead of temporal: take one acquisition pair
and form the interferogram two ways — once from the raw single-look
SLCs and once from dolphin's phase-linked SLCs. The PL ifg is what
the unwrapper in Section 2 will see.
""")

code(r"""
# Load the complex stacks once, here, since Section 2 also needs them.
slc_complex = np.stack([xr.open_dataarray(p).squeeze().values
                         for p in slc_tifs])
pl_stack    = np.stack([xr.open_dataarray(p).squeeze().values
                         for p in pl_slcs])

# Short-baseline adjacent pair - very clean, makes the denoising
# effect obvious.
pair = (0, 1)

# Complex ifgs at both pixel grids (raw is native, PL is strided).
ifg_raw_cpx = slc_complex[pair[0]] * slc_complex[pair[1]].conj()
ifg_pl_cpx  = pl_stack[pair[0]]    * pl_stack[pair[1]].conj()

fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

# aspect="auto" so the strided grid (much wider than tall) doesn't
# render as a thin strip.
axes[0].imshow(np.angle(ifg_raw_cpx), cmap="hsv",
                vmin=-np.pi, vmax=np.pi, aspect="auto")
axes[0].set_title("raw single-look interferogram")

axes[1].imshow(np.angle(ifg_pl_cpx), cmap="hsv",
                vmin=-np.pi, vmax=np.pi, aspect="auto")
axes[1].set_title("phase-linked interferogram")

for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.show()
""")

md(r"""
### 1.9 Step 1 knobs

| knob | symptom to watch for |
|---|---|
| `amp_dispersion_threshold` | **Strongly stack-size dependent.** On short stacks (~5-10 dates) the spread of $D$ is large, so the default threshold flags huge numbers of false-positive "PS" pixels. Turn it down. On long stacks some processors skip PS labeling entirely and treat every pixel as DS — a defensible choice. |
| `half_window` (y, x) | Too big and you bleed across land covers — a forest pixel pulls in rock neighbors. Too small and SHP counts are too low for stable phase. |
| `shp_alpha` | Stricter alpha = fewer SHPs, less averaging. Looser = more SHPs, closer to a plain boxcar multilook. dolphin's default is 0.001. |

Low temporal coherence is the warning sign that matters most. Anywhere
temp_coh is low, downstream ifgs are noisy, unwrap is sketchy, and the
final velocity is garbage. Mask by `temp_coh ≥ 0.5` (or higher).
""")


# ===========================================================================
# Section 2 — Step 2: Unwrap
# ===========================================================================

md(r"""
## 2. Step 2 — Unwrap interferograms

**Why this step.** Differencing two PL SLCs gives the *wrapped* phase
difference (modulo $2\pi$). Even a few cm of LOS displacement at C-band
wraps several times — we can't read displacement off the wrapped phase
directly.

**What we want out of it.** Unwrapped ifgs: phase that no longer wraps
modulo $2\pi$, so it can be converted to LOS displacement. Unwrapping
*tries* to recover the integer cycle count, but it's a hard problem in
noisy regions and the result can have residual errors — the inversion
step in Section 3 cleans those up by using the redundancy in the ifg
network.

**How dolphin does this.** Pick which acquisition pairs to form ifgs
between (the network), pick an unwrapping algorithm, run it on each
ifg. Then check the result before trusting it downstream.

---

### Unwrappers in dolphin

| `unwrap_method` | notes |
|---|---|
| `snaphu` | Statistical-cost MCF, per ifg. Default. (Chen & Zebker 2001) |
| `phass` | Residue-cut MCF variant. Sometimes wins at scene edges. |
| `icu` | Branch-cut method from ROI_PAC. Older, lighter-weight. |
| `spurt` | 3D space-time MCF on the whole stack. Best when noise is high or cycles are many. |
| `whirlwind` | dolphin's experimental fast unwrapper. |

### Networks

We pick which acquisition pairs to form ifgs between via the
`interferogram_network` config block. Common choices:

- bandwidth-$k$ — every pair within $k$ acquisitions. Adds the
  redundancy L1 inversion needs to catch cycle slips. **`dolphin
  config` writes `max_bandwidth: 3` as its default**, so this is what
  you'll see in a fresh YAML.
- Single-reference — one master vs everything ($N-1$ ifgs). Cheap,
  but a bad master image corrupts every ifg in the network.
- NN (nearest-neighbour) — adjacent pairs only, $N-1$ ifgs. No
  redundancy.
- Full — every possible pair.

### 2.1 Form one ifg from two PL SLCs

A full bandwidth-3 unwrap over our 105-acquisition stack is ~300 ifgs;
that's the slow part of the pipeline and takes much longer than fits in
this notebook. We'll unwrap a single representative ifg here to show
the call. The full-network result is already on disk and feeds Section
5.

(If you want to do the full unwrap inline, the one-liner is below — but
expect it to run for an hour or more on a laptop.)

```python
# from dolphin.unwrap import run as run_unwrap, UnwrapMethod
# run_unwrap(ifg_filenames=ifg_paths, cor_filenames=cor_paths,
#            output_path=WORK / "unwrapped",
#            unwrap_method=UnwrapMethod.SNAPHU, nlooks=24.0)
```
""")

code(r"""
# Wrapped phase of the PL interferogram from the previous section.
ifg     = ifg_pl_cpx
wrapped = np.angle(ifg)

# Boxcar spatial-coherence estimator. dolphin doesn't ship one as a
# standalone helper, so this is a small local function.
from scipy.ndimage import uniform_filter

def spatial_coherence(slc_a, slc_b, window=5):
    # Boxcar gamma estimator: |<a b*>| / sqrt(<|a|^2> <|b|^2>).
    ifg = np.nan_to_num(slc_a * slc_b.conj())
    i2  = np.nan_to_num(np.abs(slc_a) ** 2)
    j2  = np.nan_to_num(np.abs(slc_b) ** 2)
    num = (uniform_filter(ifg.real, window)
            + 1j * uniform_filter(ifg.imag, window))
    den = np.sqrt(uniform_filter(i2, window) * uniform_filter(j2, window))
    return np.abs(num) / np.maximum(den, 1e-9)

# For VISUALIZATION compute coherence on the raw single-look SLCs (at
# native grid) - more realistic, shows the spatial variation you'd
# actually see. PL-based coherence is near 1 everywhere because PL has
# already denoised, which is less visually informative.
coh_raw = spatial_coherence(slc_complex[pair[0]],
                             slc_complex[pair[1]], window=5)

# For the snaphu unwrap input we use the PL-based coherence on the
# strided grid - matches the PL ifg snaphu actually sees and works
# better as a per-pixel quality weight inside snaphu.
coh = spatial_coherence(pl_stack[pair[0]], pl_stack[pair[1]], window=5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# aspect="auto" so both panels fill the same axes box even though
# wrapped is on the strided grid and coh_raw is on the native grid.
im_w = axes[0].imshow(wrapped, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                      aspect="auto")
axes[0].set_title(f"wrapped phase  (pair {pair[0]} -> {pair[1]})")
plt.colorbar(im_w, ax=axes[0], shrink=0.7,
             ticks=[-np.pi, 0, np.pi],
             label="phase (rad)").ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

im_c = axes[1].imshow(coh_raw, cmap="Greys_r", vmin=0, vmax=1,
                      aspect="auto")
axes[1].set_title("spatial coherence  ($5{\\times}5$ window, raw SLCs)")
plt.colorbar(im_c, ax=axes[1], shrink=0.7, label=r"$\gamma$")

for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.show()
""")

md(r"""
### 2.2 Run snaphu

`snaphu.unwrap` is the standalone Python wrapper around Chen-Zebker's
classic statistical-cost min-cost-flow unwrapper. Takes the complex
ifg + coherence and returns unwrapped phase plus a connected-component
label raster (which regions snaphu kept consistent together).
""")

code(r"""
# snaphu.unwrap:
#   the actual unwrap call. dolphin's run_unwrap drives this same
#   library under the hood when unwrap_method=snaphu.
import snaphu

# nlooks: equivalent number of independent looks averaged into each
# input pixel. snaphu uses this to scale its coherence-to-noise model.
# Our PL outputs are at strides (y=6, x=3) so the effective looks =
# 6 * 3 = 18.
unw, conncomp = snaphu.unwrap(
    ifg.astype(np.complex64),
    coh.astype(np.float32),
    nlooks=18.0,
    init="mcf",                      # initialize flows via min-cost-flow
    cost="defo",                     # deformation cost model
)

# Mask outputs the same way the wrapped panel implicitly is - pixels
# where the input ifg was nodata (NaN or zero magnitude).
nodata = ~np.isfinite(ifg) | (np.abs(ifg) == 0)
unw_m      = np.where(nodata, np.nan, unw)
conncomp_m = np.ma.masked_where(nodata, conncomp)
wrapped_m  = np.where(nodata, np.nan, wrapped)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Wrapped input (what snaphu saw).
im_w = axes[0].imshow(wrapped_m, cmap="hsv", vmin=-np.pi, vmax=np.pi,
                      aspect="auto")
axes[0].set_title("wrapped phase input  (rad)")
plt.colorbar(im_w, ax=axes[0], fraction=0.046)

# Unwrapped output. Symmetric color range at the 10th/90th percentile
# magnitude so a few outlier pixels don't wash out the rest.
vlim = float(np.nanpercentile(np.abs(unw_m), 90))
im_u = axes[1].imshow(unw_m, cmap="RdBu_r", aspect="auto",
                      vmin=-vlim, vmax=+vlim)
axes[1].set_title("unwrapped phase  (rad)")
plt.colorbar(im_u, ax=axes[1], fraction=0.046)

# Connected components: regions snaphu kept consistent together.
cmap_cc = plt.cm.tab20.copy(); cmap_cc.set_bad("white")
axes[2].imshow(conncomp_m, cmap=cmap_cc, aspect="auto")
axes[2].set_title("connected components")

for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.show()
""")

md(r"""
### 2.3 Goldstein filter — a spectral-domain phase smoother

When the wrapped phase is too noisy for snaphu to follow, the
Goldstein filter (Goldstein & Werner 1998) smooths it in the spectral
domain beforehand. It takes the 2D FFT of small patches, raises the
spectrum to a power $\alpha < 1$ to sharpen the dominant fringe
direction, then transforms back. Default $\alpha$ is 0.5.

Note that **dolphin's `run_goldstein` defaults to `False`** — Goldstein
is opt-in. Turn it on (`--run-goldstein` on `dolphin config`, or
`unwrap_options.run_goldstein: true` in the YAML) when your wrapped
ifgs look too noisy for the unwrapper to follow.
""")

code(r"""
# dolphin.goldstein.goldstein:
#   the spectral phase filter. Operates on the complex ifg, parameter
#   alpha controls smoothing strength (higher = smoother).
from dolphin.goldstein import goldstein

# Goldstein takes FFTs of patches; NaN propagates through them and
# pollutes the whole result. Zero NaN regions first, run the filter,
# then re-apply the nodata mask to the output so the edges look clean.
ifg_in = np.where(np.isfinite(ifg), ifg, 0.0).astype(np.complex64)
filtered = goldstein(ifg_in, alpha=0.5, psize=32)
nodata = ~np.isfinite(ifg) | (np.abs(ifg) == 0)
filtered_view = np.where(nodata, np.nan, np.angle(filtered))
wrapped_view  = np.where(nodata, np.nan, np.angle(ifg))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

axes[0].imshow(wrapped_view, cmap="hsv", vmin=-np.pi, vmax=np.pi)
axes[0].set_title("wrapped, no filter")

axes[1].imshow(filtered_view, cmap="hsv", vmin=-np.pi, vmax=np.pi)
axes[1].set_title(r"wrapped, Goldstein  $\alpha$=0.5")

for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.show()
""")

md(r"""
### 2.4 Step 2 knobs

| knob | what it does | when to change it |
|---|---|---|
| `unwrap_options.unwrap_method` | which unwrapping algorithm | switch to spurt if snaphu makes lots of cycle errors |
| `unwrap_options.run_goldstein` | turn the Goldstein pre-filter on/off | turn on for noisy AOIs |
| `unwrap_options.preprocess_options.alpha` | Goldstein filter strength | higher = stronger smoothing; too high erases real features |
| `unwrap_options.preprocess_options.psize` | Goldstein patch size | larger = smoother on big features, slower |
| `unwrap_options.preprocess_options.interpolation_cor_threshold` | coherence below this gets treated as no-data and interpolated | raise it to fill more pixels before unwrap |
| `unwrap_options.preprocess_options.interpolation_similarity_threshold` | similarity-based no-data fill threshold | the equivalent gate for phase-similarity |
| `unwrap_options.snaphu_options.cost` | snaphu cost model: `smooth`/`defo`/`topo` | `defo` for our case (deformation-dominated) |
| `interferogram_network.max_bandwidth` | how many ifgs per acquisition | bandwidth-3 is the usual production choice |
| `interferogram_network.max_temporal_baseline` | cap pairs at N days | useful if longer baselines decorrelate hard |
| `phase_linking.temporal_coherence_threshold` | drop pixels with tc below this before unwrap | tighter = fewer but cleaner pixels |
""")


# ===========================================================================
# Section 3 — Step 3: Inversion
# ===========================================================================

md(r"""
## 3. Step 3 — Time-series inversion

**Why this step.** Each unwrapped ifg only tells us the phase
*difference* between two acquisitions. We want cumulative displacement
*per acquisition*, and from that a single velocity per pixel. We also
want to use the **redundancy** built into the ifg network — every
bandwidth-3 epoch gets ~6 ifgs touching it, so we have more
observations than unknowns and can average down unwrapping mistakes
and atmospheric noise.

**What we want out of it.** A per-acquisition cumulative displacement
referenced to a chosen first epoch (one raster per acquisition in
`displacement_*.tif`) and a per-pixel linear velocity (`velocity.tif`).

**How dolphin does this.** Stack all the ifgs into one linear system
relating per-acquisition cumulative phase to the observed ifg phase
differences, invert it, and convert the resulting phase to LOS
displacement. Pick a norm that's robust to occasional bad ifgs (L1).

---

Define:

- $N$ = number of acquisitions
- $M$ = number of unwrapped ifgs (acquisition pairs)
- $\varphi \in \mathbb{R}^N$ = **cumulative phase at each acquisition,
  referenced to acquisition 0** — what we want.
- $\Delta\varphi_{\text{obs}} \in \mathbb{R}^M$ = **observed unwrapped
  ifg phases** (each one is the phase *change* between two acquisitions)
  — what we measured.
- $A \in \mathbb{R}^{M \times N}$ = the **design matrix** that maps
  cumulative phase to phase changes ($M$ rows, $N$ columns).

Each row of $A$ corresponds to one ifg. For the ifg between
acquisitions $i$ and $j$, that row is all zeros except $-1$ at column
$i$ and $+1$ at column $j$ — i.e. the ifg measures
$\varphi_j - \varphi_i$.

Stacking all $M$ rows gives the system

$$
A\, \varphi \;=\; \Delta\varphi_{\text{obs}}
$$

We're solving the **inverse problem**: given the observed phase changes
on the right, recover the cumulative phase on the left.

The system as written has one extra degree of freedom — every
$\varphi$ only ever appears as part of a *difference* in the
observations, so any constant added to the whole $\varphi$ vector
gives an equivalent solution. To pin down a unique answer we fix one
element (the **reference epoch**, conventionally $\varphi_0 = 0$).
dolphin also lets you fix a **reference pixel** in the spatial domain
— typically a GPS station — which removes the per-ifg scene-wide
phase offset that comes from the $2\pi$-ambiguous nature of the
measured phase combined with imperfect orbital tracks.

For a NN (nearest-neighbor) network $M = N - 1$, so the system is
exactly-determined — one solution, no redundancy. For bandwidth-$k$
networks $M > N - 1$, so we have more equations than unknowns and
have to pick a residual norm to minimize. That redundancy is what L1
inversion uses to catch bad ifgs.

### 3.1 Design matrices for common networks

$A$ on a small $N = 10$ stack, drawn for four different network
choices. Same number of unknowns (9, after fixing epoch 0); very
different numbers of rows = very different conditioning and
redundancy.
""")

code(r"""
def build_A(n_epochs, pairs):
    A = np.zeros((len(pairs), n_epochs))
    for m, (i, j) in enumerate(pairs):
        A[m, i] = -1
        A[m, j] = +1
    return A

N = 10
networks = {
    "NN  (max_bw=1)":   [(i, i + 1) for i in range(N - 1)],
    "single-reference": [(0, j)     for j in range(1, N)],
    "bandwidth-3 (dolphin CLI default)": [(i, j) for i in range(N)
                                       for j in range(i + 1, min(i + 4, N))],
    "full":             [(i, j)     for i in range(N)
                                       for j in range(i + 1, N)],
}

fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
for ax, (name, pairs) in zip(axes, networks.items()):
    A = build_A(N, pairs)
    ax.imshow(A, cmap="bwr", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_title(f"{name}\n{A.shape[0]} ifgs x {A.shape[1]} epochs")
    ax.set_xlabel("epoch"); ax.set_ylabel("ifg")
plt.show()
""")

md(r"""
### 3.2 L1 vs L2

For a bandwidth-3+ network the system is over-determined, so no exact
solution exists — we minimize a residual norm.

**Residual, defined.** Suppose we have a candidate cumulative phase
$\varphi_{\rm est}$. We can use $A$ to *predict* what each ifg's
unwrapped phase change should be: $(A \varphi_{\rm est})_m$. The
residual at ifg $m$ is then

$$
r_m \;=\; (A\, \varphi_{\rm est})_m \;-\; (\Delta\varphi_{\rm obs})_m,
$$

i.e. (what the candidate predicts the ifg should read) minus (what
the ifg actually reads). One residual per ifg. With $M$ ifgs that's
an $M$-vector $\mathbf{r}$. A "good" $\varphi_{\rm est}$ makes all
$r_m$ small.

**L2** minimizes $\|\mathbf{r}\|_2^2 = \sum_m r_m^2$ — the sum of
*squared* residuals (subscript 2 = L2 norm, the exponent 2 is the
square). Closed-form via `np.linalg.lstsq`, fast. Squaring penalizes
big residuals very heavily, so the solver prefers to spread one big
error across several smaller residuals — and a single bad ifg ends up
biasing the whole time series.

**L1** minimizes $\|\mathbf{r}\|_1 = \sum_m |r_m|$ — the sum of
*absolute* residuals (subscript 1 = L1 norm, no exponent). Solved
as a linear program. One big residual costs the same as several
medium ones, so the solver is happy to leave the one bad ifg with a
huge residual rather than smear the bias around. That's why L1
**isolates** outliers and L2 spreads them.

Below: synthetic demo at one pixel. The "clean" L2 and L1 are
basically identical. To one of the ifgs we inject a synthetic **cycle
slip** — an unwrapping error of $+6\pi$ (3 cycles), which is the kind
of artifact snaphu produces over low-coherence pixels when it picks
the wrong integer number of $2\pi$'s. L2 smears the bias across the
recovered time series; L1 isolates the bad observation. The right
panel shows the per-ifg residual bars — that's where the unwrap
error "lives" in each solver's solution.
""")

code(r"""
from scipy.optimize import linprog

# Synthetic: N=8 epochs, linear deformation, bandwidth-3 network, plus
# a +2*pi cycle slip injected into one ifg.
np.random.seed(0)
N = 8
true_phi = np.linspace(0, 6, N)              # 6 rad over 8 epochs
pairs = [(i, j) for i in range(N) for j in range(i + 1, min(i + 4, N))]
A = build_A(N, pairs)
phi_obs = A @ true_phi + 0.1 * np.random.randn(len(pairs))
phi_obs_slipped = phi_obs.copy()
phi_obs_slipped[len(pairs) // 2] += 6 * np.pi   # synthetic 3-cycle slip

def invert_L2(A, b):
    # drop column 0, solve, then prepend phi_0 = 0
    phi_rest, *_ = np.linalg.lstsq(A[:, 1:], b, rcond=None)
    return np.concatenate(([0.0], phi_rest))

def invert_L1(A, b):
    # min sum |A x - b| via LP:
    #   min sum_i t_i  s.t.  A x - b <= t,  -(A x - b) <= t,  t >= 0
    n, m = A.shape[1] - 1, A.shape[0]
    A_red = A[:, 1:]
    c = np.concatenate([np.zeros(n), np.ones(m)])
    A_ub = np.vstack([
        np.hstack([ A_red, -np.eye(m)]),
        np.hstack([-A_red, -np.eye(m)]),
    ])
    b_ub = np.concatenate([b, -b])
    bounds = [(None, None)] * n + [(0, None)] * m
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return np.concatenate(([0.0], res.x[:n]))

phi_L2_clean   = invert_L2(A, phi_obs)
phi_L1_clean   = invert_L1(A, phi_obs)
phi_L2_slipped = invert_L2(A, phi_obs_slipped)
phi_L1_slipped = invert_L1(A, phi_obs_slipped)

# Per-ifg residuals (A @ phi - obs) under each inversion of the slipped
# observations. L2 spreads the slip's bias across many ifgs; L1
# concentrates it in (mostly) the one bad observation.
resid_L2 = A @ phi_L2_slipped - phi_obs_slipped
resid_L1 = A @ phi_L1_slipped - phi_obs_slipped
slip_idx = len(pairs) // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

# Left: L2 recovered phase, clean vs slipped, against truth.
axes[0].plot(true_phi,         "k-",  lw=2,  label="truth")
axes[0].plot(phi_L2_clean,    "o-",  alpha=0.7, label="L2 clean")
axes[0].plot(phi_L2_slipped, "^--", color="C3",
              label=r"L2 with $+6\pi$ slip")
axes[0].set_title("L2 inversion")
axes[0].set_xlabel("epoch index"); axes[0].set_ylabel(r"$\varphi$ (rad)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Middle: same thing for L1.
axes[1].plot(true_phi,         "k-",  lw=2,  label="truth")
axes[1].plot(phi_L1_clean,    "s-",  alpha=0.7, label="L1 clean")
axes[1].plot(phi_L1_slipped, "v--", color="C2",
              label=r"L1 with $+6\pi$ slip")
axes[1].set_title("L1 inversion")
axes[1].set_xlabel("epoch index"); axes[1].set_ylabel(r"$\varphi$ (rad)")
axes[1].legend(); axes[1].grid(alpha=0.3)

# Right: per-ifg residuals from the slipped inversions.
x = np.arange(len(pairs))
axes[2].bar(x - 0.2, resid_L2, width=0.4, color="C3", label="L2 residual")
axes[2].bar(x + 0.2, resid_L1, width=0.4, color="C2", label="L1 residual")
axes[2].axvline(slip_idx, color="k", ls="--", lw=1,
                 label=f"injected slip @ ifg #{slip_idx}")
axes[2].axhline(0, color="k", lw=0.5)
axes[2].set_xlabel("ifg index")
axes[2].set_ylabel(r"$A\varphi - \varphi_{\rm obs}$ (rad)")
axes[2].set_title("per-ifg residuals (slipped network)")
axes[2].legend(); axes[2].grid(alpha=0.3, axis="y")

plt.show()
""")

md(r"""
### 3.3 Run the L1 inversion on the full network

Now the production inversion on the full bandwidth-3 multi-year network:
309 unwrapped ifgs spanning 105 epochs, 2016 → 2024.
`dolphin.timeseries.run` does the same kind of inversion as Section 3.2
but over the whole scene, with the L1 norm, in blocks.

Reference pixel is PMAR (the GPS station 10 km south of the volcano).
Output `velocity.tif` is in m/yr LOS — positive = motion toward the
satellite = uplift.
""")

code(r"""
UNW_DIR = HERE / "three_sisters" / "dolphin" / "unwrapped"
TEMP_COH_FULL = (HERE / "three_sisters" / "dolphin" / "interferograms"
                  / "temporal_coherence_average_20160722_20240622.tif")

unw_paths = sorted(UNW_DIR.glob("*.unw.tif"))
cc_paths  = [p.parent / p.name.replace(".unw.tif", ".unw.conncomp.tif")
              for p in unw_paths]
print(f"{len(unw_paths)} unwrapped ifgs available")
""")

code(r"""
# Reference pixel = PMAR GPS station, ~10 km south of the dome.
PMAR_LAT, PMAR_LON = 43.991, -121.687
HUSB_LAT, HUSB_LON = 44.120, -121.849      # validation target
WAVELENGTH         = 0.05547                # S1 C-band, m

# xarray opens the geotiff with its CRS wkt attached as a coordinate;
# pyproj turns lat/lon into the same projected coords, then we look up
# the nearest x/y in the DataArray.
from pyproj import Transformer

unw_da = xr.open_dataarray(unw_paths[0]).squeeze()
to_utm = Transformer.from_crs(
    "EPSG:4326", unw_da.spatial_ref.attrs["crs_wkt"], always_xy=True)

def latlon_to_pixel(lat, lon, da):
    x, y = to_utm.transform(lon, lat)
    row = int(np.argmin(np.abs(da.y.values - y)))
    col = int(np.argmin(np.abs(da.x.values - x)))
    return row, col

ref_rc  = latlon_to_pixel(PMAR_LAT, PMAR_LON, unw_da)
husb_rc = latlon_to_pixel(HUSB_LAT, HUSB_LON, unw_da)
print(f"PMAR pixel: {ref_rc}")
print(f"HUSB pixel: {husb_rc}")
""")

code(r"""
# dolphin.timeseries.run:
#   the production inversion. Takes the list of unwrapped ifgs +
#   conncomp masks + a quality raster, builds A, and solves
#   A.phi = dphi_obs at every pixel with method=L1. Writes per-epoch
#   displacement_*.tif and a velocity.tif into output_dir. wavelength
#   converts radians to meters of LOS. reference_point pins absolute
#   phase to one pixel.
from dolphin.timeseries import run as run_timeseries, InversionMethod
import contextlib, io, shutil

ts_out = WORK / "timeseries"
shutil.rmtree(ts_out, ignore_errors=True)        # always start fresh

# run_timeseries doesn't have a verbose flag; redirect its stdout so
# the notebook output stays clean.
with contextlib.redirect_stdout(io.StringIO()):
    run_timeseries(
        unwrapped_paths=unw_paths,           # all 309 unwrapped ifgs
        conncomp_paths=cc_paths,             # snaphu conncomp masks (same order)
        output_dir=ts_out,                   # displacement_*.tif + velocity.tif go here
        quality_file=TEMP_COH_FULL,          # multi-year temp coh, used for masking
        method=InversionMethod.L1,           # L1 norm, robust to bad ifgs
        run_velocity=True,                   # also fit a per-pixel linear velocity
        reference_point=ref_rc,              # PMAR GPS station pixel (zero phase here)
        wavelength=WAVELENGTH,               # 0.05547 m = S1 C-band; converts rad -> m
        num_threads=4,                       # block-parallel inversion
        velocity_file=ts_out / "velocity.tif",  # explicit velocity output path
    )
""")

md(r"""
### 3.4 Validate against GNSS

dolphin just wrote a per-acquisition `displacement_*.tif` time series.
Before plotting the final velocity raster, overlay dolphin's recovered
time series at the HUSB GPS station against HUSB's own independent
GNSS measurement.

For an ascending Sentinel-1 pass we project the GPS east-north-up
displacements into LOS via a precomputed unit vector for each station.
""")

code(r"""
# Load dolphin's per-acquisition displacement rasters at the HUSB pixel.
# dolphin names them <ref_date>_<acq_date>.tif at the top of ts_out.
# Glob pattern [0-9]*_*.tif excludes residuals_*, velocity.tif, and the
# .vrt sidecars.
from datetime import datetime

disp_paths = sorted(ts_out.glob("[0-9]*_*.tif"))

def parse_acq_date(p):
    # filename is YYYYMMDD_YYYYMMDD.tif - second date = acquisition.
    return datetime.strptime(p.stem.split("_")[1][:8], "%Y%m%d")

def decyear(d):
    return d.year + (d.timetuple().tm_yday - 1) / 365.25

acq_dates  = [parse_acq_date(p) for p in disp_paths]
disp_decyr = np.array([decyear(d) for d in acq_dates])

# LOS displacement at the HUSB pixel for every acquisition (meters).
hr, hc = husb_rc
insar_los = np.array([
    float(xr.open_dataarray(p).squeeze().values[hr, hc])
    for p in disp_paths
])
print(f"loaded {len(disp_paths)} per-epoch displacement files at HUSB pixel")
""")

code(r"""
import pandas as pd

# HUSB GPS station LOS unit vector (east, north, up) - precomputed for
# track 115 ascending at burst centre incidence.
HUSB_LOS_UV = np.array([0.631870, -0.110014, 0.767227])

# Read the UNR-NGL tenv3 file: whitespace-separated table with
# YYYY.YYYY decimal year + east/north/up displacements in meters.
husb = pd.read_csv(DATA / "gnss" / "HUSB.tenv3", sep=r"\s+")
gps_t   = husb["yyyy.yyyy"].values
gps_los = (husb["__east(m)"]  * HUSB_LOS_UV[0]
           + husb["_north(m)"] * HUSB_LOS_UV[1]
           + husb["____up(m)"] * HUSB_LOS_UV[2]).values

# Reference both series to the same epoch (dolphin's first one) and
# crop the GPS to dolphin's date span so the y-scales are comparable.
ref_decyr  = decyear(parse_acq_date(disp_paths[0]))
gps_at_ref = np.interp(ref_decyr, gps_t, gps_los)
gps_los    = gps_los - gps_at_ref
sel        = (gps_t >= ref_decyr) & (gps_t <= disp_decyr.max() + 0.5)
gps_t, gps_los = gps_t[sel], gps_los[sel]
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(gps_t, gps_los * 1000, "o", color="gray", alpha=0.4, ms=3,
        label="HUSB GPS")

ax.plot(disp_decyr, insar_los * 1000, "s", color="C0", ms=4,
        label="dolphin LOS @ HUSB")

ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("year")
ax.set_ylabel("LOS displacement (mm, + toward sat)")
ax.set_title("HUSB: dolphin time series vs independent GPS")
ax.legend(); ax.grid(alpha=0.3)
plt.show()
""")

md(r"""
### 3.5 Final velocity map

Per-pixel linear velocity (m/yr LOS, converted to mm/yr for display).
Positive = motion toward the satellite = uplift.
""")

code(r"""
import contextily as cx

vel = xr.open_dataarray(ts_out / "velocity.tif").squeeze() * 1000.0
vel = vel.where(np.isfinite(vel) & (vel != 0))

fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

# Plot velocity first to establish the axis extent.
vel.plot.imshow(ax=ax, cmap="RdBu_r", vmin=-10, vmax=10, alpha=0.65,
                zorder=2,
                cbar_kwargs=dict(label="mm/yr LOS  (+ = uplift)"))
xlim, ylim = ax.get_xlim(), ax.get_ylim()   # lock the velocity-raster extent

# Topo basemap behind the velocity (zorder=0). We're in UTM 10N (the
# velocity raster's CRS); contextily fetches OpenStreetMap-style tiles
# and warps to that CRS.
cx.add_basemap(ax, crs=vel.spatial_ref.attrs["crs_wkt"],
                source=cx.providers.OpenTopoMap, attribution_size=6,
                zorder=0)
ax.set_xlim(xlim); ax.set_ylim(ylim)         # don't let basemap expand it

# Project GPS station lat/lon into the raster's UTM coords for plotting.
# Markers + labels on top of velocity (zorder=3).
husb_x, husb_y = to_utm.transform(HUSB_LON, HUSB_LAT)
pmar_x, pmar_y = to_utm.transform(PMAR_LON, PMAR_LAT)
ax.plot(husb_x, husb_y, marker="^", ms=12, color="black",
        markeredgecolor="white", markeredgewidth=1.5, zorder=3)
ax.annotate("HUSB", (husb_x, husb_y), xytext=(8, 8),
            textcoords="offset points", fontsize=10, fontweight="bold",
            zorder=3)
ax.plot(pmar_x, pmar_y, marker="*", ms=14, color="lime",
        markeredgecolor="black", markeredgewidth=1.0, zorder=3)
ax.annotate("PMAR (REF)", (pmar_x, pmar_y), xytext=(8, 8),
            textcoords="offset points", fontsize=10, fontweight="bold",
            color="darkgreen", zorder=3)

# Three Sisters summits (USGS coordinates). North Sister sits just
# above the AOI's northern edge so we skip it on this map.
SUMMITS = {
    "South Sister":  (44.1035, -121.7686),
    "Middle Sister": (44.1351, -121.7691),
}
for name, (lat, lon) in SUMMITS.items():
    sx, sy = to_utm.transform(lon, lat)
    ax.plot(sx, sy, marker="v", ms=10, color="red",
            markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax.annotate(name, (sx, sy), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color="darkred",
                zorder=3)

ax.set_title("LOS velocity from dolphin.timeseries.run  "
             f"(L1, bandwidth-3, {len(unw_paths)} ifgs, 2016-2024)")
ax.set_xlabel("UTM E (m)"); ax.set_ylabel("UTM N (m)")
plt.show()
""")


# ===========================================================================
# Section 4 — Wrap-up
# ===========================================================================

md(r"""
## 4. Wrap-up

What we just did by hand is what `dolphin run dolphin_config.yaml` does
on its own. A quick recap of the three stages:

- **Step 1 — PS / phase linking.** Use amplitude statistics ($\mu_A$,
  $\sigma_A$) to classify pixels and to find SHP neighborhoods for the
  noisy DS ones. dolphin estimates a noise-mitigated phase per
  acquisition for every pixel and writes per-pixel temporal coherence
  telling you how closely the recovered phase matches the observed
  phases.
- **Step 2 — Unwrap.** Form ifgs over a network of acquisition pairs
  and recover the integer cycle count at each pixel. snaphu is the
  default; spurt is the option for noisier scenes.
- **Step 3 — Time-series inversion.** Stack the (over-determined) ifg
  network into a linear system and solve with L1 so the occasional
  bad ifg doesn't smear bias through the whole time series. Pin a
  reference epoch and (optionally) a reference pixel.

A few things we skipped or simplified, so you know where to look for
the real thing:

- **Burst stitching happens in dolphin.** We pre-stitched the two
  bursts in `stage_cslcs.py` to keep the notebook running on a single
  combined stack. Production dolphin runs PS + PL per burst
  (`t115_*` subdirs in `dolphin/`) and uses `dolphin.stitching` to
  merge unwrapped ifgs onto a common grid afterwards.
- **We unwrapped one ifg, not a network.** A real `dolphin run`
  unwraps the entire bandwidth-3 network (~300 ifgs for this stack),
  with optional Goldstein preprocessing and per-ifg conncomp QA. Our
  §3.3 inversion reads those pre-computed unwraps from the shipped
  tarball.
- **One ministack, no compressed-SLC chaining.** Production dolphin
  runs phase linking in sequential ministacks (~10-20 acquisitions
  each) and chains them via compressed SLCs to keep memory bounded
  on long stacks. We collapsed to a single 15-acquisition ministack
  for simplicity.
- **Only L1 + snaphu.** We mentioned phass, ICU, spurt, and
  whirlwind, and we mentioned the L2 option, but only ran the L1 +
  snaphu defaults.
- **Reference pixel was hardcoded** to PMAR. Production dolphin can
  auto-pick a high-quality reference using the `quality_file` raster.

A few things worth doing on your own scene:

- Temporal coherence is usually the first place to look. If it's
  consistently low across the AOI, consider whether a longer stack,
  different season filter, smaller AOI, or different sensor would
  help before spending time tuning knobs.
- Mask the velocity by temp_coh before interpreting it. A threshold
  around 0.5-0.7 is common, but pick what fits your scene.
- If something looks off, the run-by-run knobs that tend to help most
  are `amp_dispersion_threshold`, `shp_alpha`, `half_window`, the
  unwrap method, and the Goldstein toggle/$\alpha$.
- Compare to an independent dataset (GPS, leveling, a MintPy run on
  the same stack) when you can — it's the cleanest sanity check that
  the recovered velocity is real and not a processing artifact.

For tuning beyond what's here, the dolphin docs cover
`phase_linking.beta` (Zwieback regularization for short stacks),
`ministack_size`, `compressed_slc_plan`, spurt internals (`s_cost_type`,
`use_tiles`, ...), and GPU acceleration.

---

dolphin: https://github.com/isce-framework/dolphin

References for Three Sisters: Wicks et al. 2002 (original uplift
discovery); Riddick & Schmidt 2011 (1992-2010 InSAR + GPS);
[Staniewicz et al. 2025](https://arxiv.org/abs/2511.12051) — DISP-S1
product paper using this AOI, source of the workflow and AOI figures
above.

---

*Notebook drafted by Zach Hoppinen and Scott Staniewicz with help from
Claude (Opus 4.7), EarthScope 2026 ISCE+ course.*
""")


# ===========================================================================
# Build
# ===========================================================================

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}
with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT}  ({len(cells)} cells)")
