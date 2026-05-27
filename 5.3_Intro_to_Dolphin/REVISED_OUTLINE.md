# 5.3 Intro to Dolphin — Revised Notebook Outline

**Status:** Draft for discussion. Not yet implemented.

## Pedagogical principles (the rules this outline is trying to follow)

1. **Students always see their own outputs.** No static PNGs of someone else's run. Every figure is loaded live from the dolphin output folder the student just created.
2. **Walk the pipeline through its artifacts.** Don't run `dolphin run` as a single opaque step. After it finishes, load and visualize each intermediate (PS → phase-linked SLCs → temporal coherence → wrapped IFGs → unwrapped IFGs → timeseries → velocity). Students learn the pipeline by looking at what each stage produces.
3. **Every knob is tied to a visible effect on real data.** When a parameter is introduced, it's shown changing an output the student already understands. No prose-only knob descriptions.
4. **The science comes before the software.** A short primer on PS vs DS, what phase linking estimates, and why mini-stacks exist — so the rest of the notebook reinforces concepts, not just clicks.
5. **One real QA gate.** Temporal coherence is loaded, plotted, and used to decide whether to trust the unwrap. Students leave knowing how to tell a good run from a bad one.
6. **Defaults can fail silently — show how.** At least one "if you accept the default here, you get garbage on this scene" demonstration, so students learn to inspect rather than trust.

---

## Outline (canonical OPERA-CSLC notebook)

### Section 0 — Setup (~3 cells)

- Title, learning objectives stated concretely:
  > By the end of this notebook you'll be able to: (1) run dolphin end-to-end on OPERA CSLCs, (2) read each output folder and tell what stage produced it, (3) pick reasonable values for the half-dozen most-tuned knobs and predict what they'll change, (4) use temporal coherence to decide whether a run is trustworthy.
- Imports + matplotlib inline.
- **Earthdata auth check**: actively check `~/.netrc` exists and contains `urs.earthdata.nasa.gov`. If not, raise a clear error with a one-line setup instruction. (Currently the notebook hand-waves this.)
- `dolphin.show_versions()` reproducibility check (keep from current notebook).
- One-line GPU note + a comment that this notebook is sized to run on CPU.

### Section 1 — The science in 3 cells (NEW)

Short conceptual primer so the rest of the notebook reinforces understanding rather than introducing terms cold.

- **1.1 PS vs DS**: A 2-panel synthetic figure — a "PS-like" pixel (stable amplitude across epochs) vs a "DS-like" pixel (varying amplitude, but coherent on average over a neighborhood). Tie to amplitude dispersion = std/mean. One sentence per pixel type on how dolphin handles each.
- **1.2 What phase linking does**: One paragraph + one figure. "Phase linking takes a stack of N noisy SLCs and estimates a single 'best' phase per epoch by jointly inverting the N×N coherence matrix at each pixel." Show a coherence matrix and the resulting per-epoch phases.
- **1.3 Why mini-stacks**: Explain in 3 sentences. Show a flowchart of how SLCs are batched + compressed across mini-stacks. Use the existing `docs/flowchart.png` here (it's a good figure, just needs a narrative).

### Section 2 — Input data (~4 cells)

- Download a **smaller** subset than the current notebook. Pick one burst, ~12 dates over one year (a third of current). Cite the goal explicitly: "small enough to re-run after you change a knob in section 6."
- Decode the OPERA filename (keep current notebook's nice anatomy explanation: T165, burst ID, IW2, datetime).
- **Inspect one input SLC**: open with `dolphin.io.load_gdal`, plot amplitude (dB) and phase side by side. This anchors what an "SLC" actually looks like before any processing.
- **Show co-registration**: amplitude of two epochs overlaid (or animated/blink). Demonstrates what "coregistered stack" means at the pixel level.

### Section 3 — Configure dolphin (~4 cells)

- **3.1 Generate defaults**: `!dolphin config --slc-files input_slcs/*.h5 --subdataset /data/VV` (the minimal call from current notebook).
- **3.2 Inspect the YAML, but selectively**: don't print all 342 lines. Print only the 6 sections students will tune (`input_options`, `ps_options`, `phase_linking`, `interferogram_network`, `output_options`, `unwrap_options`), with a one-sentence header per section saying what knob in there matters.
- **3.3 Build a production config**: a second `dolphin config` call with the realistic flag set from current notebook (mask file, strides, network bandwidth, parallelism). **Show a diff** between minimal and production YAML so the change is visible, not just the result.
- **3.4 Common-knob cheat sheet**: a 6-row table with knob, default, what-it-controls, when-to-change. Keep it scannable; defer deep tuning to docs.

### Section 4 — Run dolphin (1 cell, with feedback)

- Remove `%%capture`. Run `dolphin run dolphin_config.yaml` and let the log stream live. Alternative: tee to `dolphin.log` and tail the last 30 lines after the cell finishes so students see a heartbeat.
- One markdown cell beforehand setting expectations: "this takes ~N minutes; you'll see log lines for PS → phase linking → interferograms → unwrap → timeseries."

### Section 5 — Walk the outputs (the big addition, ~7 cells)

This is the section that replaces "look at this static velocity PNG" with a guided tour of what dolphin produced. Each subsection: load one artifact, plot it, explain what it represents, name the knob that controlled it.

- **5.0 Output folder map**: `!tree -L 2 dolphin/` (or `pathlib` equivalent if `tree` isn't installed). One sentence per folder. Fix the burst-ID mismatch in the current notebook's appendix while we're at it.
- **5.1 PS mask** (`phase_linking/PS/ps_pixels.tif`): load, plot, report PS fraction. Tie to knob: `amp_dispersion_threshold`.
- **5.2 Phase-linked SLCs** (`phase_linking/linked_phase/*.slc.tif`): load 2-3 epochs, plot phase. Compare side-by-side with the input SLC phase from section 2 — phase-linking should look visibly denoised. Tie to knob: `half_window`.
- **5.3 Temporal coherence** (`phase_linking/linked_phase/temporal_coherence_*.tif`): load, plot map + histogram. **This is the QA gate.** Explain: values near 1 mean phase-linking found consistent phase across the stack; low values mean noisy/decorrelated. Tie to knob: `shp_alpha`, `half_window`. Pick a threshold and use it to mask the velocity later.
- **5.4 Wrapped interferograms** (`interferograms/*.int.tif`): plot 3-4 from the network. **Also draw the network graph** (dates on x-axis, edges per pair) — the abstract phrase "max_bandwidth=3" becomes concrete. Tie to knob: `interferogram_network.max_bandwidth`.
- **5.5 Unwrapped interferograms** (`unwrapped/*.unw.tif`): same set unwrapped, plotted with a diverging colormap. Report fraction of pixels successfully unwrapped (per-IFG and mean) — this is the second QA signal. Tie to knob: `unwrap_method`.
- **5.6 Timeseries** (`timeseries/*.tif`): plot per-epoch cumulative phase relative to the reference. Mark the reference pixel location on the plot. Pull a 1D time series at 2-3 user-selected pixels (a stable point and a known-moving point if the AOI has one).
- **5.7 Velocity** (`timeseries/velocity.tif`): live plot, with the temporal-coherence mask applied. Compare to running without the mask to show why QA matters.

### Section 6 — Knob effects on real data (~4 cells)

Pick 2-3 high-leverage knobs and re-run on a small subset so students see what changes. Each one is a "before/after" panel.

- **6.1 `amp_dispersion_threshold`**: re-run *just the PS step* with thresholds 0.15 vs 0.25 vs 0.50. Show PS-count maps. Tie back to the "defaults can fail silently" principle: if your scene's max amp dispersion is below 0.25, the default flags every pixel as PS.
- **6.2 `half_window`**: re-run phase linking with a small (5×5) vs large (21×21) window. Show side-by-side phase-linked SLCs and temporal coherence. Discuss the tradeoff: bigger window = more samples = better statistics, but smears across distinct land covers.
- **6.3 `interferogram_network.max_bandwidth`**: build a bandwidth-1 (sequential) vs bandwidth-3 vs single-reference network. Draw all three network graphs. Show the resulting velocity from each. Discuss: more edges = more redundancy for unwrap, but more compute.
- *(Optional stretch)* **6.4 `unwrap_method`**: snaphu vs spurt on the same wrapped IFGs. Show the difference. Note compute cost.

Avoid going deeper than this in the notebook itself. Knobs we **don't** cover in 6: `beta`, `ministack_size`, `compressed_slc_plan`, `shp_method`, spurt internals, `output_reference_idx`. These get a one-paragraph "for deeper tuning see the dolphin docs" pointer at the end of section 7.

### Section 7 — Pitfalls and QA gates (~2 cells)

Compact pitfall list, each one paragraph:

- **The wavelength field changes units.** `wavelength: null` (or unrecognized sensor) leaves outputs in radians; setting it converts to meters of LOS displacement. Pick deliberately.
- **`amp_dispersion_threshold` interacts with your scene.** A low-amplitude-dispersion scene (e.g. uniform vegetation) needs a lower threshold or every pixel becomes PS.
- **Temporal coherence is your trust signal.** Always plot it. Always mask the velocity by it before interpreting.
- **`dolphin.log`** lives in the work directory and has the full per-stage timing + warnings. When something looks wrong, read this first.

Close with a pointer to deeper knobs in the dolphin docs (`phase_linking.beta`, `ministack_size`, spurt options, etc.) and one line of "your specific application may justify these — see the docs."

### Section 8 — Exercises (~1 markdown cell with 3-4 prompts)

Concrete, runnable. Examples:

1. Re-run with `--sx 12 --sy 6`. Compare runtime, output file size, and the smallest feature you can resolve in the velocity raster.
2. Lower the temporal coherence threshold from 0.7 to 0.5 in your QA mask. How much more area is "trusted" and does the velocity look noisier there?
3. Pick a SNOTEL station (or any GPS site) inside your AOI and extract dolphin's per-epoch phase at that pixel. Convert to LOS displacement and overlay against an independent timeseries.
4. *(Cross-module)* Compare this velocity map to MintPy's output from notebook 5.1 over the same stack. Where do they agree / disagree?

### Section 9 — Where to go next

- Cross-link: "if your input SLCs are radar-coordinate ISCE2 `topsApp` outputs instead of geocoded OPERA CSLCs, see the sibling notebook `isce2-dolphin/`."
- Cross-link to 5.1 (MintPy timeseries) and 5.2 (OPERA displacement products) for downstream use.
- Link to dolphin docs.

---

## Mirroring this structure to the ISCE2 radar-coord notebook

The same skeleton applies to `isce2-dolphin/dolphin-isce2.ipynb` with these substitutions:

- **Section 2 input**: keep the pre-staged S3 tar download but offer a smaller bundle if possible. Inspect a `.slc.tif` from `topsApp` output the same way (amplitude + phase).
- **Section 3 config**: keep the pre-built `dolphin_config.yaml` checked into the repo, but walk through it the same way (selective YAML print + cheat sheet). Show why the radar-coord case sets `subdataset: null` (single-band geotiffs) and why `strides` are asymmetric (range/azimuth pixel spacing differ).
- **Section 5 output walk**: same structure. Radar-coord outputs differ only in that there's no burst-stitching (one stack only), so the folder map is simpler — explicitly note this difference.
- **Section 6 knob exercises**: same three knobs work fine.
- **Section 7 pitfalls**: add one ISCE2-specific note — wavelength must be set explicitly for radar-coord inputs since dolphin can't infer the sensor from filenames the way it can for OPERA CSLCs.

The two notebooks should cross-link in their opening cells so students know which to start with based on their input data.

---

## Things explicitly **not** in this outline (out of scope for the course)

Borrowed-from-real-work knobs that would over-stuff a 90-minute notebook:

- `phase_linking.beta` (Zwieback regularization for short stacks)
- `ministack_size` overrides for short stacks
- `compressed_slc_plan`
- spurt deep tuning (`s_cost_type`, `use_tiles`, `temporal_coherence_threshold`, `run_ambiguity_interpolation`)
- programmatic YAML patching with `ruamel.yaml`
- runtime-switchable `UNWRAP_METHOD` framework
- `wavelength=null` for keeping outputs in radians for downstream SWE conversion

These get a single closing pointer ("for production tuning, see the dolphin docs and these specific knobs") so the curious student knows where to look without the notebook losing focus.

---

## Open questions for discussion

1. **Single notebook or keep two?** The outline above treats OPERA-CSLC and ISCE2-radar-coord as two parallel notebooks with the same skeleton. Alternative: one notebook with a branching "pick your input type" cell up front. The two-notebook approach is probably less confusing for first-time users; happy to discuss.
2. **How much "science before software" is too much?** Section 1 is three cells right now. Could compress to one cell with a single figure, or expand to a real mini-lecture with a synthetic worked example. Depends on what 5.1/5.2 already covered.
3. **Subset size.** I suggested cutting the OPERA download by ~3x for re-runability in section 6. Need to confirm this is enough data to make the knob-effect comparisons in section 6 visually meaningful (especially `interferogram_network.max_bandwidth` — needs enough dates).
4. **Section 6 ambition.** Re-running phase linking 3x for `amp_dispersion_threshold` is cheap. Re-running unwrap with spurt vs snaphu is not. Should section 6 stay at 3 knobs with a clear "this part takes longer" warning, or should we prepare cached outputs so students can compare without re-running?
5. **Assessment block.** Section 8 has exercises but no graded assessment criterion. The current ISCE2 README has an empty "Assessment:" field — do we want to put something concrete there, or leave it open?
