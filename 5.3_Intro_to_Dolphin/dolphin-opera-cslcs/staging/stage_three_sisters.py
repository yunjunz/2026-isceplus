"""
Build the three-sisters-cslc tarball used by the dolphin OPERA-CSLC tutorial.

This is a ONE-TIME prep script. The tarball it produces gets uploaded to S3
and the student notebook downloads from there. Students never run this.

What it does:
  1. Queries ASF for OPERA L2 CSLC granules on burst T115-245676-IW2
     (single burst covering South Sister volcano, Oregon).
  2. Picks 12 dates: 9 summer/fall (high coherence) + 3 winter (snow ->
     low coherence) so the tutorial's "temporal coherence is a real QA gate"
     lesson has teeth.
  3. Downloads each granule via asf_search (uses ~/.netrc for Earthdata auth).
  4. Crops each HDF5 to a ~20x20 km AOI around the paper's stated uplift peak
     (~5 km west of South Sister), preserving the CF-compliant structure
     dolphin needs to read.
  5. Tars + gzips the cropped files into three-sisters-cslc.tar.gz.

Reference site choice: Staniewicz et al. 2025, arXiv:2511.12051 (the DISP-S1
paper). Three Sisters is their canonical "challenging C-band, subtle uplift,
seasonal decorrelation" case study.

Usage:
    pip install asf_search h5py numpy pyproj
    # Confirm ~/.netrc has urs.earthdata.nasa.gov credentials
    python stage_three_sisters.py
"""

from __future__ import annotations

import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import asf_search as asf
import h5py
import numpy as np
import pyproj


# ----- Configuration ---------------------------------------------------------

# Single Sentinel-1 burst covering South Sister + the paper's uplift peak.
# Verified via ASF CMR query 2026-05-27. T115 descending, sub-swath IW2.
BURST_ID = "T115-245676-IW2"

# AOI in EPSG:4326. Centered ~5 km west of South Sister (the paper's stated
# peak uplift location). ~20 km E-W x ~22 km N-S.
AOI_LON_MIN, AOI_LON_MAX = -121.93, -121.73
AOI_LAT_MIN, AOI_LAT_MAX = 44.00, 44.20

# Three Sisters falls in UTM zone 10N (covers 126W to 120W). Stored CSLCs
# advertise this in /data/projection, but we set it here for safety/clarity.
CSLC_EPSG = 32610

# Target dates for the 12-acquisition stack. The script snaps each target to
# the closest actual S1A acquisition on this burst. The mix is deliberate:
# - 9 summer/fall dates spread across ~18 months -> clean velocity signal
# - 3 winter dates -> snow-driven coherence collapse, used in section 5.3
#   ("temporal coherence is your QA gate") and section 6 ("defaults can fail
#   silently") of the revised tutorial outline.
TARGET_DATES = [
    # 9 high-coherence (May/Jul/Aug/Sep/Oct/Nov)
    "2023-05-23",
    "2023-07-22",
    "2023-08-15",
    "2023-09-08",
    "2024-07-04",
    "2024-08-09",
    "2024-09-14",
    "2024-10-08",
    "2024-11-13",
    # 3 winter (snow expected at 1500-3000 m elevation in Cascades)
    "2023-01-11",
    "2024-02-12",
    "2024-12-19",
]

# Where things live during staging.
WORK = Path(__file__).resolve().parent / "stage_work"
DOWNLOAD = WORK / "raw"           # full-burst CSLCs from ASF
CROPPED = WORK / "cropped"        # AOI-cropped CSLCs
TARBALL = Path(__file__).resolve().parent / "three-sisters-cslc.tar.gz"


# ----- Helpers ---------------------------------------------------------------

def step(msg: str) -> None:
    """One-line progress log so the user can see what stage is running."""
    print(f"[stage] {msg}", flush=True)


def aoi_to_utm_bounds(epsg: int) -> tuple[float, float, float, float]:
    """Project the lon/lat AOI box into the CSLC's UTM CRS.
    Returns (x_min, y_min, x_max, y_max) in meters.
    """
    # Transform all four corners and take the enclosing rectangle. Just using
    # two corners would under-cover near the edges of the UTM zone.
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    corners_lon = [AOI_LON_MIN, AOI_LON_MAX, AOI_LON_MIN, AOI_LON_MAX]
    corners_lat = [AOI_LAT_MIN, AOI_LAT_MIN, AOI_LAT_MAX, AOI_LAT_MAX]
    xs, ys = transformer.transform(corners_lon, corners_lat)
    return min(xs), min(ys), max(xs), max(ys)


# ----- Step 1: query ASF -----------------------------------------------------

def query_available() -> list[asf.ASFProduct]:
    """List every OPERA L2 CSLC granule for this burst across the time window."""
    step("Querying ASF for available CSLCs on burst " + BURST_ID + " ...")
    # Use a point inside the burst footprint to drive the spatial filter;
    # ASF's burst-id filter via opera-utils requires extra setup so this
    # simpler form is fine.
    center_lon = (AOI_LON_MIN + AOI_LON_MAX) / 2
    center_lat = (AOI_LAT_MIN + AOI_LAT_MAX) / 2
    results = list(asf.search(
        dataset=asf.DATASET.OPERA_S1,
        processingLevel=asf.PRODUCT_TYPE.CSLC,
        relativeOrbit=115,
        flightDirection="DESCENDING",
        start="2022-11-01T00:00:00Z",
        end="2025-02-01T00:00:00Z",
        intersectsWith=f"POINT({center_lon} {center_lat})",
    ))
    # Filter to the exact burst we want (in case other bursts on T115 also
    # intersect the AOI - they shouldn't for a 20km box, but be defensive).
    results = [r for r in results if BURST_ID in r.properties["fileName"]]
    step(f"  -> {len(results)} granules match burst {BURST_ID}")
    return results


# ----- Step 2: pick 12 dates -------------------------------------------------

def pick_dates(available: list[asf.ASFProduct]) -> list[asf.ASFProduct]:
    """Snap each TARGET_DATES entry to the nearest actual acquisition."""
    by_date = {r.properties["startTime"][:10]: r for r in available}
    avail_dt = {datetime.fromisoformat(d).replace(tzinfo=timezone.utc): r
                for d, r in by_date.items()}
    chosen = []
    used = set()
    step("Snapping target dates to nearest actual acquisitions:")
    for target in TARGET_DATES:
        target_dt = datetime.fromisoformat(target).replace(tzinfo=timezone.utc)
        best_dt = min(
            (d for d in avail_dt if d not in used),
            key=lambda d: abs((d - target_dt).total_seconds()),
        )
        used.add(best_dt)
        chosen.append(avail_dt[best_dt])
        days_off = (best_dt - target_dt).days
        step(f"  {target}  ->  {best_dt.date()}  ({days_off:+d}d)")
    return chosen


# ----- Step 3: download ------------------------------------------------------

def download(chosen: list[asf.ASFProduct]) -> None:
    """Download via asf_search. Uses ~/.netrc for Earthdata Login."""
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    # ASFSession with no creds falls back to ~/.netrc automatically.
    session = asf.ASFSession()
    for r in chosen:
        out = DOWNLOAD / r.properties["fileName"]
        if out.exists() and out.stat().st_size > 0:
            step(f"  cached: {out.name}  ({out.stat().st_size/1e6:.0f} MB)")
            continue
        step(f"  downloading {out.name} ...")
        r.download(path=str(DOWNLOAD), session=session)


# ----- Step 4: crop one CSLC -------------------------------------------------

def crop_cslc(src: Path, dst: Path) -> None:
    """Crop an OPERA L2 CSLC HDF5 to AOI, preserving CF/dolphin structure.

    Strategy: copy the file group-by-group, but replace /data with AOI-cropped
    versions of VV, x_coordinates, y_coordinates. Drop /static_layers entirely
    (large, not needed for the tutorial's dolphin pipeline).
    """
    x_min, y_min, x_max, y_max = aoi_to_utm_bounds(CSLC_EPSG)

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        # 1. Locate the coordinate arrays. OPERA CSLC v1.1 puts them at
        #    /data/x_coordinates and /data/y_coordinates per the product spec.
        if "data/x_coordinates" in fin:
            x_path, y_path = "data/x_coordinates", "data/y_coordinates"
        elif "data/x" in fin:
            x_path, y_path = "data/x", "data/y"
        else:
            raise RuntimeError(f"Cannot find x/y coordinate arrays in {src.name}")

        x = fin[x_path][:]
        y = fin[y_path][:]

        # 2. Compute index slices that bracket the AOI. Note y typically
        #    decreases with row index (north-up rasters) so we sort first.
        x_in = np.where((x >= x_min) & (x <= x_max))[0]
        y_in = np.where((y >= y_min) & (y <= y_max))[0]
        if len(x_in) < 10 or len(y_in) < 10:
            raise RuntimeError(
                f"AOI gives too few pixels in {src.name} "
                f"(x:{len(x_in)}, y:{len(y_in)}). Check AOI vs CSLC footprint."
            )
        x_slice = slice(x_in.min(), x_in.max() + 1)
        y_slice = slice(y_in.min(), y_in.max() + 1)

        # 3. Walk the source file. For groups we want to preserve verbatim
        #    (everything except /data and /static_layers), use h5py.copy().
        #    For /data, hand-build cropped datasets.
        for name in fin.keys():
            if name == "static_layers":
                # Skip - layover/shadow/local incidence are big and unused
                # by the dolphin pipeline run by this tutorial.
                continue
            if name == "data":
                # Recreated below.
                continue
            fin.copy(name, fout)

        # 4. Rebuild /data with cropped arrays.
        data_in = fin["data"]
        data_out = fout.create_group("data")
        # 4a. Copy non-grid items in /data verbatim (e.g. /data/projection,
        #     /data/VV.coordinates, any LUTs that aren't on the SLC grid).
        for k in data_in.keys():
            if k in {"VV", "VH", "HH", "HV", x_path.split("/")[-1], y_path.split("/")[-1]}:
                continue  # handled separately below
            data_in.copy(k, data_out)
        # 4b. Cropped coordinate arrays.
        data_out.create_dataset(x_path.split("/")[-1], data=x[x_slice])
        data_out.create_dataset(y_path.split("/")[-1], data=y[y_slice])
        # 4c. Cropped polarization arrays. Read the cropped block from disk
        #     directly with h5py slicing (no full-array load needed).
        for pol in ("VV", "VH", "HH", "HV"):
            if pol not in data_in:
                continue
            arr = data_in[pol][y_slice, x_slice]
            ds = data_out.create_dataset(
                pol, data=arr,
                # Match OPERA's compression so file sizes stay sane.
                compression="gzip", compression_opts=4, shuffle=True,
                chunks=True,
            )
            # Preserve per-dataset attributes (grid_mapping, units, etc).
            for k, v in data_in[pol].attrs.items():
                ds.attrs[k] = v

        # 5. Copy group-level attributes for /data and root.
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for k, v in data_in.attrs.items():
            data_out.attrs[k] = v


def crop_all() -> None:
    CROPPED.mkdir(parents=True, exist_ok=True)
    files = sorted(DOWNLOAD.glob("OPERA_L2_CSLC-S1_*.h5"))
    step(f"Cropping {len(files)} CSLCs to AOI...")
    for src in files:
        dst = CROPPED / src.name
        if dst.exists() and dst.stat().st_size > 0:
            step(f"  cached crop: {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")
            continue
        step(f"  cropping {src.name} ...")
        crop_cslc(src, dst)
        step(f"    -> {dst.stat().st_size/1e6:.1f} MB")


# ----- Step 5: tarball -------------------------------------------------------

def build_tarball() -> None:
    files = sorted(CROPPED.glob("OPERA_L2_CSLC-S1_*.h5"))
    step(f"Building {TARBALL.name} from {len(files)} cropped files...")
    with tarfile.open(TARBALL, "w:gz", compresslevel=6) as tar:
        for p in files:
            tar.add(p, arcname=f"three-sisters/{p.name}")
    step(f"  tarball size: {TARBALL.stat().st_size/1e6:.0f} MB")


# ----- Step 6: size report ---------------------------------------------------

def report() -> None:
    full = sum(p.stat().st_size for p in DOWNLOAD.glob("*.h5"))
    crop = sum(p.stat().st_size for p in CROPPED.glob("*.h5"))
    tar_size = TARBALL.stat().st_size if TARBALL.exists() else 0
    print()
    print(f"  full-burst total : {full/1e9:.2f} GB")
    print(f"  cropped total    : {crop/1e6:.0f} MB")
    print(f"  final tarball    : {tar_size/1e6:.0f} MB  ({TARBALL})")
    print()


# ----- Main ------------------------------------------------------------------

def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    available = query_available()
    chosen = pick_dates(available)
    download(chosen)
    crop_all()
    build_tarball()
    report()


if __name__ == "__main__":
    main()
