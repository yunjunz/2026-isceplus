"""
stage_cslcs.py — build the staged tarball used by the notebook.

End-to-end one-shot prep:
  1. Download OPERA L2 CSLCs from ASF DAAC for the requested bursts +
     date range. Needs an Earthdata netrc.
  2. Crop each H5 to a UTM bbox, writing a per-burst SLC geotiff to
     <out>/slc_tif/.
  3. Stitch per-burst SLCs onto one common grid per acquisition under
     <out>/slc_stitched/.
  4. (Optional) Drop acquisitions in months passed via --exclude-months
     so you can filter out snow-decorrelated winter/spring scenes.
  5. (Optional) Download daily ENU tenv3 GNSS time series from UNR-NGL
     to <out>/gnss/ for any stations passed via --gnss-stations.
  6. (Optional) Tar+gzip everything into one .tar.gz at --tarball PATH.
     Use --tarball-include to fold in extra paths (e.g. pre-computed
     dolphin/unwrapped/ and a temp_coh raster) so the shipped archive
     covers everything the notebook reads.

Example (the call used for the Three Sisters tutorial tarball):

    python stage_cslcs.py \\
        --bursts T115-245676-IW2 T115-245677-IW2 \\
        --start 2016-07-01 --end 2024-07-01 \\
        --bbox 587950 4866900 609130 4890550 \\
        --exclude-months 11 12 1 2 3 4 5 \\
        --gnss-stations HUSB PMAR \\
        --tarball three-sisters-cslc.tar \\
        --tarball-include three_sisters/dolphin/unwrapped \\
                          three_sisters/dolphin/interferograms/temporal_coherence_average_20160722_20240622.tif \\
        --out three_sisters/data/

--bbox is UTM (xmin ymin xmax ymax) in the burst's native CRS.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS


NGL_URL_FMT = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/NA/{stn}.NA.tenv3"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bursts", nargs="+", required=True,
                     help='OPERA burst IDs like "T115-245676-IW2"')
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--bbox",  nargs=4, type=float, required=True,
                     metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                     help="UTM bbox in the burst's native CRS")
    ap.add_argument("--out", type=Path, required=True,
                     help="Output directory (gets <out>/*.h5 + <out>/slc_tif/)")
    ap.add_argument("--gnss-stations", nargs="*", default=[],
                     help="UNR-NGL station codes to download .tenv3 series for "
                          "(e.g. HUSB PMAR). Goes to <out>/gnss/.")
    ap.add_argument("--exclude-months", nargs="*", type=int, default=[],
                     metavar="MM",
                     help="Month numbers (1-12) to drop after download. "
                          "Use for season filtering: e.g. `--exclude-months "
                          "11 12 1 2 3 4` drops winter and spring scenes "
                          "that decorrelate over snow.")
    ap.add_argument("--tarball", type=Path, default=None,
                     help="If set, tar+gzip <out>/ to this path when done.")
    ap.add_argument("--tarball-include", nargs="*", type=Path, default=[],
                     help="Extra paths (files or dirs) to fold into the "
                          "tarball alongside <out>/. Use this to ship "
                          "pre-computed dolphin outputs (unwrapped ifgs + "
                          "temporal_coherence_average) so students can run "
                          "the time-series inversion without first having "
                          "to run dolphin themselves.")
    return ap.parse_args()


def _excluded_to_season(exclude_months):
    """If exclude_months forms a contiguous winter band (wrapping year-end),
    return [start_doy, end_doy] for the kept summer band. Else None.

    Examples:
      [11,12,1,2,3,4,5] excluded -> [6,7,8,9,10] kept -> [152, 304]
      [12,1,2]          excluded -> [3,..,11]   kept -> [60, 334]
      [6,7]             excluded -> [1..5,8..12] kept -> not contiguous -> None
    """
    from datetime import date
    if not exclude_months:
        return None
    excluded = {int(m) for m in exclude_months}
    kept = [m for m in range(1, 13) if m not in excluded]
    # Kept months must form a contiguous run not wrapping the year
    # (otherwise it's the EXCLUDED set that wraps, which is the
    # common case for winter filtering and what we want).
    if kept != list(range(min(kept), max(kept) + 1)):
        return None
    start_doy = date(2001, min(kept), 1).timetuple().tm_yday
    # End-of-month for max(kept): use day 1 of next month minus 1 day,
    # or just last day of that month directly.
    import calendar
    last_day = calendar.monthrange(2001, max(kept))[1]
    end_doy = date(2001, max(kept), last_day).timetuple().tm_yday
    return [start_doy, end_doy]


def download(bursts, start, end, out_dir, exclude_months=None):
    """Pull the OPERA CSLCs from ASF DAAC into out_dir.

    If exclude_months forms a contiguous winter band, uses asf_search's
    built-in season= filter (server-side) to skip those scenes. The
    post-download filter_by_month() call is still belt-and-suspenders.
    """
    import asf_search as asf
    out_dir.mkdir(parents=True, exist_ok=True)

    # asf_search expects burst IDs with underscores (T115_245676_IW2),
    # but OPERA filenames use dashes (T115-245676-IW2). Accept either
    # from the user and convert.
    burst_ids = [b.replace("-", "_") for b in bursts]

    season = _excluded_to_season(exclude_months)
    if season:
        print(f"  using asf_search season=[{season[0]}, {season[1]}] "
              f"(DOY) to skip excluded months server-side")

    results = asf.search(
        processingLevel=asf.PRODUCT_TYPE.CSLC,
        operaBurstID=burst_ids,
        start=f"{start}T00:00:00Z",
        end=f"{end}T00:00:00Z",
        season=season,
    )
    print(f"ASF returned {len(results)} CSLCs to download")
    # ASFSession() auto-picks up ~/.netrc credentials via requests'
    # standard netrc handling — no explicit auth call needed.
    results.download(
        path=str(out_dir),
        session=asf.ASFSession(),
        processes=4,
    )


def crop_h5_to_tif(h5_path: Path, bbox: tuple[float, float, float, float],
                    tif_path: Path) -> None:
    """Read /data/VV from one OPERA CSLC H5, crop to UTM bbox, write geotiff."""
    xmin, ymin, xmax, ymax = bbox
    with h5py.File(h5_path, "r") as f:
        slc       = f["data/VV"]
        x_coords  = f["data/x_coordinates"][:]
        y_coords  = f["data/y_coordinates"][:]
        epsg      = int(f["data/projection"][()])

        # Slice indices for the bbox in native coords.
        ix = np.where((x_coords >= xmin) & (x_coords <= xmax))[0]
        iy = np.where((y_coords >= ymin) & (y_coords <= ymax))[0]
        assert ix.size and iy.size, f"empty crop for {h5_path}"
        arr = slc[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]

        dx = float(f["data/x_spacing"][()])
        dy = float(f["data/y_spacing"][()])
        x0 = float(x_coords[ix[0]])
        y0 = float(y_coords[iy[0]])

    # Geotiff: top-left origin, dy is negative in pixel-space.
    transform = from_origin(x0 - dx / 2, y0 - dy / 2, dx, -dy)
    crs = CRS.from_epsg(epsg)

    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        tif_path, "w",
        driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="complex64", crs=crs, transform=transform,
        compress="deflate", tiled=True,
    ) as dst:
        dst.write(arr.astype(np.complex64), 1)


def parse_acquisition(h5_name: str) -> tuple[str, str]:
    """OPERA filename -> (burst_id, YYYYMMDD)."""
    # OPERA_L2_CSLC-S1_T115-245676-IW2_20160722T141410Z_...
    parts = h5_name.split("_")
    burst = parts[3]
    yyyymmdd = parts[4][:8]
    return burst, yyyymmdd


def stitch_bursts_per_date(slc_tif_dir: Path, out_dir: Path) -> None:
    """Combine per-burst SLC tifs into one stitched tif per acquisition.

    Both bursts already share the same spatial grid from the crop step,
    so the merge is element-wise: take whichever burst has valid
    (non-zero) data at each pixel. Result is 105 stitched complex SLC
    tifs spanning the full two-burst AOI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Discover per-acquisition pairs.
    by_date: dict[str, list[Path]] = {}
    for p in sorted(slc_tif_dir.glob("*_*.slc.tif")):
        # filename is <BURST>_<YYYYMMDD>.slc.tif
        d = p.stem.split("_")[1][:8]
        by_date.setdefault(d, []).append(p)

    print(f"stitching {len(by_date)} acquisitions...")
    for d, ps in by_date.items():
        out_path = out_dir / f"{d}.slc.tif"
        if out_path.exists():
            continue
        # Load each burst's array and union them. Outside-footprint
        # pixels in the OPERA CSLCs come back as NaN, so the fill mask
        # has to look for NaN (not zero).
        combined = None
        ref_meta = None
        for p in ps:
            with rasterio.open(p) as s:
                arr = s.read(1)
                if combined is None:
                    combined = arr.copy()
                    ref_meta = s.meta.copy()
                else:
                    fill = ~np.isfinite(combined)
                    combined[fill] = arr[fill]
        with rasterio.open(out_path, "w", **ref_meta) as dst:
            dst.write(combined, 1)


def fetch_gnss(stations: list[str], out_dir: Path) -> None:
    """Pull daily ENU .tenv3 time series for each station from UNR-NGL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stn in stations:
        out = out_dir / f"{stn}.tenv3"
        if out.exists() and out.stat().st_size > 0:
            print(f"  gnss cached: {out.name}  ({out.stat().st_size/1e6:.1f} MB)")
            continue
        url = NGL_URL_FMT.format(stn=stn)
        print(f"  downloading {stn}.tenv3 from {url}")
        urllib.request.urlretrieve(url, out)
        print(f"    -> {out.stat().st_size/1e6:.1f} MB")


def build_tarball(src_dir: Path, dest: Path,
                   extras: list[Path] | None = None) -> None:
    """tar src_dir + extras into dest. No gzip - the .slc.tifs are
    already deflate-compressed inside, so gzipping them again costs
    ~10 min of CPU for ~25% size reduction. We also exclude the
    intermediate per-burst slc_tif/ dir since the notebook only
    needs slc_stitched/.

    src_dir goes in under its basename. Each path in extras is added
    under its basename too.
    """
    extras = extras or []
    exclude_dirs = {"slc_tif"}              # only stitched + gnss ship
    def _filter(info):
        rel = Path(info.name).relative_to(src_dir.name)
        # Drop the per-burst slc_tif/ intermediate dir.
        if rel.parts and rel.parts[0] in exclude_dirs:
            return None
        # Defensive: also skip any raw OPERA H5s sitting at the top of
        # data/. Current pipeline already deletes them post-crop; this
        # protects re-tars against stale ones from older runs.
        if rel.suffix == ".h5":
            return None
        return info

    print(f"building {dest.name} from {src_dir}/ "
          f"(excluding {sorted(exclude_dirs)} + *.h5) "
          f"+ {len(extras)} extra path(s) ...")
    with tarfile.open(dest, "w") as tar:
        tar.add(src_dir, arcname=src_dir.name, filter=_filter)
        for p in extras:
            tar.add(p, arcname=p.name)
    print(f"  tarball: {dest.stat().st_size/1e6:.0f} MB at {dest}")


def filter_by_month(out_dir: Path, exclude_months: list[int]) -> None:
    """Delete any cropped/staged outputs whose acquisition month is excluded.

    Run after the H5 → tif crop and after the per-burst stitch. The
    pre-download season filter already skips most of these; this is
    just belt-and-suspenders for any that snuck through.
    """
    if not exclude_months:
        return
    excluded = {int(m) for m in exclude_months}
    print(f"dropping acquisitions in months {sorted(excluded)} ...")
    removed = 0
    for tif in (out_dir / "slc_tif").glob("*_*.slc.tif"):
        if int(tif.stem.split("_")[1][4:6]) in excluded:
            tif.unlink()
            removed += 1
    for tif in (out_dir / "slc_stitched").glob("*.slc.tif"):
        if int(tif.stem[4:6]) in excluded:
            tif.unlink()
            removed += 1
    print(f"  removed {removed} files")


def main():
    args = parse_args()
    out_dir = args.out

    # 1) Download (skipped per-file by asf_search if already on disk).
    download(args.bursts, args.start, args.end, out_dir,
              exclude_months=args.exclude_months)

    # 2) Crop each H5 to a per-burst SLC geotiff under slc_tif/, then
    #    delete the raw H5 so the staged directory stays compact (we
    #    don't keep the metadata layers around).
    h5s = sorted(out_dir.glob("OPERA_L2_CSLC-S1_*.h5"))
    print(f"cropping {len(h5s)} H5s to bbox {args.bbox}")
    for h5 in h5s:
        burst, yyyymmdd = parse_acquisition(h5.name)
        tif = out_dir / "slc_tif" / f"{burst}_{yyyymmdd}.slc.tif"
        if not tif.exists():
            crop_h5_to_tif(h5, tuple(args.bbox), tif)
            print(f"  {tif.relative_to(out_dir)}")
        h5.unlink()

    # 3) Stitch the per-burst SLCs into one tif per acquisition.
    stitch_bursts_per_date(out_dir / "slc_tif", out_dir / "slc_stitched")

    # 4) Optional season/month filter (e.g. drop winter scenes).
    filter_by_month(out_dir, args.exclude_months)

    # 5) Optional GNSS time series from UNR-NGL.
    if args.gnss_stations:
        fetch_gnss(args.gnss_stations, out_dir / "gnss")

    # 6) Optional tarball (with any --tarball-include extras folded in).
    if args.tarball is not None:
        build_tarball(out_dir, args.tarball, args.tarball_include)

    print("done.")


if __name__ == "__main__":
    main()
