"""
Utility functions for InSAR processing workflow.

Provides data I/O, array stitching, cross-multiplication, multilooking,
coherence estimation, phase filtering, unwrapping, and visualisation
for OPERA CSLC-based Sentinel-1 TOPS InSAR processing.

Authors: Zhenli Tang, Zhang Yunjun
Based on topsApp.py processing concepts.
"""

import os
import glob
from pathlib import Path

import numpy as np
import h5py
import isce3
from osgeo import gdal, osr
import re
from lxml import etree
from pyproj import Transformer
from skimage.transform import resize
import matplotlib.pyplot as plt
import yaml

import sys as _sys
from s1reader.s1_burst_id import S1BurstId

for _var, _sub in [('PROJ_DATA', 'share/proj'), ('GDAL_DATA', 'share/gdal')]:
    if _var not in os.environ:
        _path = os.path.join(_sys.prefix, _sub)
        if os.path.isdir(_path):
            os.environ[_var] = _path


# ---------------------------------------------------------------------------
# Sentinel-1 IW constants
# ---------------------------------------------------------------------------
S1_WAVELENGTH = 0.055465763
S1_RANGE_PX_SPACING = 2.33
DEFAULT_RANGE_LOOKS = 4
DEFAULT_AZIMUTH_LOOKS = 2

# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------
import psutil
MEMORY_LIMIT = 10 * 1024**3

def get_memory_usage():
    """Return current RSS memory usage in bytes."""
    return psutil.Process().memory_info().rss

def print_memory_usage(label):
    """Print current memory usage vs 10 GB limit with a label."""
    mem = get_memory_usage()
    pct = mem / MEMORY_LIMIT * 100
    print(f'  [{label}] memory: {mem/1024**3:.1f} GB / 10 GB ({pct:.1f}%)')

def clear_large_arrays():
    """Delete known large arrays from caller's global scope + garbage collect."""
    import gc
    _KNOWN = [
        # per-burst accumulators
        'ifg_list', 'coh_list', 'coh_cplx',
        # stitched arrays
        'ifg_stitched', 'coh_stitched', 'ifg_ml', 'coh_ml', 'ifg_filt',
        # phsig / unwrap
        'phsig', 'unw', 'conncomp',
        # section 4 inspection
        'rdr_slc', 'rdr_amp', 'rdr_amp_db', 'geo_slc', 'geo_amp', 'geo_amp_db',
        # per-burst temporaries (in case loop was interrupted)
        'ref_arr', 'sec_arr', 'ref_pow', 'sec_pow',
        'ifg_sum', 'ref_sum', 'sec_sum', 'ifg_burst', 'coh_burst',
        # LOS
        'los_2band',
    ]
    frame = _sys._getframe(1)
    deleted = [n for n in _KNOWN if n in frame.f_globals]
    for n in deleted:
        del frame.f_globals[n]
    gc.collect()
    if deleted:
        print(f'Cleared {len(deleted)} array(s): {", ".join(deleted[:6])}'
              f'{" ..." if len(deleted) > 6 else ""}')
    print_memory_usage('mem cleared')

# ===================================================================

def load_orbit_from_h5(h5_path):
    """Load an ISCE3 Orbit from an OPERA CSLC HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file containing ``/metadata/orbit/``.

    Returns
    -------
    orbit : isce3.core.Orbit
    t0 : float
        Time offset (seconds) of the first state vector relative to the
        H5 orbit reference epoch.  Use to convert ``azt`` values to
        ISCE3 orbit-relative seconds.
    """
    with h5py.File(h5_path, 'r') as f:
        orb_grp = f['/metadata/orbit']
        pos_x = orb_grp['position_x'][:]
        pos_y = orb_grp['position_y'][:]
        pos_z = orb_grp['position_z'][:]
        vel_x = orb_grp['velocity_x'][:]
        vel_y = orb_grp['velocity_y'][:]
        vel_z = orb_grp['velocity_z'][:]
        times = orb_grp['time'][:]
        ref_epoch_str = orb_grp['reference_epoch'][()].decode()

    ref_epoch = isce3.core.DateTime(ref_epoch_str)
    statevecs = []
    for i in range(len(times)):
        pos = np.array([pos_x[i], pos_y[i], pos_z[i]])
        vel = np.array([vel_x[i], vel_y[i], vel_z[i]])
        sv = isce3.core.StateVector(
            ref_epoch + isce3.core.TimeDelta(times[i]), pos, vel)
        statevecs.append(sv)

    return isce3.core.Orbit(statevecs), times[0]


def compute_isce3_incidence_angle(h5_path):
    """Compute per-pixel incidence angle on the radar LUT grid using ISCE3.

    Uses ``isce3.geometry.look_inc_ang_from_slant_range()`` with the
    WGS-84 ellipsoid (h=0) — consistent with the physical model employed
    by COMPASS's ``Rdr2Geo``, but without terrain-height adjustment.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file.

    Returns
    -------
    inc_deg : np.ndarray (float64)
        2-D incidence angle in degrees, shape ``(n_az, n_rg)`` matching
        the ``timing_corrections`` LUT grid.
    """
    corr_grp = '/metadata/processing_information/timing_corrections'

    with h5py.File(h5_path, 'r') as f:
        sr = f[f'{corr_grp}/slant_range'][:]
        azt = f[f'{corr_grp}/zero_doppler_time'][:]

    orbit, t0 = load_orbit_from_h5(h5_path)

    # Convert azt to ISCE3 orbit-relative seconds
    azt_orbit = azt - t0
    dem_interp = isce3.geometry.DEMInterpolator(0.0)

    n_az, n_rg = len(azt), len(sr)
    inc_deg = np.zeros((n_az, n_rg), dtype=np.float64)
    for i in range(n_az):
        _, inc_row = isce3.geometry.look_inc_ang_from_slant_range(
            sr, orbit, az_time=azt_orbit[i], dem_interp=dem_interp)
        inc_deg[i, :] = np.rad2deg(inc_row)

    return inc_deg

# 0. SAFE burst SLC extraction
# ===================================================================

def extract_burst_slc(safe_path, burst_id):
    """Extract a single burst's SLC from a Sentinel-1 SAFE measurement TIFF.

    Sentinel-1 SAFE products store multiple bursts concatenated along azimuth
    in a single TIFF.  This function parses the subswath annotation XML to
    find *burst_id*'s line range and reads only that burst.

    Parameters
    ----------
    safe_path : str or Path
        Path to the SAFE directory.
    burst_id : str
        Burst identifier, e.g. ``t124_264305_iw2`` or just ``264305``.
        If the full ``tRRR_BBBBBB_iwN`` form is given the numeric burst
        index is extracted automatically.

    Returns
    -------
    slc : np.ndarray (complex64)
        2-D complex SLC array ``[azimuth_lines, range_samples]``.
    """
    safe_path = Path(safe_path)

    # Find the measurement TIFF
    tiff_files = sorted(glob.glob(str(safe_path / 'measurement' / '*.tiff')))
    if not tiff_files:
        raise FileNotFoundError(
            f'No SLC TIFF found in {safe_path}/measurement/')

    # Parse numeric burst index from burst_id (e.g. 264305)
    burst_idx = int(str(burst_id).split('_')[1]) if '_' in str(burst_id) else int(burst_id)

    # Find the matching IW annotation file
    iw_num = str(burst_id).split('_')[-1][-1]  # 'iw2' → '2'
    ann_pattern = str(safe_path / 'annotation' / f'*-iw{iw_num}*slc*vv*.xml')
    
    # If burst_id does not contain iw info, try to infer from filenames
    candidates = sorted(glob.glob(ann_pattern))
    if not candidates:
        # Try without iw filter
        candidates = sorted(glob.glob(
            str(safe_path / 'annotation' / '*-slc-vv-*.xml')))
    
    ann_file = candidates[0]
    tree = etree.parse(ann_file)
    root = tree.getroot()

    _REL_ORBIT_OFFSET = {'S1A': 73, 'S1B': 27}
    mission_id = root.find('.//{*}missionId').text
    abs_orbit = int(root.find('.//{*}absoluteOrbitNumber').text)
    offset = _REL_ORBIT_OFFSET.get(mission_id, 73)
    rel_orbit = (abs_orbit - offset) % 175 + 1

    iw_name = root.find('.//{*}swath').text
    iw_num = iw_name[-1]

    lines_per_burst = int(root.find('.//{*}linesPerBurst').text)
    burst_list = root.find('.//{*}burstList')
    burst_id_str = str(burst_id)

    burst_index_in_list = None
    for bi, b_elem in enumerate(burst_list):
        b_id_elem = b_elem.find('{*}burstId')
        if b_id_elem is not None:
            if b_id_elem.text == str(burst_idx):
                burst_index_in_list = bi
                break
        else:
            azt_time = b_elem.find('.{*}azimuthTime')
            azt_anx = b_elem.find('.{*}azimuthAnxTime')
            if azt_time is None or azt_anx is None:
                continue
            computed = _compute_burst_id(
                azt_time.text,
                float(azt_anx.text),
                rel_orbit,
                iw_name.upper(),
            )
            if computed == burst_id_str:
                burst_index_in_list = bi
                break

    if burst_index_in_list is None:
        raise ValueError(
            f'Burst {burst_id} not found in annotation {ann_file}')

    line_start = burst_index_in_list * lines_per_burst

    ds = gdal.Open(tiff_files[0])
    slc = ds.GetRasterBand(1).ReadAsArray(0, line_start,
                                          ds.RasterXSize, lines_per_burst)
    ds = None

    return slc.astype(np.complex64)

# ===================================================================
# 7. Static troposphere delay (identical to COMPASS lut.py)
# ===================================================================

def compute_static_troposphere_delay(incidence_angle_arr, hgt_arr):
    """Compute troposphere delay using static model.

    Identical to ``compass.utils.lut::compute_static_troposphere_delay()``.

    Parameters
    ----------
    incidence_angle_arr : np.ndarray
        Incidence angle raster in degrees, on the radar grid.
    hgt_arr : np.ndarray
        Surface height raster in metres, on the radar grid (same shape).

    Returns
    -------
    tropo : np.ndarray
        Troposphere delay in slant range (m), same shape as inputs.
    """
    ZPD = 2.3
    H = 6000.0
    tropo = ZPD / np.cos(np.deg2rad(incidence_angle_arr)) * np.exp(-1 * hgt_arr / H)
    return tropo

# ===================================================================
# 1. SLC data I/O (HDF5 / OPERA CSLC)
# ===================================================================

def read_cslc_array(h5_path):
    """Read an OPERA CSLC HDF5 file into a numpy complex64 array.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA-format HDF5 CSLC file.

    Returns
    -------
    data_vv : np.ndarray (complex64)
        2-D complex SLC array ``[rows, cols]``.
    geo_transform : tuple
        GDAL-style geotransform ``(x0, dx, 0, y0, 0, dy)``.
    epsg : int
        EPSG code of the UTM projection.
    proj_wkt : str
        Projection definition in WKT format.
    """
    with h5py.File(h5_path, 'r') as f:
        data_vv = f['/data/VV'][:]
        x = f['/data/x_coordinates'][:]
        y = f['/data/y_coordinates'][:]
        epsg = int(f['/data/projection'][()])

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    geo_transform = (x[0], dx, 0, y[0], 0, dy)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    proj_wkt = srs.ExportToWkt()

    return data_vv, geo_transform, epsg, proj_wkt



# ===================================================================
def align_cslc_pair(ref_arr, ref_gt, sec_arr, sec_gt):
    """Align sec array to ref grid by intersecting their geographic extents.

    OPERA CSLC arrays share the same posting (``dx``, ``dy``) and only
    differ in their starting offsets (``x0``, ``y0``).  This function
    computes the integer pixel shift between the two grids and slices
    both arrays to their common overlap, guaranteeing pixel-to-pixel
    geographic correspondence.

    Parameters
    ----------
    ref_arr, sec_arr : np.ndarray (complex64)
        Full-size reference and secondary CSLC arrays.
    ref_gt, sec_gt : tuple
        GDAL geotransforms ``(x0, dx, 0, y0, 0, dy)``.

    Returns
    -------
    ref_aligned, sec_aligned : np.ndarray (complex64)
        Arrays cropped to the common overlapping extent.
    common_gt : tuple
        Geotransform of the overlapping region.
    """
    x0_r, dx, _, y0_r, _, dy = ref_gt
    x0_s, _, _, y0_s, _, _ = sec_gt

    nr_r, nc_r = ref_arr.shape
    nr_s, nc_s = sec_arr.shape

    # Integer pixel offset from ref grid to sec grid
    off_c = int(round((x0_s - x0_r) / dx))
    off_r = int(round((y0_s - y0_r) / dy))

    # Compute overlap region
    if off_c >= 0:
        rc0, sc0 = off_c, 0
        nc = min(nc_r - off_c, nc_s)
    else:
        rc0, sc0 = 0, -off_c
        nc = min(nc_r, nc_s + off_c)

    if off_r >= 0:
        rr0, sr0 = off_r, 0
        nr = min(nr_r - off_r, nr_s)
    else:
        rr0, sr0 = 0, -off_r
        nr = min(nr_r, nr_s + off_r)

    ref_aligned = ref_arr[rr0:rr0 + nr, rc0:rc0 + nc]
    sec_aligned = sec_arr[sr0:sr0 + nr, sc0:sc0 + nc]

    common_gt = (x0_r + rc0 * dx, dx, 0, y0_r + rr0 * dy, 0, dy)
    return ref_aligned, sec_aligned, common_gt



def get_cslc_extent(h5_path):
    """Read coordinate vectors from an OPERA CSLC HDF5 file (cheap metadata).

    Parameters
    ----------
    h5_path : str or Path

    Returns
    -------
    x0, dx, y0, dy, nrows, ncols, epsg, proj_wkt
    """
    with h5py.File(h5_path, 'r') as f:
        x = f['/data/x_coordinates'][:]
        y = f['/data/y_coordinates'][:]
        epsg = int(f['/data/projection'][()])
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    proj_wkt = srs.ExportToWkt()
    return x[0], dx, y[0], dy, len(y), len(x), epsg, proj_wkt


def compute_union_grid(extents, bbox_wsen, epsg_utm):
    if not extents:
        raise ValueError('extents list is empty')

    # extents: (x0, dx, y0, dy, nrows, ncols, epsg, proj_wkt)
    _, dx, _, dy, _, _, _, proj_wkt = extents[0]

    # Union extent in UTM
    ulx = min(e[0] for e in extents)                        # west
    lrx = max(e[0] + e[5] * e[1] for e in extents)          # east  (e[5]=ncols)
    uly = max(e[2] for e in extents)                        # north
    lry = min(e[2] + e[3] * e[4] for e in extents)          # south (e[3]=dy<0, e[4]=nrows)

    # Clip to geographic bbox in UTM
    tf = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_utm}', always_xy=True)
    xs, ys = tf.transform(
        [bbox_wsen[0], bbox_wsen[2], bbox_wsen[2], bbox_wsen[0]],
        [bbox_wsen[1], bbox_wsen[1], bbox_wsen[3], bbox_wsen[3]],
    )
    bbox_xmin, bbox_ymin = min(xs), min(ys)
    bbox_xmax, bbox_ymax = max(xs), max(ys)

    ulx = max(ulx, bbox_xmin)
    lrx = min(lrx, bbox_xmax)
    uly = min(uly, bbox_ymax)       # uly is north, bbox_ymax is north
    lry = max(lry, bbox_ymin)       # lry is south, bbox_ymin is south

    out_cols = int((lrx - ulx) / abs(dx) + 0.5)             # lrx > ulx → positive
    out_rows = int(abs(uly - lry) / abs(dy) + 0.5)          # abs(uly-lry) > 0
    out_gt = (ulx, dx, 0, uly, 0, dy)

    return out_gt, out_rows, out_cols, proj_wkt


def blit_into_stitched(dst, dst_gt, src, src_gt, nodata_thresh=1e-6):
    """Copy valid source pixels into a pre-allocated destination array.

    Computes the integer pixel offset of *src* within *dst* using their
    geotransforms, then copies pixels where ``|src| > nodata_thresh``.
    Overlap regions are overwritten (last-wins).

    Parameters
    ----------
    dst : np.ndarray  Pre-allocated destination (stitched) array.
    dst_gt : tuple (x0, dx, 0, y0, 0, dy)
    src : np.ndarray  Source (per-burst) array.
    src_gt : tuple (x0, dx, 0, y0, 0, dy)
    nodata_thresh : float  Pixels with |src| <= nodata_thresh are skipped.
    """
    xoff = int(round((src_gt[0] - dst_gt[0]) / dst_gt[1]))
    yoff = int(round((src_gt[3] - dst_gt[3]) / dst_gt[5]))

    sh, sw = src.shape
    dh, dw = dst.shape

    src_r0 = max(0, -yoff)
    src_r1 = min(sh, dh - yoff)
    src_c0 = max(0, -xoff)
    src_c1 = min(sw, dw - xoff)
    dst_r0 = max(0, yoff)
    dst_r1 = min(dh, yoff + sh)
    dst_c0 = max(0, xoff)
    dst_c1 = min(dw, xoff + sw)

    h = min(src_r1 - src_r0, dst_r1 - dst_r0)
    w = min(src_c1 - src_c0, dst_c1 - dst_c0)
    if h <= 0 or w <= 0:
        return

    src_chunk = src[src_r0:src_r0 + h, src_c0:src_c0 + w]
    valid = np.isfinite(src_chunk) & (np.abs(src_chunk) > nodata_thresh)
    dst[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w][valid] = src_chunk[valid]





# 2. SLC / interferogram stitching
# ===================================================================

def stitch_arrays(arrays_list, bbox_wsen, dx=5.0, dy=-10.0, epsg_utm=32605,
                  method='last'):
    """Stitch geocoded arrays via gdal_merge-style pixel-offset copy.

    Pure numpy implementation — no intermediate files, no rasterio merge.
    Matches ``gdal_merge.py`` algorithm exactly:
    ``int((src_x0 - out_x0) / dx)`` truncation for pixel placement,
    last-source-wins in overlaps.

    Parameters
    ----------
    arrays_list : list of (arr, geotransform, proj_wkt) tuples
    bbox_wsen : tuple  ``(west, south, east, north)`` EPSG:4326.
    dx, dy : float  pixel sizes (metres).
    epsg_utm : int  UTM EPSG code.
    method : {'first', 'last'}
        ``'last'`` — later sources overwrite earlier (default).
        ``'first'`` — earlier sources take precedence.

    Returns
    -------
    stitched : np.ndarray  ``[rows, cols]``
    out_gt : tuple  GDAL geotransform
    proj_wkt : str
    """
    from pyproj import Transformer

    if not arrays_list:
        raise ValueError("arrays_list is empty")

    sample, _, proj_wkt = arrays_list[0]
    is_complex = np.issubdtype(sample.dtype, np.complexfloating)

    # --- gdal_merge-style output extent: union of all source bounds ---
    ulx = None; uly = None; lrx = None; lry = None
    pieces = []
    for arr, gt, _ in arrays_list:
        x0, px_dx, _, y0, _, py_dy = gt
        x1 = x0 + arr.shape[1] * px_dx
        y1 = y0 + arr.shape[0] * py_dy
        if ulx is None or x0 < ulx: ulx = x0
        if lrx is None or x1 > lrx: lrx = x1
        if uly is None or y0 > uly: uly = y0
        if lry is None or y1 < lry: lry = y1
        pieces.append({'arr': arr, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    # Clip to bbox_wsen
    tf = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_utm}', always_xy=True)
    xs, ys = tf.transform(
        [bbox_wsen[0], bbox_wsen[2], bbox_wsen[2], bbox_wsen[0]],
        [bbox_wsen[1], bbox_wsen[1], bbox_wsen[3], bbox_wsen[3]],
    )
    bbox_xmin, bbox_ymin = min(xs), min(ys)
    bbox_xmax, bbox_ymax = max(xs), max(ys)

    ulx = max(ulx, bbox_xmin)
    lrx = min(lrx, bbox_xmax)
    uly = min(uly, bbox_ymax)
    lry = max(lry, bbox_ymin)

    # Output grid (gdal_merge.py style: int((extent / ps) + 0.5))
    out_cols = int((lrx - ulx) / dx + 0.5)
    out_rows = int((lry - uly) / dy + 0.5)
    out_gt = (ulx, dx, 0, uly, 0, dy)

    stitched = np.zeros((out_rows, out_cols), dtype=sample.dtype)

    items = pieces
    if method == 'first':
        items = list(reversed(items))

    for p in items:
        arr = p['arr']
        src_x0, src_y0 = p['x0'], p['y0']
        src_x1, src_y1 = p['x1'], p['y1']

        # gdal_merge.py pixel offset: int((src_ul - out_ul) / pixel_size)
        xoff = int((src_x0 - ulx) / dx)
        yoff = int((src_y0 - uly) / dy)

        # Clip source region to output bounds
        src_r0 = max(0, -yoff)
        src_r1 = min(arr.shape[0], out_rows - yoff)
        src_c0 = max(0, -xoff)
        src_c1 = min(arr.shape[1], out_cols - xoff)
        dst_r0 = max(0, yoff)
        dst_r1 = min(out_rows, yoff + arr.shape[0])
        dst_c0 = max(0, xoff)
        dst_c1 = min(out_cols, xoff + arr.shape[1])

        h = min(src_r1 - src_r0, dst_r1 - dst_r0)
        w = min(src_c1 - src_c0, dst_c1 - dst_c0)
        if h <= 0 or w <= 0:
            continue

        src = arr[src_r0:src_r0 + h, src_c0:src_c0 + w]
        if is_complex:
            valid = np.isfinite(src) & (np.abs(src) > 1e-6)
        else:
            valid = np.isfinite(src) & (np.abs(src) > 1e-6)

        if method == 'last':
            stitched[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w][valid] = src[valid]
        else:
            dst = stitched[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w]
            empty = ~np.isfinite(dst) | (np.abs(dst) < 1e-6) if is_complex else (dst == 0)
            write = valid & empty
            stitched[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w][write] = src[write]

    return stitched, out_gt, proj_wkt

def multilook_ifg(arr, az_looks, rg_looks):
    """Multilook a complex or real array by non-overlapping block averaging.

    Parameters
    ----------
    arr : np.ndarray
        Input array ``[rows, cols]``.
    az_looks : int
        Number of looks in the azimuth (row) direction.
    rg_looks : int
        Number of looks in the range (column) direction.

    Returns
    -------
    ml : np.ndarray
        Multilooked array ``[rows//az_looks, cols//rg_looks]``.
    """
    nr, nc = arr.shape
    nr = nr - nr % az_looks
    nc = nc - nc % rg_looks
    return arr[:nr, :nc].reshape(
        nr // az_looks, az_looks, nc // rg_looks, rg_looks).mean(axis=(1, 3))


# ===================================================================
# 4. Goldstein adaptive phase filter
# ===================================================================

def goldstein_filter(complex_arr, alpha=0.5, psize=32, nodata_mask=None):
    """Goldstein adaptive phase filter with overlapping patches.

    Parameters
    ----------
    complex_arr : np.ndarray (complex64)
        Input complex interferogram ``[rows, cols]``.
    alpha : float
        Filter exponent in [0, 1].
    psize : int
        FFT patch size (power of 2 recommended).
    nodata_mask : np.ndarray (bool), optional
        Boolean mask where data is invalid.

    Returns
    -------
    filtered : np.ndarray (complex64)
        Filtered complex array, same shape as input.
    """
    orig_rows, orig_cols = complex_arr.shape
    pad = psize // 2
    step = pad
    half = pad

    wx = (1.0 - np.abs(np.arange(half) - (psize / 2.0 - 1.0))
          / (psize / 2.0 - 1.0))
    wy = (1.0 - np.abs(np.arange(half) - (psize / 2.0 - 1.0))
          / (psize / 2.0 - 1.0))
    q = np.outer(wy, wx)
    wf = np.block([[q, np.flip(q, 1)],
                   [np.flip(q, 0), np.flip(np.flip(q, 0), 1)]])

    if nodata_mask is None:
        orig_nodata = np.zeros((orig_rows, orig_cols), dtype=bool)
    else:
        orig_nodata = nodata_mask.copy()

    padded = np.pad(complex_arr, ((pad, pad), (pad, pad)), mode='constant')
    p_rows, p_cols = padded.shape

    nodata = np.pad(orig_nodata, ((pad, pad), (pad, pad)),
                    mode='constant', constant_values=True)

    filtered = np.zeros((p_rows, p_cols), dtype=np.complex64)
    norm = np.zeros((p_rows, p_cols), dtype=np.float32)

    for i in range(0, p_rows - psize + 1, step):
        for j in range(0, p_cols - psize + 1, step):
            ri, rj = slice(i, i + psize), slice(j, j + psize)
            patch = padded[ri, rj].copy()

            if np.all(nodata[ri, rj]):
                continue

            patch[nodata[ri, rj]] = 0
            S = np.fft.fft2(patch, s=(psize, psize))
            H = np.power(np.abs(S), alpha)
            S = H * S
            pf = np.fft.ifft2(S, s=(psize, psize))

            w = wf[:patch.shape[0], :patch.shape[1]]
            filtered[ri, rj] += pf * w
            norm[ri, rj] += w

    valid = norm > 0
    filtered[valid] /= norm[valid]
    filtered = filtered[pad:pad + orig_rows, pad:pad + orig_cols]
    filtered[orig_nodata] = 0 + 0j

    return filtered


# ===================================================================
# 5. Phase-sigma coherence estimation
# ===================================================================

def _gaussian_kernel(size):
    """Generate a normalized 2-D Gaussian weighting kernel.

    Matches ISCE2 Fortran ph_slope.F / ph_sigma.F: sigma^2 = size/2.0.
    """
    half = size // 2
    s1 = 0.0
    kernel = np.zeros((size, size), dtype=np.float64)
    for k in range(size):
        for j in range(size):
            w1 = (k - half) ** 2 + (j - half) ** 2
            kernel[k, j] = np.exp(-w1 / (size / 2.0))
            s1 += kernel[k, j]
    return (kernel / s1).astype(np.float32)


def estimate_phsig_correlation(ifg_arr, ps_win=5, grad_win=5, nlks=3.0):
    """Estimate phase-sigma correlation from a complex interferogram.

    Matches ISCE2 Fortran ``ph_slope.F`` + ``ph_sigma.F`` algorithm:
    Gaussian-weighted phase gradient estimation, local window
    deramping with unweighted circular-mean phase reference, weighted
    phase variance, and NLKS-based correlation conversion.

    Parameters
    ----------
    ifg_arr : np.ndarray (complex64)
        Complex interferogram ``[rows, cols]``.
    ps_win : int
        Phase-sigma estimation window size (odd).
    grad_win : int
        Gradient estimation window size (odd).
    nlks : float
        Number of looks parameter. ISCE2 default is 3.0.

    Returns
    -------
    coh_phsig : np.ndarray (float32)
        Phase-sigma correlation array, clipped to [0, 1].
    """
    from scipy.ndimage import correlate

    rows, cols = ifg_arr.shape

    if ps_win % 2 == 0:
        ps_win += 1
    if grad_win % 2 == 0:
        grad_win += 1
    ps_half = ps_win // 2
    grad_half = grad_win // 2

    padded = np.pad(ifg_arr,
                    ((grad_half, grad_half), (grad_half, grad_half)),
                    mode='constant')

    rg_diff = (
        padded[grad_half:grad_half + rows,
               grad_half:grad_half + cols] *
        np.conj(padded[grad_half:grad_half + rows,
                       grad_half - 1:grad_half + cols - 1])
    )
    az_diff = (
        padded[grad_half:grad_half + rows,
               grad_half:grad_half + cols] *
        np.conj(padded[grad_half - 1:grad_half + rows - 1,
                       grad_half:grad_half + cols])
    )

    gk = _gaussian_kernel(grad_win)
    rg_smooth = correlate(rg_diff, gk)
    az_smooth = correlate(az_diff, gk)

    rg_slope = np.arctan2(rg_smooth.imag, rg_smooth.real)
    az_slope = np.arctan2(az_smooth.imag, az_smooth.real)
    rg_slope[np.abs(rg_smooth) == 0] = 0.0
    az_slope[np.abs(az_smooth) == 0] = 0.0

    # Match Fortran ph_slope.F valid range: [half+1, size-half-1]
    # Fortran computes slopes for i from half+1 to nline-half-1
    # (inclusive). Rows [0..half] and [nline-half..nline-1] are zero.
    # Same for columns.
    if grad_half > 0:
        rg_slope[:grad_half + 1, :] = 0.0
        rg_slope[-(grad_half):, :] = 0.0
        rg_slope[:, :grad_half + 1] = 0.0
        rg_slope[:, -(grad_half):] = 0.0
        az_slope[:grad_half + 1, :] = 0.0
        az_slope[-(grad_half):, :] = 0.0
        az_slope[:, :grad_half + 1] = 0.0
        az_slope[:, -(grad_half):] = 0.0

    offsets = np.arange(-ps_half, ps_half + 1)
    di_mesh, dj_mesh = np.meshgrid(offsets, offsets, indexing='ij')
    ps_weights = _gaussian_kernel(ps_win)

    coh = np.zeros((rows, cols), dtype=np.float32)

    i_idx = np.arange(ps_half, rows - ps_half)
    j_idx = np.arange(ps_half, cols - ps_half)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    i_flat = I.ravel()
    j_flat = J.ravel()

    n_total = len(i_flat)
    batch_size = 500

    for b_start in range(0, n_total, batch_size):
        b_end = min(b_start + batch_size, n_total)
        bi = i_flat[b_start:b_end]
        bj = j_flat[b_start:b_end]

        row_idx = bi[:, None, None] + di_mesh[None, :, :]
        col_idx = bj[:, None, None] + dj_mesh[None, :, :]
        windows = ifg_arr[row_idx, col_idx]

        rg_s = rg_slope[bi, bj]
        az_s = az_slope[bi, bj]
        ramp = (di_mesh[None, :, :] * az_s[:, None, None] +
                dj_mesh[None, :, :] * rg_s[:, None, None])

        exp_ramp = np.cos(ramp) - 1j * np.sin(ramp)
        comp = windows * exp_ramp

        wsum = np.sum(comp, axis=(1, 2))
        mag = np.abs(wsum)

        valid = mag > 1e-10
        if not np.any(valid):
            continue
        vidx = np.flatnonzero(valid)

        norm_sum = wsum[vidx] / mag[vidx]
        deramped = comp[vidx] * np.conj(norm_sum[:, None, None])

        phases = np.arctan2(deramped.imag, deramped.real)
        wt = ps_weights[None, :, :]
        mean_ph = np.sum(wt * phases, axis=(1, 2))
        mean_ph2 = np.sum(wt * phases * phases, axis=(1, 2))
        var = mean_ph2 - mean_ph * mean_ph

        var_pos = var > 0
        if np.any(var_pos):
            gidx = vidx[var_pos]
            coh[bi[gidx], bj[gidx]] = (
                1.0 / np.sqrt(2.0 * nlks * var[var_pos] + 1.0)
            )
        if np.any(~var_pos):
            gidx = vidx[~var_pos]
            coh[bi[gidx], bj[gidx]] = 1.0

    return np.clip(coh, 0.0, 1.0)


# ===================================================================
# 6. Save Data
# ===================================================================

def save_tiff(out_path, data, gt, proj_wkt, dtype=None):
    """Save a numpy array as a GeoTIFF (single or multi-band).

    2-D ``[rows, cols]`` → single-band.
    3-D ``[bands, rows, cols]`` → multi-band.

    Parameters
    ----------
    out_path : str or Path
    data : np.ndarray  2-D or 3-D.
    gt : tuple  GDAL geotransform.
    proj_wkt : str  Projection WKT.
    dtype : int, optional  GDAL type. Auto-detected when None.
    """
    drv = gdal.GetDriverByName('GTiff')

    if data.ndim == 2:
        bands, rows, cols = 1, *data.shape
    elif data.ndim == 3:
        bands, rows, cols = data.shape[0], data.shape[1], data.shape[2]
    else:
        raise ValueError(f'Expected 2-D or 3-D array, got {data.ndim}-D')

    if dtype is None:
        dtype_map = {
            np.float32: gdal.GDT_Float32, np.float64: gdal.GDT_Float64,
            np.int32: gdal.GDT_Int32, np.int16: gdal.GDT_Int16,
            np.uint8: gdal.GDT_Byte, np.uint16: gdal.GDT_UInt16,
            np.complex64: gdal.GDT_CFloat32,
        }
        dtype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)

    ds = drv.Create(str(out_path), cols, rows, bands, dtype)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj_wkt)

    if data.ndim == 2:
        ds.GetRasterBand(1).WriteArray(data)
    else:
        for b in range(bands):
            ds.GetRasterBand(b + 1).WriteArray(data[b])

    ds = None




# ===================================================================
# 7. COMPASS geo run-configuration writer
# ===================================================================

def write_geo_runconfig(out_path, safe_file, orbit_file, burst_id,
                        dem_file, burst_database_file, tec_file=None,
                        x_posting=5, y_posting=10):
    """Write a complete COMPASS geocoded-CSLC run-configuration YAML.

    The template mirrors the COMPASS defaults (``s1_cslc_geo.yaml``) and
    conforms to the validation schema (``s1_cslc_geo_schemas.yaml``): every
    group is written out explicitly so the full initial state is reproducible.

    Parameters
    ----------
    out_path : str or Path
        Output YAML config file path.
    safe_file : str
        Path to the SAFE directory (or zip).
    orbit_file : str
        Path to the orbit (EOF) file.
    burst_id : str
        Burst identifier, e.g. ``t124_264305_iw2``.
    dem_file : str
        Path to the DEM GeoTIFF.
    burst_database_file : str
        Path to the burst-db SQLite3 file.
    tec_file : str or None, optional
        Path to the IONEX TEC file. Omitted from the YAML when None.
    x_posting, y_posting : float, optional
        Geocoding grid spacing (metres) along X and Y.
    """
    dynamic_ancillary = {
        'dem_file': str(dem_file),
        'dem_description': 'DEM description was not provided.',
    }
    if tec_file:
        dynamic_ancillary['tec_file'] = str(tec_file)

    cfg = {
        'runconfig': {
            'name': 'cslc_s1_workflow_default',
            'groups': {
                'pge_name_group': {'pge_name': 'CSLC_S1_PGE'},
                'input_file_group': {
                    'safe_file_path': [str(safe_file)],
                    'orbit_file_path': [str(orbit_file) if orbit_file else ''],
                    'burst_id': [burst_id],
                },
                'dynamic_ancillary_file_group': dynamic_ancillary,
                'static_ancillary_file_group': {
                    'burst_database_file': str(burst_database_file),
                },
                'product_path_group': {
                    'product_path': '.',
                    'scratch_path': './scratch',
                    'sas_output_file': '',
                    'product_version': '0.2',
                    'product_specification_version': '0.1',
                },
                'primary_executable': {'product_type': 'CSLC_S1'},
                'processing': {
                    'polarization': 'co-pol',
                    'geocoding': {
                        'flatten': True,
                        'x_posting': x_posting,
                        'y_posting': y_posting,
                    },
                    'geo2rdr': {
                        'lines_per_block': 1000,
                        'threshold': 1.0e-8,
                        'numiter': 25,
                    },
                    'correction_luts': {
                        'enabled': True,
                        'range_spacing': 120,
                        'azimuth_spacing': 0.028,
                        'troposphere': {'delay_type': 'wet_dry'},
                    },
                    'rdr2geo': {
                        'threshold': 1.0e-8,
                        'numiter': 25,
                        'lines_per_block': 1000,
                        'extraiter': 10,
                        'compute_latitude': True,
                        'compute_longitude': True,
                        'compute_height': True,
                        'compute_layover_shadow_mask': True,
                        'compute_local_incidence_angle': True,
                        'compute_ground_to_sat_east': True,
                        'compute_ground_to_sat_north': True,
                    },
                },
                'worker': {
                    'internet_access': False,
                    'gpu_enabled': False,
                    'gpu_id': 0,
                },
                'quality_assurance': {
                    'browse_image': {
                        'enabled': True,
                        'complex_to_real': 'amplitude',
                        'percent_low': 0,
                        'percent_high': 95,
                        'gamma': 0.5,
                        'equalize': False,
                    },
                    'perform_qa': True,
                    'output_to_json': False,
                },
                'output': {
                    'cslc_data_type': 'complex64_zero_mantissa',
                    'compression_enabled': True,
                    'compression_level': 4,
                    'chunk_size': [128, 128],
                    'shuffle': True,
                },
            },
        },
    }
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)



def _s1_cslc_worker(args):
    """Module-level worker for multiprocessing s1_cslc.py calls."""
    burst_id, cfg_path, log_path = args
    import subprocess as _sp
    with open(log_path, 'w') as lf:
        return _sp.run(
            ['s1_cslc.py', '--grid', 'geo', str(cfg_path)],
            stdout=lf, stderr=_sp.STDOUT).returncode


def run_s1_cslc_parallel(tasks, n_workers=4):
    """Run multiple s1_cslc.py jobs in parallel.

    Parameters
    ----------
    tasks : list of (burst_id, cfg_path, log_path)
        One tuple per job.
    n_workers : int
        Maximum number of parallel workers (default 4).

    Returns
    -------
    ok : int
        Number of successful jobs (exit code 0).
    """
    import multiprocessing as _mp
    n_workers = min(_mp.cpu_count(), n_workers, len(tasks))
    print(f'CSLC processing ({len(tasks)} jobs, {n_workers} workers)...')
    with _mp.Pool(processes=n_workers) as pool:
        results = pool.map(_s1_cslc_worker, tasks)
    ok = sum(r == 0 for r in results)
    print(f'CSLC processing complete ({ok}/{len(tasks)} succeeded).')
    return ok


def write_static_layers_config(src_cfg_path, dst_cfg_path):
    """Create a static-layers runconfig by copying a CSLC config and
    changing the product_type to trigger s1_static_layers processing.

    Parameters
    ----------
    src_cfg_path : str or Path
        Path to an existing CSLC geo runconfig YAML.
    dst_cfg_path : str or Path
        Output path for the static-layers runconfig.
    """
    with open(src_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg['runconfig']['groups']['primary_executable']['product_type'] = 'CSLC_S1_STATIC'
    with open(dst_cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def generate_static_layers(burst_id_list, config_dir, process_dir,
                           template_ymd):
    """Generate static layers (layover_shadow_mask, local_incidence_angle,
    LOS vectors) for each burst ID.

    Static layers are date-independent — one run per burst covers all
    dates.  Produces ``static_layers_<burst_id>.h5`` files.

    Parameters
    ----------
    burst_id_list : list of str
        Burst identifiers.
    config_dir : Path
        Directory containing CSLC geo runconfig YAML files.
    process_dir : Path
        Process directory (parent of logs/ and CSLC/).
    template_ymd : str
        Any valid YMD for which a config exists; used as template.

    Returns
    -------
    ok : int
        Number of bursts successfully generated.
    """
    logs_dir = process_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    process_dir.joinpath('CSLC').mkdir(parents=True, exist_ok=True)

    tasks = []
    for burst_id in burst_id_list:
        src = config_dir / f'geo_runconfig_{template_ymd}_{burst_id}.yaml'
        dst = config_dir / f'static_layers_{burst_id}.yaml'
        write_static_layers_config(str(src), str(dst))
        tasks.append((burst_id, dst, logs_dir / f'static_layers_{burst_id}.log'))

    return run_s1_cslc_parallel(tasks, n_workers=4)

def find_burst_inputs(ymd, safe_base, orbit_dir, tec_dir, burst_id=None):
    """Locate the SAFE, orbit (EOF) and TEC files for one acquisition date.

    When *burst_id* is provided and multiple SAFE files exist for the same
    date (multi-IW scenario), the correct SAFE is selected by matching the
    IW swath number from *burst_id* against the annotation filenames inside
    each SAFE directory.

    Parameters
    ----------
    ymd : str
        Acquisition date as ``YYYYMMDD``.
    safe_base : str or Path
        Directory containing the SAFE products.
    orbit_dir : str or Path
        Directory containing the orbit (EOF) files.
    tec_dir : str or Path
        Directory containing the IONEX TEC files.
    burst_id : str, optional
        Burst identifier (e.g. ``t124_264305_iw2``) used to disambiguate
        SAFE files when multiple exist for the same date.

    Returns
    -------
    safe_path : str or None
        Matching SAFE path (or None if not found).
    orbit_path : str or None
        Orbit file covering the date, preferring POEORB over RESORB.
    tec_path : str or None
        First available IONEX file (or None).
    """
    safe_base = Path(safe_base)
    orbit_dir = Path(orbit_dir)
    tec_dir = Path(tec_dir)

    safe_hits = sorted(str(p) for p in safe_base.glob(f'S1*{ymd}*.SAFE'))
    safe_path = None

    if len(safe_hits) == 1:
        safe_path = safe_hits[0]
    elif burst_id is not None:
        iw_num = str(burst_id).split('_')[-1][-1]  # 'iw2' → '2'
        for safe_candidate in safe_hits:
            ann_pattern = str(Path(safe_candidate) / 'annotation' /
                              f'*iw{iw_num}*slc*vv*.xml')
            ann_files = glob.glob(ann_pattern)
            for ann_file in ann_files:
                try:
                    tree = etree.parse(ann_file)
                    root = tree.getroot()
                    burst_list = root.find('.//{*}burstList')
                    if burst_list is not None and len(burst_list) > 0:
                        safe_path = safe_candidate
                        break
                except Exception:
                    continue
            if safe_path is not None:
                break
    if safe_path is None and safe_hits:
        safe_path = safe_hits[0]

    # Prefer an orbit whose validity window covers the date (POEORB), else RESORB.
    orbit_path = None
    for eof in sorted(str(p) for p in orbit_dir.glob('*.EOF')):
        m = re.search(r'V(\d{8})T\d{6}_(\d{8})T\d{6}', eof)
        if m and m.group(1) <= ymd <= m.group(2):
            orbit_path = eof
            break
    if orbit_path is None:
        res_hits = sorted(str(p) for p in orbit_dir.glob(f'S1*RESORB*{ymd}*.EOF'))
        orbit_path = res_hits[0] if res_hits else None

    gim_hits = list(tec_dir.glob('*GIM.INX'))
    if gim_hits:
        tec_hits = sorted(str(p) for p in gim_hits)
    else:
        tec_hits = sorted(str(p) for p in tec_dir.glob('jplg*'))
    tec_path = tec_hits[0] if tec_hits else None

    return safe_path, orbit_path, tec_path



def _compute_burst_id(azimuth_time_str, azimuth_anx_time, rel_orbit, subswath):
    """Compute Sentinel-1 burst ID string using the s1reader library.

    Delegates to ``s1reader.s1_burst_id.S1BurstId.from_burst_params()``
    which implements ESA Sentinel-1 Level 1 Detailed Algorithm Definition
    §9 equations 9-89/9-91 with proper IW subswath timing offsets
    and equator-crossing handling.

    Parameters
    ----------
    azimuth_time_str : str
        ISO-8601 azimuth time from the SAFE annotation ``<azimuthTime>`` tag
        (e.g. ``2020-01-22T03:25:14.262507``).
    azimuth_anx_time : float
        Mid-burst time w.r.t. ascending node crossing (seconds), from
        ``<azimuthAnxTime>`` in the burst annotation XML.
    rel_orbit : int
        Relative orbit (track) number, 1–175.
    subswath : str
        Subswath name, e.g. ``'IW2'``.

    Returns
    -------
    str
        Full burst ID string, e.g. ``'t124_264304_iw2'``.
    """
    import datetime
    azimuth_time = datetime.datetime.fromisoformat(azimuth_time_str)
    ascending_node_dt = azimuth_time - datetime.timedelta(seconds=azimuth_anx_time)

    burst_id = S1BurstId.from_burst_params(
        sensing_time=azimuth_time,
        ascending_node_dt=ascending_node_dt,
        start_track=rel_orbit,
        end_track=rel_orbit,
        subswath=subswath,
    )
    return str(burst_id)


def discover_burst_ids(safe_base, date_ymd_list):
    """Discover burst IDs from downloaded SAFE directories.

    Scans annotation XML files inside SAFE directories to extract
    burst IDs and groups them by date, returning a deduplicated
    burst ID list and corresponding date list.

    Parameters
    ----------
    safe_base : str or Path
        Directory containing the downloaded SAFE products.
    date_ymd_list : list of str
        Acquisition dates as ``YYYYMMDD`` strings.

    Returns
    -------
    burst_id_list : list of str
        Deduplicated burst IDs (e.g. ``['t124_264305_iw2', ...]``).
    date_list : list of str
        Dates as ``YYYY-MM-DD`` for each burst.
    """
    # Relative orbit offsets: S1A = 73, S1B = 27
    _REL_ORBIT_OFFSET = {'S1A': 73, 'S1B': 27}

    safe_base = Path(safe_base)
    burst_id_set = set()
    date_list = []

    for ymd in date_ymd_list:
        safe_hits = sorted(str(p) for p in safe_base.glob(f'S1*{ymd}*.SAFE'))
        for safe_path in safe_hits:
            ann_dir = Path(safe_path) / 'annotation'
            ann_files = sorted(glob.glob(str(ann_dir / '*-slc-vv-*.xml')))
            for ann_file in ann_files:
                try:
                    tree = etree.parse(ann_file)
                    root = tree.getroot()

                    mission_id = root.find('.//{*}missionId').text
                    abs_orbit = int(root.find('.//{*}absoluteOrbitNumber').text)
                    offset = _REL_ORBIT_OFFSET.get(mission_id, 73)
                    rel_orbit = (abs_orbit - offset) % 175 + 1

                    iw_num = root.find('.//{*}swath').text.lower()
                    burst_list = root.find('.//{*}burstList')
                    if burst_list is None:
                        continue
                    for b_elem in burst_list:
                        b_id_elem = b_elem.find('.{*}burstId')
                        if b_id_elem is not None:
                            full_id = f't{rel_orbit}_{b_id_elem.text}_{iw_num}'
                        else:
                            azt_time = b_elem.find('.{*}azimuthTime')
                            azt_anx = b_elem.find('.{*}azimuthAnxTime')
                            if azt_time is None or azt_anx is None:
                                continue
                            full_id = _compute_burst_id(
                                azt_time.text,
                                float(azt_anx.text),
                                rel_orbit,
                                iw_num.upper(),
                            )
                        if full_id not in burst_id_set:
                            burst_id_set.add(full_id)
                            date_list.append(f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}')
                except Exception:
                    continue

    return sorted(burst_id_set), date_list


# ===================================================================
# 7b. LOS angle computation from ISCE3 static-layers
# ===================================================================

def compute_los_angles(static_h5_path):
    """Compute ISCE2-style incidence and azimuth angles from ISCE3 static_layer HDF5.

    Reads ``los_east`` and ``los_north`` (ground-to-satellite unit vector
    components in ENU) from an OPERA-format static_layers HDF5 file and
    converts them to the incidence and azimuth angle convention used by
    ISCE2's ``los.rdr`` / ``los.rdr.geo`` products.

    * Band 1 — incidence angle: angle between satellite→target LOS and the
      local vertical at the target, in degrees (always positive).
    * Band 2 — azimuth angle: direction of the ground→satellite LOS measured
      anti-clockwise from North, in degrees [0°, 360°).

    Parameters
    ----------
    static_h5_path : str or Path
        Path to a ``static_layers_<burst_id>.h5`` HDF5 file produced by
        COMPASS / OPERA CSLC-S1-STATIC.

    Returns
    -------
    incidence : np.ndarray (float32)
        2-D incidence angle (degrees), same shape as the static layer grid.
    azimuth : np.ndarray (float32)
        2-D azimuth angle (degrees, anti-clockwise from North).
    gt : tuple
        GDAL geotransform ``(x0, dx, 0, y0, 0, dy)``.
    epsg : int
        EPSG code of the output projection (UTM).
    """
    with h5py.File(static_h5_path, 'r') as f:
        los_east = f['/data/los_east'][:]
        los_north = f['/data/los_north'][:]
        x0 = float(f['/data/x_coordinates'][0])
        dx = float(f['/data/x_spacing'][()])
        y0 = float(f['/data/y_coordinates'][0])
        dy = float(f['/data/y_spacing'][()])
        epsg = int(f['/data/projection'][()])

    up_sq = np.maximum(0, 1 - los_east**2 - los_north**2)
    up = np.sqrt(up_sq)

    incidence = np.arccos(up, out=np.full_like(up, np.nan), where=up > 0) * 180.0 / np.pi
    azimuth = (np.arctan2(los_north, los_east) - np.pi / 2) * 180.0 / np.pi
    azimuth = azimuth % 360.0
    azimuth[up == 0] = np.nan

    gt = (x0, dx, 0, y0, 0, dy)
    return incidence.astype(np.float32), azimuth.astype(np.float32), gt, epsg


def multilook_nearest(arr, az_looks, rg_looks):
    """Decimate by nearest-neighbour (every N-th row and column).

    Suitable for non-continuous data such as incidence and azimuth angles
    where averaging would distort the meaning.

    Parameters
    ----------
    arr : np.ndarray
        Input array, shape ``[rows, cols]`` or ``[bands, rows, cols]``.
    az_looks : int
        Decimation factor in the rows (azimuth) direction.
    rg_looks : int
        Decimation factor in the columns (range) direction.

    Returns
    -------
    ml : np.ndarray
        Downsampled array.
    """
    if arr.ndim == 2:
        return arr[::az_looks, ::rg_looks]
    else:
        return arr[:, ::az_looks, ::rg_looks]


def stitch_los_tiff(process_cslc_dir, burst_id_list, date_ymd,
                    out_gt=None, out_shape=None,
                    az_looks=1, rg_looks=1):
    """Stitch LOS angles from multiple static-layer bursts into two arrays.

    Computes incidence and azimuth angles from per-burst static-layer HDF5
    files, multilooks each burst (nearest-neighbour), then stitches the
    bands independently via :func:`stitch_arrays`.  The result is optionally
    cropped to match an existing interferogram extent.

    Parameters
    ----------
    process_cslc_dir : str or Path
        Top-level CSLC output directory (contains burst subdirectories).
    burst_id_list : list of str
        Burst identifiers, e.g. ``['t124_264305_iw2', ...]``.
    date_ymd : str
        Acquisition date string ``YYYYMMDD`` for locating the static-layer
        HDF5 file under ``<cslc_dir>/<burst_id>/<YYYYMMDD>/``.
    out_gt : tuple, optional
        GDAL geotransform ``(x0, dx, 0, y0, 0, dy)`` of the target output
        grid. If provided, the stitched LOS is cropped to this grid.
        When combined with multilooking, this should be the multilooked
        interferogram geotransform (e.g. ``gt_ml``).
    out_shape : tuple (rows, cols), optional
        Shape of the target output grid. Required when *out_gt* is given.
    az_looks : int, default 1
        Number of azimuth looks for multilooking before stitching (nearest).
    rg_looks : int, default 1
        Number of range looks for multilooking before stitching (nearest).

    Returns
    -------
    inc_stitched : np.ndarray (float32)
        Stitched incidence angle array (degrees), shape ``(rows, cols)``.
    az_stitched : np.ndarray (float32)
        Stitched azimuth angle array (degrees), shape ``(rows, cols)``.
    final_gt : tuple
        Geotransform of the output arrays.
    epsg : int
        EPSG code of the projection.
    """

    process_cslc_dir = Path(process_cslc_dir)

    inc_pieces = []
    az_pieces = []
    epsg = None
    ml_dx = ml_dy = None

    for burst_id in burst_id_list:
        h5_path = process_cslc_dir / burst_id / date_ymd / f'static_layers_{burst_id}.h5'
        if not h5_path.is_file():
            continue
        inc, az, gt, _epsg = compute_los_angles(h5_path)
        if epsg is None:
            epsg = _epsg

        if az_looks > 1 or rg_looks > 1:
            inc = multilook_nearest(inc, az_looks, rg_looks)
            az = multilook_nearest(az, az_looks, rg_looks)
            x0, dx, _, y0, _, dy = gt
            gt = (x0, dx * rg_looks, 0, y0, 0, dy * az_looks)

        ml_dx = gt[1]
        ml_dy = gt[5]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        wkt = srs.ExportToWkt()

        inc_pieces.append((inc, gt, wkt))
        az_pieces.append((az, gt, wkt))

    if not inc_pieces:
        raise FileNotFoundError(
            f'No static_layers HDF5 files found in {process_cslc_dir} '
            f'for bursts {burst_id_list} on {date_ymd}')

    # Build a geographic bbox covering the target extent (or all pieces)
    tf = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    if out_gt is not None and out_shape is not None:
        rows, cols = out_shape
        x0, _, _, y0, _, dy_val = out_gt
        x1 = x0 + cols * ml_dx
        y1 = y0 + rows * dy_val if dy_val < 0 else y0 + rows * ml_dy
        corners_x = [min(x0, x1), max(x0, x1), max(x0, x1), min(x0, x1)]
        corners_y = [max(y0, y1), max(y0, y1), min(y0, y1), min(y0, y1)]
    else:
        ux_all = []
        uy_all = []
        for _, gt, _ in inc_pieces:
            x0, dx, _, y0, _, dy = gt
            arr = inc_pieces[0][0]
            ux_all.extend([x0, x0 + arr.shape[1] * dx])
            uy_all.extend([y0, y0 + arr.shape[0] * dy])
        corners_x = [min(ux_all), max(ux_all), max(ux_all), min(ux_all)]
        corners_y = [max(uy_all), max(uy_all), min(uy_all), min(uy_all)]

    lons, lats = tf.transform(corners_x, corners_y)
    bbox_wsen = (min(lons), min(lats), max(lons), max(lats))

    # Stitch each band
    inc_stitched, union_gt, proj_wkt = stitch_arrays(
        inc_pieces, bbox_wsen, dx=ml_dx, dy=ml_dy, epsg_utm=epsg, method='last')
    az_stitched, _, _ = stitch_arrays(
        az_pieces, bbox_wsen, dx=ml_dx, dy=ml_dy, epsg_utm=epsg, method='last')

    # Crop to exact output grid if requested
    if out_gt is not None and out_shape is not None:
        rows, cols = out_shape
        ox, oy = out_gt[0], out_gt[3]
        px = int(round((ox - union_gt[0]) / union_gt[1]))
        py = int(round((oy - union_gt[3]) / union_gt[5]))
        inc_stitched = inc_stitched[py:py + rows, px:px + cols]
        az_stitched = az_stitched[py:py + rows, px:px + cols]
        union_gt = (ox, union_gt[1], 0, oy, 0, union_gt[5])

        if inc_stitched.shape[0] != rows or inc_stitched.shape[1] != cols:
            inc_padded = np.full((rows, cols), np.nan, dtype=np.float32)
            az_padded  = np.full((rows, cols), np.nan, dtype=np.float32)
            h = min(inc_stitched.shape[0], rows)
            w = min(inc_stitched.shape[1], cols)
            inc_padded[:h, :w] = inc_stitched[:h, :w]
            az_padded[:h, :w]  = az_stitched[:h, :w]
            inc_stitched = inc_padded
            az_stitched  = az_padded

    return inc_stitched, az_stitched, union_gt, epsg

# ===================================================================
# 8. Auxiliary dataset I/O
# ===================================================================

def read_aux_datasets(h5_path):
    """Read all auxiliary correction datasets from an OPERA CSLC H5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file.

    Returns
    -------
    aux : dict
        Dictionary containing:
        - azimuth_carrier_phase, azimuth_fm_rate_mismatch, slant_range,
          zero_doppler_time, bistatic_delay, geometry_steering_doppler,
          los_solid_earth_tides, azimuth_solid_earth_tides,
          los_ionospheric_delay, wet_los_troposphere_delay,
          dry_los_troposphere_delay (all numpy arrays)
        - x_coordinates, y_coordinates, epsg (geocoded grid metadata)
        - tropo_total (wet + dry, computed)
    """
    corr_grp = '/metadata/processing_information/timing_corrections'
    aux = {}

    with h5py.File(h5_path, 'r') as f:
        for name in ['slant_range', 'zero_doppler_time', 'bistatic_delay',
                      'geometry_steering_doppler', 'los_solid_earth_tides',
                      'azimuth_solid_earth_tides', 'los_ionospheric_delay',
                      'azimuth_fm_rate_mismatch']:
            ds_path = f'{corr_grp}/{name}'
            if ds_path in f:
                aux[name] = f[ds_path][:]

        has_wet = 'wet_los_troposphere_delay' in f[corr_grp]
        has_dry = 'dry_los_troposphere_delay' in f[corr_grp]
        aux['wet_los_troposphere_delay'] = (
            f[f'{corr_grp}/wet_los_troposphere_delay'][:] if has_wet
            else np.zeros_like(aux['los_solid_earth_tides']))
        aux['dry_los_troposphere_delay'] = (
            f[f'{corr_grp}/dry_los_troposphere_delay'][:] if has_dry
            else np.zeros_like(aux['los_solid_earth_tides']))
        aux['tropo_total'] = aux['wet_los_troposphere_delay'] + aux['dry_los_troposphere_delay']

        aux['azimuth_carrier_phase'] = f['/data/azimuth_carrier_phase'][:]
        aux['x_coordinates'] = f['/data/x_coordinates'][:]
        aux['y_coordinates'] = f['/data/y_coordinates'][:]
        aux['epsg'] = int(f['/data/projection'][()])

    return aux


def compute_static_troposphere_correction(h5_path, dem_path):
    """Compute static troposphere delay on the CSLC radar LUT grid.

    Clips the DEM to the CSLC geo-footprint, resamples to the LUT grid,
    computes per-pixel incidence angle, and applies the COMPASS static
    troposphere delay model (identical to compass.utils.lut).

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file.
    dem_path : str or Path
        Path to the DEM GeoTIFF.

    Returns
    -------
    tropo_disp : np.ndarray
        Static troposphere delay (m) on the radar LUT grid,
        shape ``(n_az, n_rg)``.
    """
    aux = read_aux_datasets(h5_path)
    sr = aux['slant_range']
    azt = aux['zero_doppler_time']
    x = aux['x_coordinates']
    y = aux['y_coordinates']
    epsg = aux['epsg']

    n_az, n_rg = len(azt), len(sr)

    dem_ds = gdal.Open(str(dem_path))
    dem_gt = dem_ds.GetGeoTransform()
    tf_utm2ll = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x_c = [x[0] - dx / 2, x[-1] + dx / 2, x[-1] + dx / 2, x[0] - dx / 2]
    y_c = [y[0] - dy / 2, y[0] - dy / 2, y[-1] + dy / 2, y[-1] + dy / 2]
    lon_c, lat_c = tf_utm2ll.transform(x_c, y_c)

    margin_deg = 0.1
    lon_margin = margin_deg / abs(dem_gt[1])
    lat_margin = margin_deg / abs(dem_gt[5])

    col0 = max(0, int(np.floor((min(lon_c) - dem_gt[0]) / dem_gt[1] - lon_margin)))
    col1 = min(dem_ds.RasterXSize,
               int(np.ceil((max(lon_c) - dem_gt[0]) / dem_gt[1] + lon_margin)) + 1)
    row0 = max(0, int(np.floor((max(lat_c) - dem_gt[3]) / dem_gt[5] - lat_margin)))
    row1 = min(dem_ds.RasterYSize,
               int(np.ceil((min(lat_c) - dem_gt[3]) / dem_gt[5] + lat_margin)) + 1)
    cols = col1 - col0
    rows = row1 - row0

    h_dem = dem_ds.GetRasterBand(1).ReadAsArray(col0, row0, cols, rows)
    dem_ds = None
    h_rg = resize(h_dem.astype(np.float32), (n_az, n_rg),
                  order=1, mode='edge', anti_aliasing=False)
    h_rg = np.maximum(h_rg, 0.0)

    inc_deg = compute_isce3_incidence_angle(h5_path)
    tropo_disp = compute_static_troposphere_delay(inc_deg, h_rg)

    return tropo_disp


# ===================================================================
# 9. Auxiliary dataset visualisation (Section 4.3)
# ===================================================================

def plot_geom_corrections(h5_path, burst_id='burst', date_str=''):
    """Read and visualise geometric correction datasets from a CSLC H5.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file.
    burst_id : str
        Burst label for print output.
    date_str : str
        Date label for print output.
    """
    import matplotlib.ticker as ticker

    aux = read_aux_datasets(h5_path)
    sr, azt = aux['slant_range'], aux['zero_doppler_time']
    bistatic = aux['bistatic_delay']
    geo_doppler = aux['geometry_steering_doppler']

    print(f'Geometry corrections for {burst_id} ({date_str}):')
    print(f'  geometry_steering_doppler — mean: {geo_doppler.mean():.4f} m, '
          f'std: {geo_doppler.std():.4f} m')
    print(f'  bistatic_delay            — mean: {bistatic.mean():.6f} s, '
          f'std: {bistatic.std():.6f} s')

    extent = [sr[0], sr[-1], azt[-1], azt[0]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_data(ax1, geo_doppler, title='geometry_steering_doppler',
              cmap='RdBu_r', extent=extent, cbar_label='Range shift (m)')
    ax1.set_xlabel('Slant range (m)')
    ax1.set_ylabel('Azimuth time (s)')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20000))
    plot_data(ax2, bistatic, title='bistatic_delay',
              cmap='RdBu_r', extent=extent, cbar_label='Azimuth shift (s)')
    ax2.set_xlabel('Slant range (m)')
    ax2.set_ylabel('Azimuth time (s)')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(20000))
    fig.suptitle('Geometry corrections — 2-D profiles', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    show_and_close()


def plot_phys_corrections(h5_path, dem_path, burst_id='burst', date_str=''):
    """Read and visualise physical correction datasets from a CSLC H5.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file.
    dem_path : str or Path
        Path to the DEM GeoTIFF.
    burst_id : str
        Burst label for print output.
    date_str : str
        Date label for print output.
    """
    aux = read_aux_datasets(h5_path)
    sr, azt = aux['slant_range'], aux['zero_doppler_time']
    tide_rg = aux['los_solid_earth_tides']
    tide_az = aux['azimuth_solid_earth_tides']
    iono = aux['los_ionospheric_delay']
    tropo_tot = aux['tropo_total']
    tropo_disp = compute_static_troposphere_correction(h5_path, dem_path)

    print(f'Physical corrections for {burst_id} ({date_str}):')
    print(f'  Solid Earth tide (LOS):      {np.nanmean(np.abs(tide_rg)):.4f} m')
    print(f'  Solid Earth tide (azimuth):  {np.nanmean(np.abs(tide_az)):.4f} s')
    print(f'  Ionospheric delay (LOS):     {np.nanmean(np.abs(iono)):.4f} m')
    print(f'  Static troposphere (LOS):    {np.nanmean(tropo_disp):.4f} m')
    print(f'  Weather-model tropo:         {np.nanmean(np.abs(tropo_tot)):.4f} m')

    extent = [sr[0], sr[-1], azt[-1], azt[0]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_items = [
        ('los_solid_earth_tides',  tide_rg,    'm'),
        ('los_ionospheric_delay',  iono,       'm'),
        ('Static troposphere',     tropo_disp, 'm'),
    ]
    for ax, (name, data, unit) in zip(axes, plot_items):
        plot_data(ax, data, title=name, cmap='RdBu_r', extent=extent,
                  cbar_label=f'Delay ({unit})', shrink=0.85)
        ax.set_xlabel('Slant range (m)')
        ax.set_ylabel('Azimuth time (s)')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    fig.suptitle('Physical corrections', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    show_and_close()


def plot_focus_corrections(h5_path, burst_id='burst', date_str=''):
    """Read and visualise focusing correction datasets from a CSLC H5.

    Parameters
    ----------
    h5_path : str or Path
        Path to the OPERA CSLC H5 file.
    burst_id : str
        Burst label for print output.
    date_str : str
        Date label for print output.
    """
    import matplotlib.ticker as ticker

    aux = read_aux_datasets(h5_path)
    sr, azt = aux['slant_range'], aux['zero_doppler_time']
    fm_mismatch = aux['azimuth_fm_rate_mismatch']
    az_carrier = aux['azimuth_carrier_phase']
    x_utm, y_utm, epsg = aux['x_coordinates'], aux['y_coordinates'], aux['epsg']

    print(f'Focusing corrections for {burst_id} ({date_str}):')
    print(f'  FM rate mismatch:  mean={fm_mismatch.mean():.6e} s, '
          f'std={fm_mismatch.std():.6e} s')
    print(f'  Azimuth carrier phase: mean={np.nanmean(az_carrier):.2f} rad, '
          f'std={np.nanstd(az_carrier):.2f} rad')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_data(ax1, fm_mismatch, title='azimuth_fm_rate_mismatch',
              cmap='bwr', extent=[sr[0], sr[-1], azt[-1], azt[0]],
              cbar_label='Azimuth shift (s)')
    ax1.set_xlabel('Slant range (m)')
    ax1.set_ylabel('Azimuth time (s)')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20000))

    ss_rg = slice(0, az_carrier.shape[1], 10)
    ss_az = slice(0, az_carrier.shape[0], 10)
    plot_data(ax2, az_carrier[ss_az, ss_rg], title='azimuth_carrier_phase (10x subsampled)',
              cmap='twilight', extent=[x_utm[0], x_utm[-1], y_utm[-1], y_utm[0]],
              cbar_label='Phase (rad)')
    ax2.set_xlabel(f'Easting (m, EPSG:{epsg})')
    ax2.set_ylabel(f'Northing (m, EPSG:{epsg})')
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    fig.suptitle('Focusing corrections — 2-D profiles', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    show_and_close()


def load_water_mask(gt_ml, ml_shape, epsg_utm, wbd_dir=None):
    """Load water mask resampled to a target UTM grid via GDAL nearest-neighbour reprojection.

    Reads the pre-downloaded ``swbd_nasadem.wbd`` binary raster and warps it
    onto the target multilooked grid using :func:`gdal.ReprojectImage` with
    ``GRA_NearestNeighbour``.  Returns a uint8 array where **1 = water, 0 = land**.

    Parameters
    ----------
    gt_ml : tuple
        GDAL geotransform ``(x0, dx, 0, y0, 0, dy)`` of the target grid.
    ml_shape : tuple (rows, cols)
        Shape of the target grid.
    epsg_utm : int
        UTM EPSG code of the target grid.
    wbd_dir : Path or str, optional
        Directory containing ``swbd_nasadem.wbd`` and ``swbd_nasadem.json``.
        Defaults to ``<project>/process/DEM``.

    Returns
    -------
    mask : np.ndarray (uint8)
        ``[rows, cols]`` with 1 = water, 0 = land.
    """
    import json as _json

    if wbd_dir is None:
        wbd_dir = Path(__file__).resolve().parent / 'process' / 'DEM'
    wbd_dir = Path(wbd_dir)

    # --- read WBD metadata and binary raster ---
    with open(wbd_dir / 'swbd_nasadem.json') as _fj:
        meta = _json.load(_fj)
    raw = np.fromfile(str(wbd_dir / 'swbd_nasadem.wbd'), dtype=np.uint8)
    wbd = raw.reshape(meta['height'], meta['width'])

    # --- wrap WBD as in-memory GDAL dataset (WGS84) ---
    mem_drv = gdal.GetDriverByName('MEM')

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)
    wbd_gt = (meta['lon0'], meta['dlon'], 0, meta['lat0'], 0, meta['dlat'])

    src_ds = mem_drv.Create('', meta['width'], meta['height'], 1, gdal.GDT_Byte)
    src_ds.SetGeoTransform(wbd_gt)
    src_ds.SetProjection(srs_wgs84.ExportToWkt())
    src_ds.GetRasterBand(1).WriteArray(wbd)

    # --- create target UTM grid ---
    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromEPSG(epsg_utm)
    rows, cols = ml_shape

    dst_ds = mem_drv.Create('', cols, rows, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(gt_ml)
    dst_ds.SetProjection(srs_utm.ExportToWkt())

    # --- warp: WGS84 → UTM, nearest-neighbour ---
    gdal.ReprojectImage(
        src_ds, dst_ds,
        srs_wgs84.ExportToWkt(), srs_utm.ExportToWkt(),
        gdal.GRA_NearestNeighbour,
    )

    warped = dst_ds.GetRasterBand(1).ReadAsArray()

    # close MEM datasets
    src_ds = None
    dst_ds = None

    # 1 = water, 0 = land
    return (warped > 0).astype(np.bool_)


# ===================================================================
# 10. Water body mask
# ===================================================================

def download_nasadem_water_mask(bbox_wsen, output_dir):
    """Download NASADEM HGT tiles and stitch a water-body mask raster.

    Downloads 1-arcsecond NASADEM tiles covering *bbox_wsen* from the
    NASA Earthdata Cloud, extracts the water mask from bit 15 of each
    int16 pixel, and saves a BYTE raster (255=water, 0=land) as
    ``swbd_nasadem.wbd`` in *output_dir*.

    Authentication uses ``~/.netrc``.  Already-downloaded tiles are
    cached in ``~/.cache/sardem/``.

    Parameters
    ----------
    bbox_wsen : tuple
        WGS84 bounding box ``(west, south, east, north)`` in degrees.
    output_dir : str or Path
        Directory where ``swbd_nasadem.wbd`` is written.
    """
    import requests
    import subprocess
    import zipfile
    import io
    from netrc import netrc as _netrc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'swbd_nasadem.wbd'

    # Earthdata credentials
    nrc = _netrc(Path.home() / '.netrc')
    auth = nrc.authenticators('urs.earthdata.nasa.gov')
    if auth is None:
        auth = nrc.authenticators('e4ftl01.cr.usgs.gov')
    user, _, pwd = auth if auth else (None, None, None)
    if not user or not pwd:
        raise RuntimeError('No Earthdata credentials in ~/.netrc')

    # Determine tile range
    west, south, east, north = bbox_wsen
    lon_start = int(np.floor(west))
    lon_end = int(np.floor(east))
    lat_start = int(np.floor(south))
    lat_end = int(np.floor(north))

    cache_dir = Path.home() / '.cache' / 'sardem'
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_url = ('https://data.lpdaac.earthdatacloud.nasa.gov/'
                'lp-prod-protected/NASADEM_HGT.001')

    # Full raster: 1 arcsec, coverage per tile = 3601×3601 (1°×1° + edge overlap)
    stride = 3600
    total_lat = (lat_end - lat_start + 1) * stride
    total_lon = (lon_end - lon_start + 1) * stride
    full = np.full((total_lat, total_lon), 255, dtype=np.uint8)

    for lat_idx in range(lat_start, lat_end + 1):
        for lon_idx in range(lon_start, lon_end + 1):
            lat_pfx = 's' if lat_idx < 0 else 'n'
            lon_pfx = 'w' if lon_idx < 0 else 'e'
            tile = f'{lat_pfx}{abs(lat_idx):02d}{lon_pfx}{abs(lon_idx):03d}'
            filename = f'NASADEM_HGT_{tile}'
            zip_path = cache_dir / f'{filename}.zip'
            hgt_path = cache_dir / f'{tile}.hgt'

            # Download if not cached
            if not hgt_path.exists():
                url = f'{base_url}/{filename}/{filename}.zip'
                print(f'  Downloading {tile} ...', end=' ', flush=True)
                r = requests.get(url, auth=(user, pwd), timeout=60)
                if r.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        f.write(r.content)
                    with zipfile.ZipFile(zip_path) as zf:
                        for member in zf.namelist():
                            if member.endswith('.hgt'):
                                zf.extract(member, cache_dir)
                                extracted = Path(cache_dir) / member
                                extracted.rename(hgt_path)
                                break
                    print('OK')
                elif r.status_code == 404:
                    print('not found (ocean tile)')
                    continue
                else:
                    print(f'HTTP {r.status_code}')
                    continue
            else:
                print(f'  {tile}: using cached {hgt_path}')

            # Read tile, extract water mask
            h = np.fromfile(hgt_path, dtype='>i2').reshape(3601, 3601)
            # Remove 1-pixel overlap edge: 3601→3600
            h = h[:stride, :stride]
            water = ((h >> 15) & 1) | (h == -32768) | (h <= 0)
            water = water.astype(np.uint8) * 255

            # Place in full raster (row 0 = north = highest lat)
            row = (lat_end - lat_idx) * stride
            col = (lon_idx - lon_start) * stride
            full[row:row + stride, col:col + stride] = water

    full.tofile(str(out_path))

    # Save geo-metadata alongside the .wbd file
    import json as _json
    _dlon = 1.0 / stride
    _dlat = -1.0 / stride
    meta = {
        'width': full.shape[1],
        'height': full.shape[0],
        'lon0': float(lon_start),
        'lat0': float(lat_end + 1),
        'dlon': _dlon,
        'dlat': _dlat,
    }
    with open(str(out_path).replace('.wbd', '.json'), 'w') as _f:
        _json.dump(meta, _f)

    water_pct = 100.0 * (full == 255).sum() / full.size
    land_pct = 100.0 * (full == 0).sum() / full.size
    print(f'Saved {out_path} ({full.shape[1]}×{full.shape[0]}, '
          f'water={water_pct:.1f}%, land={land_pct:.1f}%)')



# ===================================================================
# Plotting helpers: coordinate extents and axis formatting
# ===================================================================

def extent_utm(gt, shape):
    """Compute imshow extent in UTM coordinates from GDAL geotransform.

    Parameters
    ----------
    gt : tuple
        GDAL geotransform ``(x0, dx, 0, y0, 0, dy)``.
    shape : tuple
        Array shape ``(rows, cols)``.

    Returns
    -------
    extent : list
        ``[left, right, bottom, top]`` in UTM metres.
    """
    nrows, ncols = shape
    x0, dx, _, y0, _, dy = gt
    left = x0
    right = x0 + ncols * dx
    bottom = y0 + nrows * dy
    top = y0
    return [left, right, bottom, top]


def extent_latlon(gt, shape, src_epsg):
    """Compute imshow extent in EPSG:4326 from a UTM geotransform.

    Transforms the four corners of the image extent from *src_epsg*
    (e.g. 32605) to EPSG:4326 and returns the bounding box for
    ``imshow(..., extent=...)`` with ``origin='upper'``.

    Parameters
    ----------
    gt : tuple
        GDAL geotransform ``(x0, dx, 0, y0, 0, dy)`` in *src_epsg*.
    shape : tuple
        Array shape ``(rows, cols)``.
    src_epsg : int
        Source EPSG code (e.g. 32605 for UTM zone 5N).

    Returns
    -------
    extent : list
        ``[lon_left, lon_right, lat_bottom, lat_top]`` in decimal degrees.
    """
    nrows, ncols = shape
    x0, dx, _, y0, _, dy = gt
    xs = [x0, x0 + ncols * dx, x0 + ncols * dx, x0]
    ys = [y0, y0, y0 + nrows * dy, y0 + nrows * dy]
    tf = Transformer.from_crs(f'EPSG:{src_epsg}', 'EPSG:4326', always_xy=True)
    lons, lats = tf.transform(xs, ys)
    return [min(lons), max(lons), min(lats), max(lats)]

def extent_pixel(shape):
    """Compute imshow extent for pixel-index display.
    
    Returns [-0.5, cols-0.5, rows-0.5, -0.5] so pixel centres
    align with integer indices.
    """
    nrows, ncols = shape
    return [-0.5, ncols - 0.5, nrows - 0.5, -0.5]

def set_ax_utm(ax, epsg, fmt='.0f'):
    """Format axis for UTM (easting/northing) display.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    epsg : int
        EPSG code for the UTM zone.
    fmt : str
        Tick format string (default ``'%.0f'`` for integer metres).
    """
    from matplotlib.ticker import FuncFormatter
    ax.set_xlabel(f'Easting (m, EPSG:{epsg})')
    ax.set_ylabel(f'Northing (m, EPSG:{epsg})')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:{fmt}}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:{fmt}}'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


def set_ax_pixel(ax):
    """Format axis for pixel-index display.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    """
    ax.set_xlabel('Column (px)')
    ax.set_ylabel('Row (px)')


# ===================================================================
# Plotting: convenience functions for imshow + colorbar + title
# ===================================================================

def plot_data(ax, data, title=None, cmap='jet', vmin=None, vmax=None,
              extent=None, aspect='auto', cbar_label=None, alpha=None,
              origin='upper', shrink=0.8):
    """Plot a 2-D array on *ax* with imshow, colorbar and title.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : np.ndarray   2-D array to display.
    title : str, optional   Title text (no title when None).
    cmap : str   Colormap name (default ``'jet'``).
    vmin, vmax : float, optional   Imshow value range.
    extent : list, optional   ``[left, right, bottom, top]`` for imshow.
    aspect : str   Aspect ratio (default ``'auto'``).
    cbar_label : str, optional   Colorbar label (no bar when None).
    alpha : float, optional   Transparency.
    origin : str   Image origin (default ``'upper'``).
    shrink : float   Colorbar shrink factor.

    Returns
    -------
    im : matplotlib.image.AxesImage
    """
    kw = dict(cmap=cmap, aspect=aspect, origin=origin, extent=extent)
    if vmin is not None:
        kw['vmin'] = vmin
    if vmax is not None:
        kw['vmax'] = vmax
    if alpha is not None:
        kw['alpha'] = alpha
    im = ax.imshow(data, **kw)
    if title is not None:
        ax.set_title(title)
    if cbar_label is not None:
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=shrink)
    return im


def plot_phase(ax, phase, title=None, extent=None, **kwargs):
    """Plot wrapped phase with jet colormap (default -pi to pi).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    phase : np.ndarray   2-D wrapped phase (radians).
    title : str, optional
    extent : list, optional   ``[left, right, bottom, top]``.
    **kwargs   Passed to :func:`plot_data`.
    """
    kwargs.setdefault('cmap', 'jet')
    kwargs.setdefault('vmin', -np.pi)
    kwargs.setdefault('vmax', np.pi)
    kwargs.setdefault('cbar_label', 'Phase (rad)')
    return plot_data(ax, phase, title=title, extent=extent, **kwargs)


def plot_amplitude(ax, amp_db, title=None, extent=None, **kwargs):
    """Plot amplitude in dB with gray colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    amp_db : np.ndarray   2-D amplitude in dB.
    title : str, optional
    extent : list, optional
    **kwargs   Passed to :func:`plot_data`.
    """
    kwargs.setdefault('cmap', 'gray')
    kwargs.setdefault('vmin', 30)
    kwargs.setdefault('vmax', 45)
    kwargs.setdefault('cbar_label', 'Amplitude (dB)')
    return plot_data(ax, amp_db, title=title, extent=extent, **kwargs)


def plot_coherence(ax, coh, title=None, extent=None, **kwargs):
    """Plot coherence 0-1 with gray colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    coh : np.ndarray   2-D coherence array, values in [0, 1].
    title : str, optional
    extent : list, optional
    **kwargs   Passed to :func:`plot_data`.
    """
    kwargs.setdefault('cmap', 'gray')
    kwargs.setdefault('vmin', 0)
    kwargs.setdefault('vmax', 1)
    kwargs.setdefault('cbar_label', 'γ')
    return plot_data(ax, coh, title=title, extent=extent, **kwargs)


def plot_los(ax, angle, title=None, extent=None, **kwargs):
    """Plot LOS angle or incidence angle with viridis colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    angle : np.ndarray   2-D angle in degrees.
    title : str, optional
    extent : list, optional
    **kwargs   Passed to :func:`plot_data`.
    """
    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('cbar_label', 'deg')
    return plot_data(ax, angle, title=title, extent=extent, **kwargs)


def plot_phase_over_hillshade(ax, phase, hillshade, title=None, extent=None,
                               alpha=0.8, **kwargs):
    """Plot phase overlay on a hillshade background.

    Renders the hillshade in gray, then overlays the phase with
    the given *alpha* transparency and jet colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    phase : np.ndarray   2-D phase array (radians).
    hillshade : np.ndarray   2-D hillshade (0-1).
    title : str, optional
    extent : list, optional
    alpha : float   Phase transparency (default 0.8).
    **kwargs   Passed to :func:`plot_data` for the phase layer.
    """
    ax.imshow(hillshade, cmap='gray', extent=extent, origin='upper', vmin=0, vmax=1)
    kwargs.setdefault('cmap', 'jet')
    kwargs.setdefault('vmin', -np.pi)
    kwargs.setdefault('vmax', np.pi)
    kwargs.setdefault('cbar_label', 'Phase (rad)')
    return plot_data(ax, phase, title=title, extent=extent, alpha=alpha, **kwargs)


def show_and_close(fig=None):
    """Display all figures and close them."""
    plt.show()
    plt.close('all')


# ===================================================================
# Composite plotting: full lifecycle from figsize to close
# ===================================================================

def _plot_two(data1, data2, *,
              plot1=plot_data, kw1=None,
              plot2=plot_data, kw2=None,
              coord=None, epsg=None,
              xlabel1=None, ylabel1=None,
              xlabel2=None, ylabel2=None,
              figsize=(8, 3), tight_layout=True, suptitle=None):
    """Generic 1x2 panel plot (internal).

    Parameters
    ----------
    data1, data2 : np.ndarray
    plot1, plot2 : callable   Per-axis functions (plot_phase, plot_coherence, ...).
    kw1, kw2 : dict, optional   Keyword arguments for each plotter.
    coord : 'pixel', 'utm' or None   Coordinate axis formatting.
    epsg : int   Required when coord='utm'.
    xlabel1, ylabel1, xlabel2, ylabel2 : str, optional   Per-axis labels.
    figsize : tuple
    tight_layout : bool
    suptitle : str, optional
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    k1 = kw1 if kw1 is not None else {}
    k2 = kw2 if kw2 is not None else {}
    plot1(ax1, data1, **k1)
    plot2(ax2, data2, **k2)
    if xlabel1 is not None:
        ax1.set_xlabel(xlabel1)
    if ylabel1 is not None:
        ax1.set_ylabel(ylabel1)
    if xlabel2 is not None:
        ax2.set_xlabel(xlabel2)
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2)
    if coord == 'pixel':
        set_ax_pixel(ax1)
        set_ax_pixel(ax2)
    elif coord == 'utm':
        set_ax_utm(ax1, epsg)
        set_ax_utm(ax2, epsg)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=12, fontweight='bold')
    if tight_layout:
        plt.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    show_and_close()


def plot_amplitude_pair(amp1, amp2, title1, title2, ext1=None, ext2=None,
                        xlabel1=None, ylabel1=None, xlabel2=None, ylabel2=None,
                        figsize=(8, 4), suptitle=None):
    """1x2: two amplitude panels (gray, dB)."""
    _plot_two(amp1, amp2,
              plot1=plot_amplitude, kw1=dict(title=title1, extent=ext1),
              plot2=plot_amplitude, kw2=dict(title=title2, extent=ext2),
              xlabel1=xlabel1, ylabel1=ylabel1,
              xlabel2=xlabel2, ylabel2=ylabel2,
              figsize=figsize, suptitle=suptitle)


def plot_ifg_coherence(phase, coh, phase_extent=None, coh_extent=None,
                       phase_title='Stitched interferogram (phase)',
                       coh_title='Complex coherence (5x5 window)',
                       figsize=(8, 3)):
    """1x2: phase (jet) + coherence (gray), pixel coordinates."""
    _plot_two(phase, coh,
              plot1=plot_phase, kw1=dict(title=phase_title, extent=phase_extent),
              plot2=plot_coherence, kw2=dict(title=coh_title, extent=coh_extent),
              coord='pixel', figsize=figsize)


def plot_phase_triple(ph1, ph2, ph3, title1, title2, title3,
                      ext1=None, figsize=(8, 2)):
    """1x3: three phase panels (jet), only last column has colorbar.

    Parameters
    ----------
    ph1, ph2, ph3 : np.ndarray   Wrapped phase arrays (radians).
    title1, title2, title3 : str   Panel titles.
    ext1 : list, optional   Extent for panel 1 only.
    figsize : tuple   Figure size.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    plot_phase(axes[0], ph1, title=title1, cbar_label=None, extent=ext1)
    set_ax_pixel(axes[0])
    plot_phase(axes[1], ph2, title=title2, cbar_label=None)
    set_ax_pixel(axes[1])
    plot_phase(axes[2], ph3, title=title3)
    set_ax_pixel(axes[2])
    show_and_close()


def plot_coherence_pair(coh1, coh2, ext1=None, ext2=None,
                        title1='Complex coherence',
                        title2='Phase-sigma coherence',
                        cbar_label1='\u03b3', cbar_label2='\u03b3_phsig',
                        figsize=(8, 3)):
    """1x2: two coherence panels (gray, 0-1), pixel coordinates."""
    _plot_two(coh1, coh2,
              plot1=plot_coherence, kw1=dict(title=title1, cbar_label=cbar_label1, extent=ext1),
              plot2=plot_coherence, kw2=dict(title=title2, cbar_label=cbar_label2, extent=ext2),
              coord='pixel', figsize=figsize)


def plot_los_pair(inc, az, extent, epsg,
                  inc_title='LOS Incidence Angle (deg)',
                  az_title='LOS Azimuth Angle (deg)',
                  figsize=(8, 3)):
    """1x2: LOS incidence and azimuth angle (viridis), UTM coordinates."""
    _plot_two(inc, az,
              plot1=plot_los, kw1=dict(title=inc_title, extent=extent),
              plot2=plot_los, kw2=dict(title=az_title, extent=extent),
              coord='utm', epsg=epsg, figsize=figsize)


def plot_phase_hillshade_pair(ph1, ph2, hillshade, extent_deg,
                               title1='Wrapped interferogram',
                               title2='Unwrapped phase',
                               alpha1=0.8, alpha2=0.6,
                               vmin1=None, vmax1=None,
                               vmin2=None, vmax2=None,
                               figsize=(8, 3), suptitle=None):
    """1x2: phase overlays on hillshade, EPSG:4326 coordinates.

    Parameters
    ----------
    ph1, ph2 : np.ndarray   Wrapped / unwrapped phase (radians).
    hillshade : np.ndarray   Hillshade array [0, 1].
    extent_deg : list   [lon_left, lon_right, lat_bottom, lat_top].
    title1, title2 : str   Panel titles.
    alpha1, alpha2 : float   Phase opacity.
    vmin1, vmax1, vmin2, vmax2 : float, optional   Phase colour range.
    figsize : tuple   Figure size.
    suptitle : str, optional   Overall figure title.
    """
    import matplotlib.ticker as ticker
    _vmin1 = vmin1 if vmin1 is not None else -np.pi
    _vmax1 = vmax1 if vmax1 is not None else np.pi
    _vmin2 = vmin2
    _vmax2 = vmax2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plot_phase_over_hillshade(ax1, ph1, hillshade, title=title1,
                               extent=extent_deg, alpha=alpha1,
                               vmin=_vmin1, vmax=_vmax1)
    ax1.set_xlabel('Longitude (\u00b0E)')
    ax1.set_ylabel('Latitude (\u00b0N)')
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

    kw2 = dict(extent=extent_deg, alpha=alpha2)
    if _vmin2 is not None:
        kw2['vmin'] = _vmin2
    if _vmax2 is not None:
        kw2['vmax'] = _vmax2
    plot_phase_over_hillshade(ax2, ph2, hillshade, title=title2, **kw2)
    ax2.set_xlabel('Longitude (\u00b0E)')
    ax2.set_ylabel('Latitude (\u00b0N)')
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else None)
    show_and_close()
