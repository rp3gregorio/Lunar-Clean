"""
dem.py — Load the LOLA Digital Elevation Model (DEM) and extract
         topographic properties at any lunar location.

The DEM is a PDS3-formatted binary file (LDEM_*.IMG) distributed by the
Lunar Reconnaissance Orbiter (LRO) mission.  Label files (*.LBL) describe
the binary format so we can read it correctly.

Key functions
-------------
load_ldem()                   — Auto-detect and load a LDEM file from disk.
latlon_to_pixel()             — Convert (lat, lon) → pixel row/column.
compute_slope_aspect()        — Slope and aspect from central differences.
extract_point(lat, lon, ...)  — One-stop function: pixel + slope + aspect.
"""

import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lunar.constants import R_MOON


# ─────────────────────────────────────────────────────────────────────────────
# PDS3 LABEL PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pds3_label(lbl_path: Path) -> dict:
    """
    Read a PDS3 label file (.LBL) and return a flat key-value dict.

    The label describes the binary image: size, pixel bit-depth, byte order,
    offset to the start of image data, and which IMG/JP2 file holds the data.
    """
    txt = lbl_path.read_text(encoding='utf-8', errors='ignore')
    label = {}

    # Strip block comments  /* … */
    clean = re.sub(r'/\*.*?\*/', '', txt, flags=re.S)

    for line in clean.splitlines():
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        m = re.match(r'^([A-Za-z0-9_\.]+)\s*=\s*(.+)$', line)
        if m:
            k, v = m.group(1).upper(), m.group(2).strip()
            v = re.split(r'//', v)[0].strip()   # strip inline comment
            v = v.strip('"').strip("'").replace('<', '').replace('>', '')
            label[k] = v

    # Find the image filename referenced by ^IMAGE
    m = re.search(r'\^IMAGE\s*=\s*"?([A-Za-z0-9_\-\.]+\.(?:IMG|JP2))"?',
                  txt, flags=re.I)
    label['__IMAGE_FILE__'] = m.group(1) if m else None

    return label


def _parse_num(value, default=0.0):
    """Parse a string that may contain units, returning a float."""
    if value is None:
        return default
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return default


def _numpy_dtype(bits: int, sample_type: str) -> np.dtype:
    """Build a numpy dtype from PDS3 bit-depth and sample_type strings."""
    st      = (sample_type or 'LSB_INTEGER').upper()
    endian  = '>' if any(t in st for t in ('MSB', 'SUN', 'NETWORK')) else '<'
    if 'UNSIGNED' in st:
        base = {8: 'u1', 16: 'u2', 32: 'u4'}[bits]
    elif 'REAL' in st or 'FLOAT' in st:
        base = {32: 'f4', 64: 'f8'}[bits]
    else:
        base = {8: 'i1', 16: 'i2', 32: 'i4'}[bits]
    return np.dtype(endian + base)


def _read_img(img_path: Path, label: dict) -> np.ndarray:
    """Read a PDS3 IMG binary file, row by row, with a progress bar."""
    lines   = int(_parse_num(label.get('LINES'), 720))
    samples = int(_parse_num(label.get('LINE_SAMPLES') or label.get('SAMPLES'), 1440))
    bits    = int(_parse_num(label.get('SAMPLE_BITS'), 16))
    stype   = str(label.get('SAMPLE_TYPE') or 'LSB_INTEGER')
    rec_b   = int(_parse_num(label.get('RECORD_BYTES'), 0))
    lbl_rec = int(_parse_num(label.get('LABEL_RECORDS'), 0))
    lprefix = int(_parse_num(label.get('LINE_PREFIX_BYTES'), 0))
    lsuffix = int(_parse_num(label.get('LINE_SUFFIX_BYTES'), 0))

    dtype      = _numpy_dtype(bits, stype)
    byte_off   = lbl_rec * rec_b if (lbl_rec and rec_b) else 0
    row_bytes  = lprefix + samples * dtype.itemsize + lsuffix
    arr        = np.empty((lines, samples), dtype=np.float32)

    with open(img_path, 'rb') as fh, \
         tqdm(total=lines, unit='row', desc=f'Loading {img_path.name}') as pbar:
        fh.seek(byte_off)
        for i in range(lines):
            buf = fh.read(row_bytes)
            if len(buf) != row_bytes:
                raise RuntimeError(f'Unexpected EOF at row {i}')
            row = np.frombuffer(buf[lprefix:lprefix + samples * dtype.itemsize],
                                dtype=dtype, count=samples).astype(np.float32)
            arr[i] = row
            pbar.update(1)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_ldem(search_dirs=None):
    """
    Auto-detect and load the LOLA LDEM elevation grid.

    Searches the directories listed in *search_dirs* (default: current
    directory and a few common notebook locations) for LDEM_*.LBL files,
    in order of decreasing resolution.

    Returns
    -------
    elev_m   : (H, W) float32 array — elevation in metres above the mean sphere
    pixel_m  : pixel size in metres
    map_res  : resolution in pixels per degree
    label    : raw PDS3 metadata dict (rarely needed directly)
    """
    if search_dirs is None:
        # Search standard local locations only — no hard-coded environment-
        # specific cloud paths (/mnt/project, /home/claude) that silently
        # fail in any non-cloud environment.
        search_dirs = [
            Path('.'),                   # current working directory
            Path('data'),                # data/ subfolder (common layout)
            Path('..') / 'data',         # ../data/ (notebook one level up)
            Path.home() / 'data',        # ~/data/
            Path.home() / 'lunar_data',  # ~/lunar_data/
        ]

    # Prefer higher-resolution products (listed from finest to coarsest)
    patterns = [
        'LDEM_512*.LBL', 'LDEM_256*.LBL', 'LDEM_128*.LBL',
        'LDEM_64*.LBL',  'LDEM_16*.LBL',  'LDEM_4*.LBL',
    ]

    searched = []   # collect all examined paths for a helpful error message
    for d in search_dirs:
        if not d.exists():
            continue
        print(f'Searching: {d.resolve()}')
        for pat in patterns:
            for lbl_path in sorted(d.glob(pat)):
                searched.append(str(lbl_path))
                label = _parse_pds3_label(lbl_path)
                ref   = label.get('__IMAGE_FILE__')
                if not ref:
                    print(f'  [skip] {lbl_path.name}: no ^IMAGE reference found in label')
                    continue
                img_path = lbl_path.parent / ref
                if not img_path.exists():
                    print(f'  [skip] {lbl_path.name}: image file {ref!r} not found '
                          f'in {lbl_path.parent}')
                    continue

                print(f'Loading: {lbl_path.name}  ({img_path.suffix.upper()})')

                # Read raster data
                if img_path.suffix.upper() == '.JP2':
                    try:
                        import imageio.v3 as iio
                        arr = iio.imread(img_path)
                        arr = arr[..., 0] if arr.ndim == 3 else arr
                        arr = arr.astype(np.float32)
                    except ImportError:
                        raise ImportError('imageio is required for JP2 files.  '
                                          'Install with: pip install imageio[pillow]')
                else:
                    arr = _read_img(img_path, label)

                # Apply scale/offset and subtract mean lunar radius → elevation
                scale  = _parse_num(label.get('SCALING_FACTOR'), 1.0)
                offset = _parse_num(label.get('OFFSET'), 0.0)
                elev_m = (arr * scale + offset) - 1_737_400.0

                map_res = _parse_num(label.get('MAP_RESOLUTION'), 4.0)  # pix/deg
                pixel_m = (1.0 / map_res) * np.pi * R_MOON / 180.0

                print(f'  Grid : {elev_m.shape[0]} × {elev_m.shape[1]} pixels')
                print(f'  Pixel: {pixel_m:.1f} m  ({map_res} pix/deg)')

                return elev_m, pixel_m, map_res, label

    hint = (f'\nSearched {len(searched)} label file(s): {searched}' if searched
            else '\nNo LDEM_*.LBL files found in any search directory.')
    raise FileNotFoundError(
        'No valid LDEM file found.  Download LDEM_*.LBL + LDEM_*.IMG (or .JP2) '
        'from https://pds.lroc.asu.edu/data/LRO-L-LOLA-4-GDR-V1.0/ and place '
        'them in the data/ subfolder next to this notebook.' + hint
    )


def latlon_to_pixel(lat_deg, lon_deg, n_rows, n_cols, map_res):
    """
    Convert geographic coordinates to pixel indices in the LDEM grid.

    The LDEM uses a simple cylindrical (equirectangular) projection:
      • Row 0 = 90 °N, rows increase southward.
      • Column 0 = 0 °E, columns increase eastward (0–360°).
      • Pixel centres are at half-integer offsets.

    Returns
    -------
    row, col      : integer pixel indices (clamped to valid range)
    actual_lat    : latitude of the pixel centre (degrees)
    actual_lon    : longitude of the pixel centre (degrees)
    """
    pix_deg = 1.0 / map_res

    row = int(round((90.0 - lat_deg) / pix_deg - 0.5))
    col = int(round(lon_deg          / pix_deg - 0.5))

    row = max(0, min(n_rows - 1, row))
    col = max(0, min(n_cols - 1, col))

    actual_lat = 90.0 - (row + 0.5) * pix_deg
    actual_lon =        (col + 0.5) * pix_deg

    return row, col, actual_lat, actual_lon


def compute_slope_aspect(elev_m, row, col, pixel_m):
    """
    Compute slope and aspect at a single pixel using central differences.

    Returns
    -------
    slope  : slope angle (radians)
    aspect : clockwise from north (radians); 0 = north, π/2 = east
    """
    H, W = elev_m.shape

    dz_dy = (elev_m[min(row + 1, H - 1), col] -
             elev_m[max(row - 1, 0),     col]) / (2.0 * pixel_m)
    dz_dx = (elev_m[row, min(col + 1, W - 1)] -
             elev_m[row, max(col - 1, 0)    ]) / (2.0 * pixel_m)

    slope  = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect = np.arctan2(dz_dx, -dz_dy)   # -dz_dy: rows increase southward
    if aspect < 0:
        aspect += 2.0 * np.pi

    return slope, aspect


def extract_point(lat_deg, lon_deg, elev_m, pixel_m, map_res):
    """
    One-stop helper: snap (lat, lon) to the DEM and return all geometry.

    Returns
    -------
    row, col    : pixel indices
    actual_lat  : actual latitude of the snapped pixel centre (degrees)
    actual_lon  : actual longitude (degrees)
    elevation   : elevation at the pixel (metres)
    slope       : slope angle (radians)
    aspect      : aspect angle, clockwise from north (radians)
    """
    H, W = elev_m.shape
    row, col, actual_lat, actual_lon = latlon_to_pixel(
        lat_deg, lon_deg, H, W, map_res
    )
    elevation        = float(elev_m[row, col])
    slope, aspect    = compute_slope_aspect(elev_m, row, col, pixel_m)

    return row, col, actual_lat, actual_lon, elevation, slope, aspect
