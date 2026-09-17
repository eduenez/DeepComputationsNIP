#!/usr/bin/env python3
"""
Render the Newton fractal for p(z) = z^3 - 1 (the example named in
ApprDeepComps.tex, sec. exa:DEQ-Newton) in two matched versions:

  * a high-resolution COLOR version (one hue per root, shaded by
    convergence speed), and
  * a GREYSCALE version suitable for non-colour print: the three basins
    occupy three well-separated grey bands, shaded within each band by
    convergence speed so the fractal boundary filigree stays visible.

No third-party image library is required: PNGs are written with a small
pure-stdlib encoder (numpy + zlib).  Same iteration data feeds both images,
so the colour and grey figures are faithful counterparts of each other.
"""

import struct
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 2000                      # pixels per side (square)
XMIN, XMAX = -2.0, 2.0
YMIN, YMAX = -2.0, 2.0
MAX_ITER = 80
TOL = 1e-6
OUT_DIR = Path("~/repos/DeepComputationsNIP/images").expanduser()

# p(z) = z^3 - 1
ROOTS = np.array([1.0 + 0j,
                  np.exp(2j * np.pi / 3),
                  np.exp(4j * np.pi / 3)])

# ---------------------------------------------------------------------------
# Newton iteration, recording (basin index, iterations-to-converge)
# ---------------------------------------------------------------------------
x = np.linspace(XMIN, XMAX, N)
y = np.linspace(YMAX, YMIN, N)          # top row = +imaginary
Z = x[None, :] + 1j * y[:, None]

root_idx = np.full((N, N), -1, dtype=np.int8)
iters = np.zeros((N, N), dtype=np.float64)
done = np.zeros((N, N), dtype=bool)

with np.errstate(divide="ignore", invalid="ignore"):
    for it in range(MAX_ITER):
        dz = 3.0 * Z * Z
        step = (Z ** 3 - 1.0) / dz
        step[~np.isfinite(step)] = 0.0      # critical point z=0 etc.
        Z = Z - step
        for k, r in enumerate(ROOTS):
            hit = (~done) & (np.abs(Z - r) < TOL)
            root_idx[hit] = k
            iters[hit] = it
            done[hit] = True
        if done.all():
            break

conv = done
# Robust normalisation of convergence speed (avoid a few outliers flattening it)
p99 = np.percentile(iters[conv], 99) if conv.any() else 1.0
shade = np.clip(iters / max(p99, 1.0), 0.0, 1.0)   # 0 = fast (interior), 1 = slow (boundary)

# ---------------------------------------------------------------------------
# Colour version: hue per root, darkened toward the boundary
# ---------------------------------------------------------------------------
palette = np.array([
    [0.84, 0.25, 0.22],   # root 1  (warm red)
    [0.20, 0.56, 0.30],   # root 2  (green)
    [0.23, 0.38, 0.82],   # root 3  (blue)
])
bright = (1.0 - 0.60 * shade)[..., None]
rgb = np.zeros((N, N, 3), dtype=np.float64)
for k in range(3):
    m = root_idx == k
    rgb[m] = palette[k] * bright[m]
rgb[~conv] = 0.0
rgb_u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

# ---------------------------------------------------------------------------
# Greyscale version: three separated grey bands, shaded within each band
#   band centres 0.88 / 0.60 / 0.32, minus up to 0.22 for slow convergence
#   -> bands [0.66,0.88], [0.38,0.60], [0.10,0.32] never overlap
# ---------------------------------------------------------------------------
band = np.array([0.88, 0.60, 0.32])
grey = np.zeros((N, N), dtype=np.float64)
for k in range(3):
    m = root_idx == k
    grey[m] = band[k] - 0.22 * shade[m]
grey[~conv] = 0.0
grey_u8 = (np.clip(grey, 0, 1) * 255 + 0.5).astype(np.uint8)

# ---------------------------------------------------------------------------
# Minimal PNG writer (stdlib only)
# ---------------------------------------------------------------------------
def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

def write_png(path, arr):
    if arr.ndim == 3:                      # RGB
        h, w, _ = arr.shape
        color_type, stride = 2, w * 3
    else:                                   # grey
        h, w = arr.shape
        color_type, stride = 0, w
    raw = arr.reshape(h, -1).astype(np.uint8)
    # prepend filter byte 0 to each scanline
    lines = np.hstack([np.zeros((h, 1), np.uint8), raw]).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", zlib.compress(lines, 9))
           + _chunk(b"IEND", b""))
    Path(path).write_bytes(png)

OUT_DIR.mkdir(parents=True, exist_ok=True)
color_path = OUT_DIR / f"newton_z3m1_color_{N}.png"
grey_path = OUT_DIR / f"newton_z3m1_grey_{N}.png"
write_png(color_path, rgb_u8)
write_png(grey_path, grey_u8)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
frac_conv = conv.mean()
print(f"resolution        : {N}x{N}")
print(f"converged pixels  : {frac_conv*100:.4f}%")
print(f"iters 99th pct    : {p99:.1f}, max {iters[conv].max():.0f}")
counts = [int((root_idx == k).sum()) for k in range(3)]
print(f"basin pixel counts: {counts}")
# grey-band separation sanity check
for k in range(3):
    m = root_idx == k
    if m.any():
        print(f"  root {k}: grey range "
              f"[{grey[m].min():.3f}, {grey[m].max():.3f}]")
print("wrote:", color_path)
print("wrote:", grey_path)
