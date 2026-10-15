#!/usr/bin/env python
"""Waterfall (frequency vs time) of one product from kotekan vis HDF5 files.

Time axis is plotted in America/Vancouver local time. The HDF5
``time_center_ut1_ns`` field stores nanoseconds since the J2000 epoch
(2000-01-01 12:00:00 TT, which we treat as 2000-01-01 12:00:00 UTC - the
TT-UTC offset of ~69 s is irrelevant at the seconds-resolution shown).
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from _vis_render import regrid_time_uniform, save_waterfall_png


def load_one(path, input_a, input_b, gain_denom=None):
    """Load (vis, t_ns, freq) for the (input_a, input_b) product.

    The HDF5 file stores the upper-triangle, so if (a, b) is missing we try
    (b, a) and conjugate (since V_ba = conj(V_ab)).

    If ``gain_denom`` is given (shape (n_freq, 1) complex), divide vis by it.
    """
    with h5py.File(path, "r") as f:
        prod = f["index_map/prod"][:]
        freq = f["index_map/freq"][:]
        t_ns = f["time_center_ut1_ns"][:]
        match = np.where((prod["input_a"] == input_a) &
                         (prod["input_b"] == input_b))[0]
        conj = False
        if match.size == 0:
            match = np.where((prod["input_a"] == input_b) &
                             (prod["input_b"] == input_a))[0]
            conj = True
        if match.size == 0:
            raise SystemExit(f"{path}: no product for ({input_a}, {input_b})")
        p_idx = int(match[0])
        vis = f["vis"][:, p_idx, :]
        if conj:
            vis = np.conj(vis)
    if gain_denom is not None:
        vis = vis / gain_denom
    return vis, t_ns, freq


def load_gain_denom(paths, input_a, input_b, freq_centre):
    """Find the newest file with digital_gains and return g_a*conj(g_b).

    Returns shape (n_freq, 1) complex, with zeros replaced by 1 (so dividing
    is a no-op for dead channels). Returns None if no file has gains.
    ``gain_exp`` is uniform across elements, so it's a global scale we ignore.
    """
    for p in reversed(paths):
        with h5py.File(p, "r") as f:
            if "digital_gains" not in f:
                continue
            f_dg = f["digital_gains/index_map/freq"][:]["centre"]
            i0 = int(np.searchsorted(f_dg, freq_centre[0]))
            gs = f["digital_gains/gain_coeff"][0,
                                               i0:i0 + freq_centre.size, :]
        g_a = gs[:, input_a]
        g_b = gs[:, input_b]
        denom = (g_a * np.conj(g_b)).astype(np.complex64)
        denom[denom == 0] = 1
        print(f"using digital gains from {os.path.basename(p)}")
        return denom[:, None]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="One or more input vis_*.h5 files")
    ap.add_argument("--element", type=int, default=1,
                    help="Element index (both input_a and input_b unless --input-b given)")
    ap.add_argument("--input-b", type=int, default=None,
                    help="Second input for a cross product")
    ap.add_argument("--tz", default="America/Vancouver")
    ap.add_argument("--out", default=None,
                    help="Output image path (default: <prod_tag>_waterfall.png)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--max-height", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8,
                    help="Threads for parallel HDF5 reads (default: 8)")
    ap.add_argument("--apply-gain", action="store_true",
                    help="Divide vis by per-element digital gains (default: "
                         "off, plots raw |V|)")
    args = ap.parse_args()

    tz = ZoneInfo(args.tz)
    input_a = args.element
    input_b = args.input_b if args.input_b is not None else args.element

    n_workers = max(1, min(args.workers, len(args.files)))
    files_sorted = sorted(args.files)

    gain_denom = None
    if args.apply_gain:
        # Need vis freq centre to slice the gain table; peek at one file.
        with h5py.File(files_sorted[0], "r") as _f:
            freq_centre0 = _f["index_map/freq"][:]["centre"]
        gain_denom = load_gain_denom(files_sorted, input_a, input_b,
                                     freq_centre0)
        if gain_denom is None:
            print("no file had digital_gains; rendering raw |V|")

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        loaded = list(ex.map(
            lambda p: load_one(p, input_a, input_b, gain_denom=gain_denom),
            files_sorted))

    def _first_real_t(t):
        nz = t[t > 0]
        return int(nz[0]) if nz.size else 0

    loaded.sort(key=lambda x: _first_real_t(x[1]))

    freq = loaded[0][2]
    for _, _, fr in loaded[1:]:
        if fr.shape != freq.shape or not np.array_equal(fr["centre"], freq["centre"]):
            raise SystemExit("Frequency axes differ across files; refusing to concat")

    vis = np.concatenate([x[0] for x in loaded], axis=1)
    t_ns = np.concatenate([x[1] for x in loaded])
    print(f"loaded {len(loaded)} file(s), {t_ns.size} time samples total")

    keep_t = t_ns > 0
    if not keep_t.all():
        print(f"dropping {(~keep_t).sum()} samples with t_ns == 0")
        t_ns = t_ns[keep_t]
        vis = vis[:, keep_t]

    mag = np.abs(vis).astype(np.float64)

    pre = mag.shape[1]
    mag, t_ns = regrid_time_uniform(mag, t_ns)
    if mag.shape[1] != pre:
        cadence_s = (int(t_ns[1]) - int(t_ns[0])) * 1e-9 if t_ns.size > 1 else 0
        print(f"regridded time to uniform cadence ({cadence_s:.2f}s): "
              f"{pre} -> {mag.shape[1]} samples (gaps visible)")

    if args.out is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(outdir, exist_ok=True)
        args.out = os.path.join(outdir, f"waterfall_e{input_a}xe{input_b}.png")

    n_files = len(args.files)
    first_base = os.path.basename(args.files[0])
    title_files = first_base if n_files == 1 else f"{n_files} files starting {first_base}"

    save_waterfall_png(
        mag, t_ns, freq["centre"],
        input_a=input_a, input_b=input_b,
        title_files=title_files,
        tz=tz, tz_name=args.tz,
        dpi=args.dpi, width=args.width, max_height=args.max_height,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
