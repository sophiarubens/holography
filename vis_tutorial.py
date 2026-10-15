"""
CHORD pathfinder visibility data demo

Routines for finding, inspecting, and plotting the N^2 visibility HDF5 files
written by the pathfinder receiver
(`config/chord_pathfinder_recv.j2`).

- **Data**: `/mnt/cs00/data/kotekan_vis_files/{full,subset}/acq_*/vis_*.h5`
- **Format reference**: `docs/sphinx/user/file_formats/n2_vis_hdf5.rst`
  (generic) and `n2_vis_hdf5_pathfinder.rst` (this deployment), in the
  kotekan repository
- **Environment**: run with the kotekan Python env from this repository's root
  so the local plotting helpers (`_vis_render.py`, `plot_vis_waterfall.py`)
  import.

Quick background: the **subset** stream holds the 36 cross-products of
correlator inputs 0–7 (the six connected feeds `A01p1`…`A07p2` plus two
unconnected inputs); the **full** stream holds all 8256 products of 128
inputs. Each file spans ~200 s (20 ERA-aligned ~10 s accumulation bins)
across 6145 frequency channels (300–1500 MHz). Missing (frequency, time)
slots are zero-filled — always apply the `frames_added` mask. Beware also
that acquisitions taken while the second-stage RFI excision flagged all
samples are **fully excised**: every dataset is zero (`frac_lost` = 1)
even though `frames_added` is 1. The inventory below marks those `EMPTY`.
"""

import glob
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import h5py
import hdf5plugin  # registers the bitshuffle filter used by /vis etc.
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,CenteredNorm
import cmasher as cm

# Local helpers shared with the plotting scripts in this directory
# (run the notebook from this repository's root so these resolve).
sys.path.insert(0, os.getcwd())
from _vis_render import edges, j2000_ns_to_datetime, regrid_time_uniform
from plot_vis_waterfall import load_one, load_gain_denom

# DATA_ROOT = "/mnt/cs00/data/kotekan_vis_files"
DATA_ROOT="/Users/sophiarubens/Downloads/research/code/holography/kotekan_vis_files"
LOCAL_TZ = ZoneInfo("America/Vancouver")

np.set_printoptions(precision=4, suppress=True, threshold=20, edgeitems=2)

"""
Each acquisition is a directory `acq_<UTC start>_<ns>/` holding files named
`vis_<abs_file_idx>_<YYYYMMDDT_HHMMSS>_<ns>.h5`, where the timestamp is the
UTC time of the earliest FPGA tick in the file. We can therefore build the
inventory from file names alone (no need to open every file); each file
covers ~200 s of data.
"""

FNAME_RE = re.compile(r"vis_(\d+)_(\d{8})T_(\d{6})_(\d{9})\.h5$")
FILE_SPAN = timedelta(seconds=200)  # 20 bins x ~9.98 s


def file_start_utc(path):
    """UTC start time encoded in a vis_*.h5 file name (None if no match)."""
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
    return dt.replace(tzinfo=timezone.utc) + timedelta(
        microseconds=int(m.group(4)) / 1000.0)


def acq_has_data(acq_dir, n_sample=3):
    """True if any sampled file holds accumulated data (frac_lost < 1).

    Acquisitions taken while the RFI gate was closed (or the F-engine was
    down) are fully excised: every dataset exists but holds only zeros and
    frac_lost stays at 1.0. Sampling the first/middle/last file catches
    that cheaply (frac_lost exists in every file vintage).
    """
    files = sorted(glob.glob(os.path.join(acq_dir, "vis_*.h5")))
    if not files:
        return False
    for i in sorted({0, len(files) // 2, len(files) - 1})[:n_sample]:
        try:
            with h5py.File(files[i], "r") as fh:
                if (fh["frac_lost"][:] < 1).any():
                    return True
        except (OSError, KeyError):
            pass
    return False


def list_acquisitions(stream):
    """[(acq_dir, n_files, start_utc, end_utc, has_data)] for one stream."""
    rows = []
    for acq in sorted(glob.glob(os.path.join(DATA_ROOT, stream, "acq_*"))):
        starts = [t for t in (file_start_utc(p) for p in
                              glob.glob(os.path.join(acq, "vis_*.h5"))) if t]
        if starts:
            rows.append((acq, len(starts), min(starts),
                         max(starts) + FILE_SPAN, acq_has_data(acq)))
    return rows


# inventory = {}
# for stream in ("full", "subset"):
#     inventory[stream] = list_acquisitions(stream)
#     print(f"=== {stream} ===")
#     print(f"{'acquisition':<34}{'files':>6}   {'start (UTC)':<21}"
#           f"{'end (UTC)':<21}{'duration':<11}data?")
#     for acq, n, t0, t1, has in inventory[stream]:
#         print(f"{os.path.basename(acq):<34}{n:>6}   {t0:%Y-%m-%d %H:%M:%S}   "
#               f"{t1:%Y-%m-%d %H:%M:%S}   {str(t1 - t0).split('.')[0]:<11}"
#               f"{'yes' if has else 'EMPTY'}")
#     print()
## ^commented that out for now because it is just probing the subdirectory structure associated with how the data is stored on Fir and I've only exported a single acquisition

"""
## 2. Inside a subset file

Load one file from the middle of the most recent subset acquisition **that
contains data** (see the `data?` column above — recent acquisitions can be
fully RFI-excised) and list
everything in it: the root attributes, then every dataset with its shape,
type, a short value preview, and a one-line description. (See the format
reference for the full semantics. In this pipeline `gain`, `flags`, and
`radiometer_chi2` hold sentinels — no stage populates them — while the
subset stream's `eval`/`evec`/`erms` are real output of the upstream
`EigenN2Iter` eigensolver.)
"""

# latest_acq = next(acq for acq, *_, has in reversed(inventory["subset"]) if has)
latest_acq=DATA_ROOT+"/acq_20261006_233255_029438780/" # the only one I've downloaded locally
subset_files = sorted(glob.glob(os.path.join(latest_acq, "vis_*.h5")))
print("len(subset_files)=",len(subset_files))
demo_file = subset_files[len(subset_files) // 2]
# print(f"acquisition: {os.path.basename(latest_acq)}  ({len(subset_files)} files)")
# print(f"demo file:   {os.path.basename(demo_file)}")

ATTR_NOTES = {
    "version": "file format version",
    "file_mode": "dataset layout flavour (CHORD or legacy CHIME)",
    "abs_file_idx": "absolute file index = bin_abs_index // num_file_t",
    "num_file_t": "time bins per file",
    "num_file_f": "science frequency channels in the file",
    "num_telescope_f": "total FPGA channels of the telescope",
    "num_elements": "correlator inputs in this stream",
    "num_prod": "correlation products = Ne(Ne+1)/2",
    "num_ev": "eigenpairs stored",
    "n2_layout": "visibility matrix packing",
    "input_order": "element ordering (see docs: Element Ordering)",
    "num_dishes": "input slots in the telescope description",
    "instrument_name": "telescope class",
    "nyquist_zone": "sampling Nyquist zone",
    "gps_time_enabled": "FPGA clock is GPS-disciplined",
    "frame0_unix_ns": "instrument time of FPGA tick 0 (ns, UNIX epoch)",
    "fpga_seq_length_ns": "duration of one FPGA tick (ns)",
    "origin_itrs_lat_deg": "telescope origin ITRS latitude (deg)",
    "origin_itrs_lon_deg": "telescope origin ITRS longitude (deg)",
    "itrs_lat_deg": "telescope origin ITRS latitude (deg)",
    "itrs_lon_deg": "telescope origin ITRS longitude (deg)",
    "grid_orientation": "3x3 rotation, topocentric -> dish grid frame",
    "dish_orientation": "3x3 rotation, topocentric -> dish pointing frame",
    "dish_coelev_deg": "commanded dish co-elevation (deg)",
    "EOP_t_inst_ns": "EOP table epochs, instrument time (ns)",
    "EOP_t_ut1_ns": "EOP table epochs, UT1 (ns since J2000)",
    "EOP_delta_UT1_inst": "UT1 - instrument time (s)",
    "EOP_ERA_deg": "Earth Rotation Angle at the epochs (deg)",
    "EOP_xp_as": "polar motion x_p (arcsec)",
    "EOP_yp_as": "polar motion y_p (arcsec)",
    "digital_gains_source_file": "gains archive copied into /digital_gains",
}


def short(v, width=44):
    v = np.asarray(v)
    if v.ndim == 0:
        s = repr(v.item())
    else:
        s = np.array2string(v.ravel()[:3], precision=4)
        if v.size > 3:
            s = s[:-1] + " ...]" + f" ({v.shape})"
    return s if len(s) <= width else s[:width - 3] + "..."


f = h5py.File(demo_file, "r")
# print("== root attributes ==")
# for k in sorted(f.attrs):
#     print(f"  {k:<28}{short(f.attrs[k]):<46}  {ATTR_NOTES.get(k, '')}")

DS_NOTES = {
    "vis": "visibilities (freq, prod, time), mean per valid FPGA sample",
    "vis_weight": "inverse-variance weight per visibility (0 = missing)",
    "eval": "eigenvalues from upstream EigenN2Iter (subset stream)",
    "evec": "eigenvectors, packed (eigenpair, element)",
    "erms": "RMS residual after removing the eigen model",
    "gain": "applied gain per input (sentinel -1+0j; real gains in /digital_gains)",
    "flags": "per-input flags from upstream flagging (unpopulated: 0)",
    "radiometer_chi2": "radiometer test per pol pair XX,XY,YY (sentinel -1)",
    "frames_added": "1 where a frame was received -- THE validity mask",
    "valid_fpga_count": "FPGA ticks accumulated after excision",
    "rfi_fpga_count": "ticks excised by the RFI mask (may overlap PL)",
    "rfi_only_fpga_count": "ticks excised by RFI only",
    "pl_fpga_count": "ticks lost to packet loss",
    "frac_lost": "1 - valid/frame_length (1.0 = nothing received)",
    "frac_rfi": "rfi_fpga_count / frame_length",
    "frac_rfi_only": "rfi_only_fpga_count / frame_length",
    "frac_pl": "pl_fpga_count / frame_length",
    "fpga_start_tick": "first FPGA tick of each bin",
    "frame_length_fpga_ticks": "nominal bin length in ticks (~9.98 s)",
    "bin_abs_index": "absolute ERA-bin counter (1/24 deg per bin)",
    "time_center_t_inst_ns": "instrument time at data midpoint (ns, UNIX)",
    "time_center_ut1_ns": "UT1 at data midpoint (ns since J2000)",
    "bin_t_inst_ns": "instrument time at nominal bin centre",
    "bin_ut1_ns": "UT1 at nominal bin centre",
    "bin_delta_ut1_inst": "UT1 - instrument time (s)",
    "bin_ERA_deg": "Earth Rotation Angle at bin centre",
    "bin_xp_as": "polar motion x_p (arcsec)",
    "bin_yp_as": "polar motion y_p (arcsec)",
    "bin_start_ERA_deg": "ERA at bin start (exact 1/24 deg multiples)",
    "bin_end_ERA_deg": "ERA at bin end",
    "bin_start_ERAL_deg": "local ERA (ERA + longitude + s') at bin start",
    "bin_end_ERAL_deg": "local ERA at bin end",
    "rfi_frame_excision_enabled": "2nd-stage RFI excision active?",
    "rfi_frame_excision_num": "number of active SK thresholds",
    "rfi_frame_excision_threshold": "SK thresholds (sigma)",
    "rfi_frame_excision_fraction": "trigger fraction per threshold",
    "config_json": "JSON config snapshots (kotekan + FPGA) from ConfigTracker",
    "index_map/freq": "channel centre/width (MHz) per frequency index",
    "index_map/prod": "(input_a, input_b) per product index, upper triangle",
    "index_map/grid_x_idx": "dish-grid column per input",
    "index_map/grid_y_idx": "dish-grid row per input",
    "index_map/dish_positions_in_grid_coords": "dish positions (m), grid frame",
    "index_map/feed_pos_disp_m": "feed displacement from grid position (m)",
    "index_map/coelev_disp_deg": "per-input co-elevation offset (deg)",
    "index_map/type": "input type: -1 Fake, 0 ArrayDish, 1 RFIDish",
    "index_map/label": "human-readable input labels",
    "digital_gains/gain_coeff": "F-engine gain per (update, channel, input)",
    "digital_gains/gain_exp": "per-input binary exponent (gain * 2^exp)",
    "digital_gains/compute_time": "UNIX time gains were computed",
    "digital_gains/update_id": "gain update identifier",
    "digital_gains/index_map/freq": "full-band channel centres/widths (MHz)",
    "digital_gains/index_map/input": "channel id + correlator input serial",
    "digital_gains/index_map/update_time": "UNIX time of each update",
}


def preview(ds, width=38):
    """A tiny, cheap peek at a dataset's leading values."""
    if ds.size == 0:
        return "(empty)"
    sel = tuple(slice(0, min(3, n)) for n in ds.shape)
    vals = np.asarray(ds[sel]).ravel()
    if vals.dtype == object:  # variable-length strings (labels, JSON, ...)
        items = [v.decode() if isinstance(v, bytes) else str(v) for v in vals[:2]]
        s = " | ".join(i[:30] for i in items)
    else:
        s = np.array2string(vals[:3], precision=3)
    return s if len(s) <= width else s[:width - 3] + "..."


# print("== datasets ==")
# print(f"  {'name':<40}{'shape':<18}{'dtype':<12}"
#       f"{'preview':<40}description")


def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        dt = str(obj.dtype)
        if len(dt) > 10:
            dt = dt[:9] + "+"
        print(f"  {name:<40}{str(obj.shape):<18}{dt:<12}"
              f"{preview(obj):<40}{DS_NOTES.get(name, '')}")


# f.visititems(show)

"""
## 3. Visibility matrix at a single time

The `vis` dataset packs the upper triangle of the element-by-element
visibility matrix along its `product` axis; `index_map/prod` gives the
(input_a, input_b) pair of each product, and the input labels come from
`index_map/label` (in this deployment the first `num_elements` entries label
the subset's elements 0–7).

Below: for the best-covered time bin in the demo file, the band-averaged
amplitude ⟨|V|⟩ of every product, and the phase at a single frequency
channel near 600 MHz. The lower triangle is not stored
(V(b,a) = V(a,b)*), so it renders blank.
"""

prod = f["index_map/prod"][:]
inputlabelmap=f["index_map/label"][:] # cf. f["index_map/prod"][:]
n_el = int(f.attrs["num_elements"])
labels = [v.decode() if isinstance(v, bytes) else str(v)
          for v in f["index_map/label"][:n_el]]
freq_centre = f["index_map/freq"][:]["centre"]

valid = f["frac_lost"][:] < 1                       # data actually accumulated
t_idx = int(np.argmax(valid.sum(axis=0)))           # best-covered time bin
ch_ok = valid[:, t_idx]
t_unix = f["time_center_t_inst_ns"][t_idx] * 1e-9
t_str = datetime.fromtimestamp(t_unix, tz=timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC")

vis_t = f["vis"][:, :, t_idx]                       # (freq, prod)
amp = np.abs(vis_t[ch_ok]).mean(axis=0)             # band-averaged |V|
ch600 = int(np.flatnonzero(ch_ok)[np.argmin(
    np.abs(freq_centre[ch_ok] - 600.0))])           # good channel near 600 MHz
phase = np.degrees(np.angle(vis_t[ch600]))

A = np.full((n_el, n_el), np.nan)
P = np.full((n_el, n_el), np.nan)
A[prod["input_a"], prod["input_b"]] = amp
P[prod["input_a"], prod["input_b"]] = phase

A[~(A > 0)] = np.nan                                # keep LogNorm happy
pos = amp[amp > 0]

fig, axes = plt.subplots(1, 2, figsize=(20,8))
im0 = axes[0].imshow(A, norm=mcolors.LogNorm(vmin=pos.min(), vmax=pos.max()),
                     cmap="viridis")
axes[0].set_title(r"$\langle|V|\rangle$ over "
                  f"{int(ch_ok.sum())} channels")
im1 = axes[1].imshow(P, vmin=-180, vmax=180, cmap="twilight_shifted")
axes[1].set_title(f"phase [deg] at {freq_centre[ch600]:.1f} MHz")
for ax, im in ((axes[0], im0), (axes[1], im1)):
    ax.set_xticks(range(n_el), labels, rotation=45, ha="right")
    ax.set_yticks(range(n_el), labels)
    ax.set_xlabel("input_b")
    ax.set_ylabel("input_a")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"{os.path.basename(demo_file)} — time bin {t_idx} ({t_str})")
fig.tight_layout()
plt.savefig("vis_matrix_single_time.png")
"""
## 4. Waterfall over the whole acquisition

Read the chosen product from **all** subset files of the most recent
acquisition and render a frequency-vs-time waterfall, the way
`plot_vis_waterfall.py` does (per-element digital-gain division, uniform
time regridding so gaps show as black, histogram-equalized colour scale) —
but frequency-downsampled so it renders inline instead of producing the
scripts' full-resolution multi-megapixel PNGs. Reading every file of a long
acquisition over network storage takes a few minutes; set `MAX_FILES` for a
quicker look. For full resolution use the CLI script, e.g.:

```sh
/opt/kotekan_env/bin/python plot_vis_waterfall.py --element 1 \
    /mnt/cs00/data/kotekan_vis_files/subset/acq_*/vis_*.h5
```
"""

f.close()
ELEMENT = 1      # autocorrelation to plot (plot_vis_waterfall.py's default)
# MAX_FILES = 180  # e.g. 100 to only read the last 100 files (Kendrick's notation) or first 100 files (my notation)
MAX_FILES=None

wf_files = subset_files if MAX_FILES is None else subset_files[:MAX_FILES] # subset_files[-MAX_FILES:] (Kendrick's notation is commented out)
with h5py.File(wf_files[0], "r") as f0:
    freq_centre = f0["index_map/freq"][:]["centre"]

gain_denom = load_gain_denom(wf_files, ELEMENT, ELEMENT, freq_centre)
with ThreadPoolExecutor(max_workers=8) as ex:       # parallel HDF5 reads
    loaded = list(ex.map(
        lambda p: load_one(p, ELEMENT, ELEMENT, gain_denom=gain_denom),
        wf_files))
loaded.sort(key=lambda x: int(x[1][x[1] > 0][0]) if (x[1] > 0).any() else 0)

vis = np.concatenate([v for v, t, fr in loaded], axis=1)
t_ns = np.concatenate([t for v, t, fr in loaded])
keep = t_ns > 0                       # drop never-filled time slots
vis, t_ns = vis[:, keep], t_ns[keep]

mag = np.abs(vis).astype(np.float64)
mag[mag == 0] = np.nan                # missing (freq, time) cells
mag, t_ns = regrid_time_uniform(mag, t_ns)  # uniform cadence; gaps -> NaN
print(f"{len(wf_files)} files -> {mag.shape[0]} channels x {mag.shape[1]} time samples")

B = 16                                # frequency downsample factor
nf = (mag.shape[0] // B) * B
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    mag_lo = np.nanmean(mag[:nf].reshape(nf // B, B, -1), axis=1)
freq_lo = freq_centre[:nf].reshape(-1, B).mean(axis=1)
print(f"downsampled to {mag_lo.shape[0]} channels x {mag_lo.shape[1]} samples")

finite = mag_lo[np.isfinite(mag_lo) & (mag_lo > 0)]
cmap = mpl.colormaps["viridis"].with_extremes(bad="black")
bounds = np.unique(np.quantile(finite, np.linspace(0, 1, cmap.N + 1)))
norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

t_hr = (t_ns - t_ns[0]) * 1e-9 / 3600.0
te, fe = edges(t_hr), edges(freq_lo)
t0_local = j2000_ns_to_datetime(t_ns[:1], LOCAL_TZ)[0]


np.savetxt("t_ns.txt",t_ns)
np.savetxt("freq_lo.txt",freq_lo)

fig, ax = plt.subplots(figsize=(12, 5))
pcm = ax.imshow(np.ma.masked_invalid(mag_lo), cmap=cmap, norm=norm,
                aspect="auto", origin="lower", interpolation="nearest",
                extent=(te[0], te[-1], fe[0], fe[-1]))
ax.set_facecolor("black")
ax.set_xlabel(f"hours since {t0_local:%Y-%m-%d %H:%M:%S %Z}")
ax.set_ylabel("frequency [MHz]")
ax.set_title(f"|V| autocorr element {ELEMENT} - "
             f"{os.path.basename(latest_acq)} (freq x{B} downsampled)")
cb = fig.colorbar(pcm, ax=ax)
qs = [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]
tv = np.quantile(finite, qs)
cb.set_ticks(tv)
cb.set_ticklabels([f"{v:.3g} (p{int(q * 100)})" for v, q in zip(tv, qs)])
cb.set_label("|V| (histogram-equalized)")
plt.savefig("waterfall_over_whole_acq.png")

# f["vis"][:, :, t_idx]                       # (freq, prod)
idx_a=12 # indices in the index_map to pick antenna polarizations !!as of Oct 6th acq, this goes up to 48!! [2-3 is A02X-A03X] 
idx_b=15
Ne=len(inputlabelmap)
input_a=inputlabelmap[idx_a].decode() # which inputs these actually correspond to
input_b=inputlabelmap[idx_b].decode()
print("baseline fo the coherency calculation: {}-{}".format(input_a,input_b))
def find_prod(prod, a, b): # kotekan stores only a <= b (upper triangle)
    conj = a > b
    lo, hi = (b, a) if conj else (a, b)
    hits = np.flatnonzero((prod["input_a"] == lo) & (prod["input_b"] == hi))
    if hits.size != 1:
        raise ValueError(f"pair ({a},{b}) found {hits.size} times in prod")
    return hits[0], conj

idx_aa, _       = find_prod(prod, idx_a, idx_a)
idx_bb, _       = find_prod(prod, idx_b, idx_b)
idx_ab, conj_ab = find_prod(prod, idx_a, idx_b)

recalc=True
if recalc:
  ############################################ need to read in the whole observation, not just one acquisition! the same format as was used for the other waterfall will be helpful here!
  wf_files = subset_files if MAX_FILES is None else subset_files[:MAX_FILES] # subset_files[-MAX_FILES:]
  with h5py.File(wf_files[0], "r") as f0:
      freq_centre = f0["index_map/freq"][:]["centre"]
  gain_denom = load_gain_denom(wf_files, idx_a, idx_a, freq_centre)
  with ThreadPoolExecutor(max_workers=8) as ex:       # parallel HDF5 reads
      loaded = list(ex.map(
          lambda p: load_one(p, idx_a, idx_a, gain_denom=gain_denom),
          wf_files))
  loaded.sort(key=lambda x: int(x[1][x[1] > 0][0]) if (x[1] > 0).any() else 0)
  Vaa = np.concatenate([v for v, t, fr in loaded], axis=1)



  wf_files = subset_files if MAX_FILES is None else subset_files[:MAX_FILES] # subset_files[-MAX_FILES:]
  with h5py.File(wf_files[0], "r") as f0:
      freq_centre = f0["index_map/freq"][:]["centre"]
  gain_denom = load_gain_denom(wf_files, idx_a, idx_b, freq_centre)
  with ThreadPoolExecutor(max_workers=8) as ex:       # parallel HDF5 reads
      loaded = list(ex.map(
          lambda p: load_one(p, idx_a, idx_b, gain_denom=gain_denom),
          wf_files))
  loaded.sort(key=lambda x: int(x[1][x[1] > 0][0]) if (x[1] > 0).any() else 0)
  Vab = np.concatenate([v for v, t, fr in loaded], axis=1)



  wf_files = subset_files if MAX_FILES is None else subset_files[:MAX_FILES] # subset_files[-MAX_FILES:]
  with h5py.File(wf_files[0], "r") as f0:
      freq_centre = f0["index_map/freq"][:]["centre"]
  gain_denom = load_gain_denom(wf_files, idx_b, idx_b, freq_centre)
  with ThreadPoolExecutor(max_workers=8) as ex:       # parallel HDF5 reads
      loaded = list(ex.map(
          lambda p: load_one(p, idx_b, idx_b, gain_denom=gain_denom),
          wf_files))
  loaded.sort(key=lambda x: int(x[1][x[1] > 0][0]) if (x[1] > 0).any() else 0)
  Vbb = np.concatenate([v for v, t, fr in loaded], axis=1)

  np.save("V_{}_{}.npy".format(input_a,input_a),Vaa)
  np.save("V_{}_{}.npy".format(input_a,input_b),Vab)
  np.save("V_{}_{}.npy".format(input_b,input_b),Vbb)
  ############################################

  coherency_only=Vab.real/np.sqrt(Vaa*Vbb)
  print("coherency_only.shape =",coherency_only.shape)
  assert np.nansum(coherency_only.imag)<1e-12, "coherency_only should be real by construction"
  coherency_only=coherency_only.real
  # valid_coherency_only=np.ma.masked_invalid(coherency_only)
  np.save("coherency_{}_{}.npy".format(input_a,input_b),coherency_only)
else:
    coherency_only=np.load("coherency_{}_{}.npy".format(input_a,input_b))

fig, ax = plt.subplots(figsize=(12, 5))
pcm = ax.imshow(coherency_only, cmap="twilight_shifted", 
                # norm=CenteredNorm(vcenter=0.,halfrange=0.5*np.nanpercentile(valid_coherency_only,95)), # dynamic range ok but maybe too small
                norm=CenteredNorm(vcenter=0.,halfrange=0.5*np.nanpercentile(coherency_only,99)),
                # norm=CenteredNorm(vcenter=0.), # dynamic range too high
                aspect="auto", origin="lower", interpolation="nearest",
                extent=(te[0], te[-1], fe[0], fe[-1]))
ax.set_facecolor("white")
ax.set_xlabel(f"hours since {t0_local:%Y-%m-%d %H:%M:%S %Z}")
ax.set_ylabel("frequency [MHz]")
ax.set_title("Coherency of baseline {}-{}".format(input_a,input_b)) # |Vab|/sqrt(Vaa Vbb)$ 
# ax.set_title(f"|V| autocorr element {ELEMENT} - "
            #  f"{os.path.basename(latest_acq)} (freq x{B} downsampled)")
cb = fig.colorbar(pcm, ax=ax)
# qs = [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]
# tv = np.quantile(finite, qs)
# cb.set_ticks(tv)
# cb.set_ticklabels([f"{v:.3g} (p{int(q * 100)})" for v, q in zip(tv, qs)])
# cb.set_label()
plt.savefig("waterfall_coherency_only.png")

"""
## Where to go from here

- **Full stream**: everything above works identically on
  `kotekan_vis_files/full/` - but with 128 elements that includes the RFI antennas, no
  eigenvalues, and ~12 GB per file.
- **Future acquisitioms**: will hopefully give us data from additional dishes!
- **Format details**: `docs/sphinx/user/file_formats/n2_vis_hdf5.rst` and
  `n2_vis_hdf5_pathfinder.rst` (kotekan repository) document every dataset
  and attribute.
"""