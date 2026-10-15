"""Shared helpers for the waterfall / batch-waterfall scripts."""

import os
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from PIL import Image

# Full waterfall PNGs can exceed PIL's default decompression-bomb limit
# (~179 Mpx); disable so write_thumbnail can re-open them.
Image.MAX_IMAGE_PIXELS = None


J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def j2000_ns_to_datetime(t_ns, tz):
    return [
        (J2000_UTC + timedelta(microseconds=int(t) / 1000.0)).astimezone(tz)
        for t in t_ns
    ]


def edges(c):
    c = np.asarray(c, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    d = np.diff(c)
    return np.concatenate([[c[0] - d[0] / 2],
                           c[:-1] + d / 2,
                           [c[-1] + d[-1] / 2]])


def regrid_time_uniform(mag, t_ns):
    """Regrid the last (time) axis of ``mag`` to a uniform cadence.

    Returns ``(mag_regridded, t_uniform)``. Cells with no original sample
    within half a cadence are filled with NaN so they render as the cmap's
    "bad" colour. This makes data gaps occupy multiple cells (visible in
    proportion to gap duration) rather than collapsing to a single pixel.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    if t_ns.size < 2:
        return mag, t_ns
    diffs = np.diff(t_ns)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return mag, t_ns
    cadence = int(np.median(diffs))

    t_uniform = np.arange(int(t_ns[0]), int(t_ns[-1]) + cadence, cadence,
                          dtype=np.int64)
    idx = np.searchsorted(t_ns, t_uniform)
    left = np.clip(idx - 1, 0, len(t_ns) - 1)
    right = np.clip(idx, 0, len(t_ns) - 1)
    left_diff = np.abs(t_uniform - t_ns[left])
    right_diff = np.abs(t_uniform - t_ns[right])
    nearest = np.where(right_diff < left_diff, right, left)
    nearest_diff = np.minimum(left_diff, right_diff)
    mapping = np.where(nearest_diff < cadence / 2, nearest, -1)

    new_shape = mag.shape[:-1] + (t_uniform.size,)
    out = np.full(new_shape, np.nan, dtype=mag.dtype)
    v_mask = mapping >= 0
    src = mapping[v_mask].astype(np.int64)
    out[..., v_mask] = mag[..., src]
    return out, t_uniform


def write_thumbnail(src_png, max_dim=500):
    d, name = os.path.split(src_png)
    thumb = os.path.join(d, f"thumbnail_{name}")
    with Image.open(src_png) as im:
        im.thumbnail((max_dim, max_dim))
        im.save(thumb)
    print(f"wrote {thumb}")


def save_waterfall_png(mag, t_ns, freq_centre, *,
                       input_a, input_b, title_files,
                       tz, tz_name,
                       dpi=150, width=12.0, max_height=60.0,
                       out_path, verbose=True):
    """Render a single waterfall PNG (and its thumbnail).

    ``mag`` is |V|, shape (n_freq, n_time), already selected for this pair.
    ``t_ns`` is int64 ns-since-J2000, shape (n_time,), pre-sync samples dropped.
    ``freq_centre`` is freq centre in MHz, shape (n_freq,).
    """
    mag = np.asarray(mag, dtype=np.float64)
    n_freq, n_time = mag.shape

    f_edges = edges(freq_centre)
    times = j2000_ns_to_datetime(t_ns, tz)
    t_sec = (t_ns - t_ns[0]) * 1e-9
    t_edges_sec = edges(t_sec)

    bad = (mag <= 0) | ~np.isfinite(mag)
    n_empty = int(bad.sum())
    cell_w = np.diff(t_edges_sec)
    typical_w = np.median(cell_w)
    gap_cells = cell_w > 1.5 * typical_w
    if gap_cells.any():
        if verbose:
            print(f"masking {gap_cells.sum()} cells that straddle data gaps "
                  f"(>1.5x median cadence of {typical_w:.2f}s)")
        bad |= gap_cells[np.newaxis, :]
    if n_empty and verbose:
        print(f"masking {n_empty} empty/non-finite pixels")
    mag_ma = np.ma.array(mag, mask=bad)

    finite_pos = mag[~bad]
    cmap = mpl.colormaps["viridis"].with_extremes(bad="black")
    q_edges = np.linspace(0.0, 1.0, cmap.N + 1)
    bounds = np.unique(np.quantile(finite_pos, q_edges))
    if bounds.size < 2:
        bounds = np.array([finite_pos.min(), finite_pos.max() + 1e-12])
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

    left_in = 1.0
    bottom_in = 1.0
    top_in = 0.6
    cbar_pad_in = 0.15
    cbar_w_in = 0.3
    cbar_labels_in = 1.1
    right_in = cbar_pad_in + cbar_w_in + cbar_labels_in

    axes_w_min_in = max(width - left_in - right_in, 0.1)
    axes_w_in = max(axes_w_min_in, n_time / dpi)
    axes_h_need_in = n_freq / dpi
    axes_h_cap_in = max(max_height - top_in - bottom_in, 0.1)
    axes_h_in = min(axes_h_need_in, axes_h_cap_in)

    axes_px_w = axes_w_in * dpi
    axes_px_h = axes_h_in * dpi
    if verbose:
        if axes_px_h < n_freq:
            print(f"WARNING: axes height {axes_px_h:.0f}px < {n_freq} channels "
                  f"(raise --dpi or --max-height to avoid aliasing)")
        else:
            print(f"axes {axes_px_w:.0f}x{axes_px_h:.0f}px covers "
                  f"{n_time} time x {n_freq} freq (>=1px/cell)")

    fig_w = left_in + axes_w_in + right_in
    fig_h = top_in + axes_h_in + bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    try:
        ax = fig.add_axes([left_in / fig_w, bottom_in / fig_h,
                           axes_w_in / fig_w, axes_h_in / fig_h])
        ax.set_facecolor("black")
        # imshow with a 2D float array goes through Agg's fast image renderer
        # (LUT lookup + per-pixel composite) instead of pcolormesh's slow
        # per-quad rasterization. Bad pixels are painted via cmap's "bad" colour.
        pcm = ax.imshow(mag_ma,
                        cmap=cmap, norm=norm,
                        aspect="auto", origin="lower",
                        extent=(float(t_edges_sec[0]), float(t_edges_sec[-1]),
                                float(f_edges[0]), float(f_edges[-1])),
                        interpolation="nearest")
        ax.set_ylabel("Frequency [MHz]")
        cax = fig.add_axes([(left_in + axes_w_in + cbar_pad_in) / fig_w,
                            bottom_in / fig_h,
                            cbar_w_in / fig_w,
                            axes_h_in / fig_h])
        cb = fig.colorbar(pcm, cax=cax, spacing="uniform")
        tick_qs = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        tick_vals = np.quantile(finite_pos, tick_qs)
        cb.set_ticks(tick_vals)
        cb.set_ticklabels([f"{v:.2g}\n(p{int(q*100)})"
                           for v, q in zip(tick_vals, tick_qs)])
        cb.set_label(r"$|V|$ (histogram-equalized)")

        t0_local = times[0]

        def fmt_local(x, _pos):
            return (t0_local + timedelta(seconds=float(x))).strftime("%H:%M:%S")

        ax.xaxis.set_major_formatter(FuncFormatter(fmt_local))

        # Major time tick every wallclock hour, minor every 15 min.
        total_sec = float(t_sec[-1])
        secs_off = (t0_local.minute * 60 + t0_local.second
                    + t0_local.microsecond * 1e-6)
        first_hour = (3600.0 - secs_off) % 3600.0
        hour_offsets = np.arange(first_hour, total_sec + 1, 3600.0)
        q_off = ((t0_local.minute % 15) * 60 + t0_local.second
                 + t0_local.microsecond * 1e-6)
        first_q = (900.0 - q_off) % 900.0
        q_offsets = np.arange(first_q, total_sec + 1, 900.0)
        if hour_offsets.size:
            ax.xaxis.set_major_locator(FixedLocator(hour_offsets))
        if q_offsets.size:
            ax.xaxis.set_minor_locator(FixedLocator(q_offsets))

        # Frequency: major every 100 MHz, minor every 25 MHz.
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(25))

        # Subtle grid for orientation against the dark background.
        # ax.grid(True, which="major", color="white", linewidth=0.5, alpha=0.3)
        # ax.grid(True, which="minor", color="white", linewidth=0.3, alpha=0.15)

        fig.autofmt_xdate()

        date_str = t0_local.strftime("%Y-%m-%d %Z")
        ax.set_xlabel(f"Local time ({tz_name}) — {date_str}")
        if input_a == input_b:
            prod_label = f"Autocorr element {input_a}"
        else:
            prod_label = f"Cross ({input_a},{input_b})"
        ax.set_title(f"{prod_label}  |  {title_files}")

        fig.savefig(out_path, dpi=dpi)
    finally:
        plt.close(fig)
    if verbose:
        print(f"wrote {out_path}")
    write_thumbnail(out_path)
