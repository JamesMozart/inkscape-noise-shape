import inkex
import math
import random
import numpy as np
from inkex.paths import Path, Move, ZoneClose


# ============ Vectorized Lightweight Simplex Noise (2D) ============
class SimplexNoise:
    """
    NumPy-based 2D Simplex noise.

    Reproducibility:
      - perm table is built from Python's random.Random(seed) -> deterministic.
      - noise_batch computes the same math per element as the scalar version,
        but in one vectorized call.
      - This class never consumes random state after __init__, so callers can
        batch freely without changing output for a given seed.
    """

    def __init__(self, seed=0):
        rng = random.Random(seed)
        perm_list = list(range(256))
        rng.shuffle(perm_list)
        self.perm = np.asarray(perm_list * 2, dtype=np.int64)

        self.grad3 = np.array([
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1),
        ], dtype=np.float64)

        self.F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - math.sqrt(3.0)) / 6.0

    def noise_batch(self, xin, yin):
        xin = np.asarray(xin, dtype=np.float64)
        yin = np.asarray(yin, dtype=np.float64)
        perm = self.perm
        grad3 = self.grad3
        F2, G2 = self.F2, self.G2

        s = (xin + yin) * F2
        i = np.floor(xin + s).astype(np.int64)
        j = np.floor(yin + s).astype(np.int64)
        t = (i + j) * G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)

        cond = x0 > y0
        i1 = np.where(cond, 1, 0).astype(np.int64)
        j1 = np.where(cond, 0, 1).astype(np.int64)

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        g0x = grad3[gi0, 0]; g0y = grad3[gi0, 1]
        g1x = grad3[gi1, 0]; g1y = grad3[gi1, 1]
        g2x = grad3[gi2, 0]; g2y = grad3[gi2, 1]

        t0 = 0.5 - x0 * x0 - y0 * y0
        m0 = t0 > 0
        t0m = np.where(m0, t0 * t0, 0.0)
        n0 = np.where(m0, t0m * t0m * (g0x * x0 + g0y * y0), 0.0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        m1 = t1 > 0
        t1m = np.where(m1, t1 * t1, 0.0)
        n1 = np.where(m1, t1m * t1m * (g1x * x1 + g1y * y1), 0.0)

        t2 = 0.5 - x2 * x2 - y2 * y2
        m2 = t2 > 0
        t2m = np.where(m2, t2 * t2, 0.0)
        n2 = np.where(m2, t2m * t2m * (g2x * x2 + g2y * y2), 0.0)

        return 70.0 * (n0 + n1 + n2)

    def fbm_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * self.noise_batch(x * frequency, y * frequency)
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return total / max_amp if max_amp > 0 else np.zeros_like(x)

    def turbulence_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * np.abs(self.noise_batch(x * frequency, y * frequency))
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return (2.0 * total / max_amp - 1.0) if max_amp > 0 else np.zeros_like(x)


# ============ Path sampling helpers ============

def _path_subpaths(path):
    """Split an inkex Path into subpaths (lists of segments)."""
    subpaths = []
    current = []
    for seg in path:
        if isinstance(seg, Move) and current:
            subpaths.append(current)
            current = [seg]
        else:
            current.append(seg)
    if current:
        subpaths.append(current)
    return subpaths


def _subpath_polyline(subpath, samples_per_seg=24):
    """
    Sample a subpath into a polyline (list of (x, y)).

    NOTE: inkex's proxy_iterator / point(t) API is scalar; this loop is
    inherently per-sample in Python. It produces the same point sequence
    as the original implementation, so downstream NumPy code is bit-safe.
    """
    if not subpath:
        return []

    path = Path(subpath)
    poly = []
    subpath_start = None
    last_point = None

    for proxy in path.proxy_iterator():
        name = proxy.name

        if name == "Move":
            p = proxy.end_point
            poly.append((p.x, p.y))
            subpath_start = (p.x, p.y)
            last_point = (p.x, p.y)
            continue

        if name == "ZoneClose":
            if subpath_start is not None and last_point is not None:
                sx, sy = subpath_start
                lx, ly = last_point
                if math.hypot(sx - lx, sy - ly) > 1e-9:
                    for i in range(1, samples_per_seg + 1):
                        t = i / samples_per_seg
                        poly.append((lx + t * (sx - lx), ly + t * (sy - ly)))
            last_point = subpath_start
            continue

        for i in range(1, samples_per_seg + 1):
            t = i / samples_per_seg
            p = proxy.point(t)
            poly.append((p.x, p.y))
        last_point = (p.x, p.y)

    return poly


def _resample_polyline_by_arclength_np(poly, n, closed):
    """
    Vectorized resampling of a polyline to n points evenly spaced by arc length.
    Returns an (n, 2) float64 array.
    """
    if len(poly) < 2:
        return np.zeros((n, 2), dtype=np.float64)

    pts = np.asarray(poly, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]

    if total <= 0.0:
        return np.tile(pts[0], (n, 1))

    if closed:
        t_vals = (np.arange(n, dtype=np.float64) / n) * total
    else:
        if n > 1:
            t_vals = (np.arange(n, dtype=np.float64) / (n - 1)) * total
        else:
            t_vals = np.zeros(1, dtype=np.float64)

    # np.interp handles out-of-range t == total gracefully (clamps to last x).
    x_out = np.interp(t_vals, cum, pts[:, 0])
    y_out = np.interp(t_vals, cum, pts[:, 1])
    return np.stack([x_out, y_out], axis=1)


def _tangents_np(pts, closed):
    """
    Vectorized finite-difference tangents (unit length) for an (n, 2) array.
    Matches the original scalar behavior:
      - closed: prev = (i-1) % n, next = (i+1) % n
      - open:   endpoints use self as neighbor (same as original clamp)
    Returns (n, 2) array.
    """
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if closed:
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
    else:
        prev = pts.copy()
        nxt = pts.copy()
        if n >= 2:
            prev[1:] = pts[:-1]      # prev[i] = pts[i-1], prev[0] = pts[0]
            nxt[:-1] = pts[1:]       # nxt[i]  = pts[i+1], nxt[-1] = pts[-1]

    d = nxt - prev
    length = np.hypot(d[:, 0], d[:, 1])
    # Avoid division by zero: fall back to (1, 0) where length == 0
    safe = length > 0
    out = np.empty_like(d)
    out[safe] = d[safe] / length[safe, None]
    out[~safe] = (1.0, 0.0)
    return out


def _catmull_rom_controls_np(pts, closed):
    """
    Returns (p1, c1, c2, p2) arrays for Catmull-Rom -> Bezier conversion.
    Shapes: p1, c1, c2, p2 are (m, 2), where m = n (closed) or n-1 (open).

    For closed: m = n, segment i goes from pts[i] to pts[(i+1) % n].
    For open:   m = n-1, segment i goes from pts[i] to pts[i+1].
    """
    n = pts.shape[0]
    tension = 1.0 / 6.0

    if n < 2:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, empty, empty

    if closed:
        p0 = np.roll(pts, 1, axis=0)
        p1 = pts.copy()
        p2 = np.roll(pts, -1, axis=0)
        p3 = np.roll(pts, -2, axis=0)
    else:
        # ext = [pts[0]] + pts + [pts[-1]]
        ext = np.empty((n + 2, 2), dtype=np.float64)
        ext[0] = pts[0]
        ext[1:-1] = pts
        ext[-1] = pts[-1]
        p0 = ext[0:n - 1]
        p1 = ext[1:n]
        p2 = ext[2:n + 1]
        p3 = ext[3:n + 2]

    c1 = p1 + (p2 - p0) * tension
    c2 = p2 - (p3 - p1) * tension
    return p1, c1, c2, p2


# ============ Inkscape extension ============
class NoiseOffsetPath(inkex.EffectExtension):
    def add_arguments(self, parser):
        parser.add_argument("--points", type=int, default=80,
                            help="Number of samples along the selected path(s).")
        parser.add_argument("--amplitude", type=float, default=20,
                            help="Noise offset amplitude in document units.")
        parser.add_argument("--scale", type=float, default=1.2,
                            help="Noise frequency scale along the path.")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--noise_type",
            type=str,
            default="simplex",
            choices=["simplex", "fbm", "turbulence"],
            dest="noise_type",
        )
        parser.add_argument("--octaves", type=int, default=4)
        parser.add_argument("--lacunarity", type=float, default=2.0)
        parser.add_argument("--gain", type=float, default=0.5)
        parser.add_argument(
            "--smooth",
            type=inkex.Boolean,
            default=True,
            help="Smooth output (Catmull-Rom); false = polyline.",
        )

    def _sample_noise_batch(self, noise, nx, ny):
        kind = self.options.noise_type
        if kind == "fbm":
            return noise.fbm_batch(nx, ny,
                                   octaves=self.options.octaves,
                                   lacunarity=self.options.lacunarity,
                                   gain=self.options.gain)
        if kind == "turbulence":
            return noise.turbulence_batch(nx, ny,
                                          octaves=self.options.octaves,
                                          lacunarity=self.options.lacunarity,
                                          gain=self.options.gain)
        return noise.noise_batch(nx, ny)

    def effect(self):
        if not self.svg.selection:
            inkex.errormsg("Please select at least one object.")
            return

        noise = SimplexNoise(self.options.seed)
        n_samples = max(8, min(500, self.options.points))
        amp = self.options.amplitude
        scale = self.options.scale

        any_converted = False

        # Snapshot the selection as a list, because we may modify the tree.
        selected = list(self.svg.selection.values())

        # ---- Step 1: convert every selected element to a PathElement ----
        path_elems = []
        for elem in selected:
            if isinstance(elem, inkex.PathElement):
                path_elems.append(elem)
                continue

            try:
                new_elem = elem.to_path_element()
            except Exception as e:
                inkex.errormsg(f"Could not convert <{elem.tag}> to path: {e}")
                continue

            if new_elem is None:
                inkex.errormsg(f"Conversion of <{elem.tag}> returned no path.")
                continue

            parent = elem.getparent()
            if parent is None:
                continue
            index = parent.index(elem)
            parent.remove(elem)
            parent.insert(index, new_elem)
            path_elems.append(new_elem)

        if not path_elems:
            inkex.errormsg("No path-like elements to process.")
            return

        # ---- Step 2: apply noise offset to each path ----
        for elem in path_elems:
            path = elem.path.to_absolute()
            transform = elem.transform
            if transform:
                path = path.transform(transform)
                clear_transform = True
            else:
                clear_transform = False

            subpaths = _path_subpaths(path)

            new_path_segments = []
            for subpath in subpaths:
                if not subpath:
                    continue

                poly = _subpath_polyline(subpath, samples_per_seg=24)
                if len(poly) < 2:
                    continue

                closed = any(isinstance(s, ZoneClose) for s in subpath)
                if not closed and len(poly) > 2:
                    if math.hypot(poly[0][0] - poly[-1][0],
                                  poly[0][1] - poly[-1][1]) < 1e-6:
                        closed = True

                # (n, 2)
                samples = _resample_polyline_by_arclength_np(poly, n_samples, closed)

                # (n, 2) unit tangents
                tangents = _tangents_np(samples, closed)
                normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

                # Build noise input coordinates in one vectorized pass.
                if closed:
                    theta = (np.arange(n_samples, dtype=np.float64) / n_samples) * 2.0 * math.pi
                    nz_x = np.cos(theta) * scale
                    nz_y = np.sin(theta) * scale
                else:
                    if n_samples > 1:
                        t = np.arange(n_samples, dtype=np.float64) / (n_samples - 1)
                    else:
                        t = np.zeros(1, dtype=np.float64)
                    nz_x = t * scale
                    nz_y = np.zeros_like(nz_x)

                v = self._sample_noise_batch(noise, nz_x, nz_y)  # (n,)

                offset_pts = samples + normals * (v[:, None] * amp)  # (n, 2)

                d = self._catmull_or_poly_np(offset_pts, closed)
                new_path_segments.append(d)

            if not new_path_segments:
                continue

            combined_d = " ".join(new_path_segments)
            elem.set("d", combined_d)
            if clear_transform:
                elem.set("transform", "")
            any_converted = True

        if not any_converted:
            inkex.errormsg("No valid path elements were selected.")

    # ---- Output formatting ----

    def _catmull_or_poly_np(self, pts, closed):
        if self.options.smooth:
            return self._catmull_rom_to_svg_path_np(pts, closed)
        return self._polyline_to_svg_path_np(pts, closed)

    @staticmethod
    def _polyline_to_svg_path_np(pts, closed):
        n = pts.shape[0]
        if n < 2:
            return ""
        parts = [f"M {pts[0, 0]:.4f},{pts[0, 1]:.4f}"]
        for i in range(1, n):
            parts.append(f"L {pts[i, 0]:.4f},{pts[i, 1]:.4f}")
        if closed:
            parts.append("Z")
        return " ".join(parts)

    @staticmethod
    def _catmull_rom_to_svg_path_np(pts, closed):
        n = pts.shape[0]
        if n < 2:
            return ""
        if not closed and n < 3:
            return NoiseOffsetPath._polyline_to_svg_path_np(pts, closed)

        p1, c1, c2, p2 = _catmull_rom_controls_np(pts, closed)
        parts = [f"M {pts[0, 0]:.4f},{pts[0, 1]:.4f}"]

        # Iterate over segments; each row is one cubic.
        for i in range(p1.shape[0]):
            parts.append(
                f"C {c1[i, 0]:.4f},{c1[i, 1]:.4f} "
                f"{c2[i, 0]:.4f},{c2[i, 1]:.4f} "
                f"{p2[i, 0]:.4f},{p2[i, 1]:.4f}"
            )
        if closed:
            parts.append("Z")
        return " ".join(parts)


if __name__ == "__main__":
    NoiseOffsetPath().run()