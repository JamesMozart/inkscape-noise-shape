import inkex
import math
import numpy as np
from path_helper import boundary_polylines
from simplex_noise import SimplexNoise


def _tangent_at_batch(pts, closed):
    """Vectorized tangents at every point of a polyline.

    pts: (N, 2) float64 array
    returns: (N, 2) unit tangent vectors
    """
    n = len(pts)
    if n < 2:
        return np.tile(np.array([1.0, 0.0]), (n, 1))

    if closed:
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
    else:
        prev = np.empty_like(pts)
        nxt = np.empty_like(pts)
        prev[0] = pts[0]
        prev[1:] = pts[:-1]
        nxt[-1] = pts[-1]
        nxt[:-1] = pts[1:]

    d = nxt - prev
    length = np.hypot(d[:, 0], d[:, 1])
    # Avoid division by zero; fall back to (1, 0) where length == 0
    safe = length > 0
    out = np.empty_like(d)
    out[safe] = d[safe] / length[safe, None]
    out[~safe] = np.array([1.0, 0.0])
    return out


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
            "--tangent_offset",
            type=float,
            default=50.0,
            help="Tangential offset as a percentage of the normal displacement "
                 "(can be negative). 0 = pure normal offset.",
        )
        parser.add_argument(
            "--smooth",
            type=inkex.Boolean,
            default=True,
            help="Smooth output (Catmull-Rom); false = polyline.",
        )

    def _sample_noise_batch(self, noise, nx, ny):
        """Vectorized noise sampling over arrays nx, ny."""
        kind = self.options.noise_type
        if kind == "fbm":
            return noise.fbm_batch(
                nx, ny,
                octaves=self.options.octaves,
                lacunarity=self.options.lacunarity,
                gain=self.options.gain,
            )
        if kind == "turbulence":
            return noise.turbulence_batch(
                nx, ny,
                octaves=self.options.octaves,
                lacunarity=self.options.lacunarity,
                gain=self.options.gain,
            )
        return noise.noise_batch(nx, ny)

    def effect(self):
        if not self.svg.selection:
            inkex.errormsg("Please select at least one object.")
            return

        noise = SimplexNoise(self.options.seed)
        n_samples = max(8, min(500, self.options.points))
        amp = self.options.amplitude
        scale = self.options.scale
        # Tangent offset ratio: percentage -> fraction
        tangent_ratio = self.options.tangent_offset / 100.0

        any_converted = False
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

            new_path_segments = []
            for poly, closed in boundary_polylines(path, samples_per_seg=24,
                                           target_resample=n_samples):
                n = poly.shape[0]

                # ---- vectorized tangents & normals ----
                tangents = _tangent_at_batch(poly, closed)
                normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

                # ---- vectorized noise coordinates ----
                if closed:
                    theta = (np.arange(n) / n_samples) * (2.0 * math.pi)
                    nz_x = np.cos(theta) * scale
                    nz_y = np.sin(theta) * scale
                else:
                    t = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
                    nz_x = t * scale
                    nz_y = np.zeros_like(t)

                v = self._sample_noise_batch(noise, nz_x, nz_y)

                # ---- normal + tangential displacement ----
                # Normal displacement: v * amp
                # Tangential displacement: same scalar scaled by tangent_ratio
                normal_disp = v[:, None] * amp
                tangent_disp = (np.sqrt(np.abs(v)) * np.sign(v))[:, None] * amp * tangent_ratio
                offset_pts = (
                    poly
                    + normals * normal_disp
                    + tangents * tangent_disp
                )

                d = self._catmull_or_poly(offset_pts, closed)
                new_path_segments.append(d)

            if not new_path_segments:
                continue

            elem.set("d", " ".join(new_path_segments))
            if clear_transform:
                elem.set("transform", "")
            any_converted = True

        if not any_converted:
            inkex.errormsg("No valid path elements were selected.")

    def _catmull_or_poly(self, pts, closed):
        if self.options.smooth:
            return self._catmull_rom_to_svg_path(pts, closed)
        return self._polyline_to_svg_path(pts, closed)

    def _polyline_to_svg_path(self, pts, closed):
        n = len(pts)
        if n < 2:
            return ""
        # Vectorized "M x,y L x,y ..."
        coords = np.asarray(pts, dtype=np.float64)
        body = " ".join(f"L {x:.4f},{y:.4f}" for x, y in coords[1:])
        head = f"M {coords[0, 0]:.4f},{coords[0, 1]:.4f}"
        return f"{head} {body}" + (" Z" if closed else "")

    def _catmull_rom_to_svg_path(self, pts, closed):
        pts = np.asarray(pts, dtype=np.float64)
        n = pts.shape[0]
        if n < 2:
            return ""
        if not closed and n < 3:
            return self._polyline_to_svg_path(pts, closed)

        tension = 1.0 / 6.0

        if closed:
            p0 = np.roll(pts, 1, axis=0)
            p1 = pts
            p2 = np.roll(pts, -1, axis=0)
            p3 = np.roll(pts, -2, axis=0)
        else:
            ext = np.vstack([pts[0], pts, pts[-1]])
            p0 = ext[:-3]
            p1 = ext[1:-2]
            p2 = ext[2:-1]
            p3 = ext[3:]

        c1 = p1 + (p2 - p0) * tension
        c2 = p2 - (p3 - p1) * tension

        # Emit "M p1[0]" then one "C c1 c2 p2" per segment
        head = f"M {pts[0, 0]:.4f},{pts[0, 1]:.4f}"
        segs = [
            f"C {a:.4f},{b:.4f} {c:.4f},{d:.4f} {e:.4f},{f:.4f}"
            for (a, b), (c, d), (e, f) in zip(c1, c2, p2)
        ]
        tail = " Z" if closed else ""
        return f"{head} " + " ".join(segs) + tail


if __name__ == "__main__":
    NoiseOffsetPath().run()