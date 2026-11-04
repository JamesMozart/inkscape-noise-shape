import inkex
import math
import random
from path_helper import boundary_polylines

# ============ Lightweight Simplex Noise (2D) ============
class SimplexNoise:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        self.perm = list(range(256))
        rng.shuffle(self.perm)
        self.perm = self.perm * 2
        self.grad3 = [
            (1,1), (-1,1), (1,-1), (-1,-1),
            (1,0), (-1,0), (0,1), (0,-1)
        ]
        self.F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - math.sqrt(3.0)) / 6.0

    def _dot(self, g, x, y):
        return g[0]*x + g[1]*y

    def noise(self, xin, yin):
        s = (xin + yin) * self.F2
        i = math.floor(xin + s)
        j = math.floor(yin + s)
        t = (i + j) * self.G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)

        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2

        ii = int(i) & 255
        jj = int(j) & 255

        gi0 = self.perm[ii + self.perm[jj]] % 8
        gi1 = self.perm[ii + i1 + self.perm[jj + j1]] % 8
        gi2 = self.perm[ii + 1 + self.perm[jj + 1]] % 8

        n0 = n1 = n2 = 0.0
        t0 = 0.5 - x0*x0 - y0*y0
        if t0 > 0:
            t0 *= t0
            n0 = t0 * t0 * self._dot(self.grad3[gi0], x0, y0)

        t1 = 0.5 - x1*x1 - y1*y1
        if t1 > 0:
            t1 *= t1
            n1 = t1 * t1 * self._dot(self.grad3[gi1], x1, y1)

        t2 = 0.5 - x2*x2 - y2*y2
        if t2 > 0:
            t2 *= t2
            n2 = t2 * t2 * self._dot(self.grad3[gi2], x2, y2)

        return 70.0 * (n0 + n1 + n2)

    def fbm(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * self.noise(x * frequency, y * frequency)
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return total / max_amp if max_amp > 0 else 0.0

    def turbulence(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * abs(self.noise(x * frequency, y * frequency))
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return (2.0 * total / max_amp - 1.0) if max_amp > 0 else 0.0


def _tangent_at(pts, i, closed):
    """Approximate tangent at pts[i] using neighbors."""
    n = len(pts)
    if closed:
        prev = pts[(i - 1) % n]
        nxt = pts[(i + 1) % n]
    else:
        if i == 0:
            prev = pts[0]
            nxt = pts[1]
        elif i == n - 1:
            prev = pts[n - 2]
            nxt = pts[n - 1]
        else:
            prev = pts[i - 1]
            nxt = pts[i + 1]
    dx = nxt[0] - prev[0]
    dy = nxt[1] - prev[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (1.0, 0.0)
    return (dx / length, dy / length)


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

    def _sample_noise(self, noise, nx, ny):
        kind = self.options.noise_type
        if kind == "fbm":
            return noise.fbm(
                nx, ny,
                octaves=self.options.octaves,
                lacunarity=self.options.lacunarity,
                gain=self.options.gain,
            )
        if kind == "turbulence":
            return noise.turbulence(
                nx, ny,
                octaves=self.options.octaves,
                lacunarity=self.options.lacunarity,
                gain=self.options.gain,
            )
        return noise.noise(nx, ny)

    def effect(self):
        if not self.svg.selection:
            inkex.errormsg("Please select at least one object.")
            return

        noise = SimplexNoise(self.options.seed)
        n_samples = max(8, min(500, self.options.points))
        amp = self.options.amplitude
        scale = self.options.scale

        any_converted = False

        # Snapshot the selection as a list, because we may modify the tree
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
                inkex.errormsg(
                    f"Could not convert <{elem.tag}> to path: {e}"
                )
                continue

            if new_elem is None:
                inkex.errormsg(f"Conversion of <{elem.tag}> returned no path.")
                continue

            # Replace original element with the converted path in the tree
            parent = elem.getparent()
            if parent is None:
                continue
            index = parent.index(elem)
            parent.remove(elem)
            parent.insert(index, new_elem)
            # Copy over the id and any style that to_path_element may not preserve
            # (to_path_element usually preserves id, style, transform, but let's
            # be safe about transform — it is normally kept.)
            path_elems.append(new_elem)

        if not path_elems:
            inkex.errormsg("No path-like elements to process.")
            return

        # ---- Step 2: apply noise offset to each path ----
        for elem in path_elems:
            path = elem.path.to_absolute()
            transform = elem.transform
            # Bake the element's own transform into the path coordinates so
            # that we can write back in parent space and clear transform.
            if transform:
                path = path.transform(transform)
                clear_transform = True
            else:
                clear_transform = False

            new_path_segments = []
            for poly, closed in boundary_polylines(path, samples_per_seg=24, target_resample=n_samples):
                if len(poly) < 2:
                    continue

                offset_pts = []
                for i, (x, y) in enumerate(poly):
                    tx, ty = _tangent_at(poly, i, closed)
                    nx_dir, ny_dir = -ty, tx  # normal = tangent rotated +90°

                    if closed:
                        theta = (i / n_samples) * 2.0 * math.pi
                        nz_x = math.cos(theta) * scale
                        nz_y = math.sin(theta) * scale
                    else:
                        t = i / (n_samples - 1) if n_samples > 1 else 0.0
                        nz_x = t * scale
                        nz_y = 0.0

                    v = self._sample_noise(noise, nz_x, nz_y)
                    offset_pts.append((
                        x + nx_dir * v * amp,
                        y + ny_dir * v * amp,
                    ))

                d = self._catmull_or_poly(offset_pts, closed)
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

    def _catmull_or_poly(self, pts, closed):
        if self.options.smooth:
            return self._catmull_rom_to_svg_path(pts, closed)
        else:
            return self._polyline_to_svg_path(pts, closed)

    def _polyline_to_svg_path(self, pts, closed):
        n = len(pts)
        if n < 2:
            return ""
        parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]
        for i in range(1, n):
            parts.append(f"L {pts[i][0]:.4f},{pts[i][1]:.4f}")
        if closed:
            parts.append("Z")
        return " ".join(parts)

    def _catmull_rom_to_svg_path(self, pts, closed):
        n = len(pts)
        if n < 2:
            return ""
        if not closed and n < 3:
            return self._polyline_to_svg_path(pts, closed)

        tension = 1.0 / 6.0
        parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]

        if closed:
            for i in range(n):
                p0 = pts[(i - 1) % n]
                p1 = pts[i % n]
                p2 = pts[(i + 1) % n]
                p3 = pts[(i + 2) % n]
                c1x = p1[0] + (p2[0] - p0[0]) * tension
                c1y = p1[1] + (p2[1] - p0[1]) * tension
                c2x = p2[0] - (p3[0] - p1[0]) * tension
                c2y = p2[1] - (p3[1] - p1[1]) * tension
                parts.append(
                    f"C {c1x:.4f},{c1y:.4f} {c2x:.4f},{c2y:.4f} {p2[0]:.4f},{p2[1]:.4f}"
                )
            parts.append("Z")
        else:
            # Open path: handle endpoints with clamped neighbors
            ext = [pts[0]] + pts + [pts[-1]]
            for i in range(len(pts) - 1):
                p0 = ext[i]
                p1 = ext[i + 1]
                p2 = ext[i + 2]
                p3 = ext[i + 3]
                c1x = p1[0] + (p2[0] - p0[0]) * tension
                c1y = p1[1] + (p2[1] - p0[1]) * tension
                c2x = p2[0] - (p3[0] - p1[0]) * tension
                c2y = p2[1] - (p3[1] - p1[1]) * tension
                parts.append(
                    f"C {c1x:.4f},{c1y:.4f} {c2x:.4f},{c2y:.4f} {p2[0]:.4f},{p2[1]:.4f}"
                )
        return " ".join(parts)


if __name__ == "__main__":
    NoiseOffsetPath().run()