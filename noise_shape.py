import inkex
import math
import random

# ============ Lightweight Simplex Noise (2D) ============
# Simplified from Stefan Gustavson's public implementation
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

        # Roughly in [-1, 1] (actually about [-0.7, 0.7]); multiply by 70 to normalize
        return 70.0 * (n0 + n1 + n2)


# ============ Inkscape extension ============
class NoiseShape(inkex.EffectExtension):
    def add_arguments(self, parser):
        parser.add_argument("--points", type=int, default=60)
        parser.add_argument("--radius", type=float, default=80)
        parser.add_argument("--amplitude", type=float, default=30)
        parser.add_argument("--scale", type=float, default=1.2)
        parser.add_argument("--seed", type=int, default=42)

    def effect(self):
        n = max(12, min(200, self.options.points))
        R = self.options.radius
        amp = self.options.amplitude
        scale = self.options.scale
        seed = self.options.seed

        noise = SimplexNoise(seed)
        rng = random.Random(seed)

        # Randomly split 360 degrees into n positive increments that sum to 2*pi
        weights = [rng.random() + 0.05 for _ in range(n)]
        wsum = sum(weights)
        increments = [2.0 * math.pi * w / wsum for w in weights]

        pts = []
        theta = 0.0
        for k in range(n):
            # Sampling coordinates on the unit circle, scaled to control noise frequency
            nx = math.cos(theta) * scale
            ny = math.sin(theta) * scale
            v = noise.noise(nx, ny)  # roughly [-1, 1]
            r = R + amp * v
            r = max(1.0, r)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            pts.append((x, y))
            theta += increments[k]  # advance by the random increment

        # After the loop theta should be ~2*pi; the shape closes implicitly
        pts = self._center_at_view(pts)
        d = self._catmull_rom_closed_to_svg_path(pts)

        path = inkex.PathElement()
        path.set("d", d)
        path.set("style", "fill:#ffffff;stroke:#000000;stroke-width:1;")
        self.svg.get_current_layer().append(path)

    def _center_at_view(self, pts):
        """Translate points so their bounding-box center lands on the view center."""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0

        # self.view_center is the current view center in document coordinates
        target = self.svg.namedview.center
        dx = target.x - cx
        dy = target.y - cy

        return [(x + dx, y + dy) for (x, y) in pts]

    def _catmull_rom_closed_to_svg_path(self, pts):
        """Convert a closed point list into a smooth SVG Bezier path string."""
        n = len(pts)
        if n < 3:
            return ""

        # Tension factor for Catmull-Rom to Bezier; 1/6 is the standard conversion
        tension = 1.0 / 6.0

        parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]

        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i % n]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]

            # Control points
            c1x = p1[0] + (p2[0] - p0[0]) * tension
            c1y = p1[1] + (p2[1] - p0[1]) * tension
            c2x = p2[0] - (p3[0] - p1[0]) * tension
            c2y = p2[1] - (p3[1] - p1[1]) * tension

            parts.append(
                f"C {c1x:.4f},{c1y:.4f} {c2x:.4f},{c2y:.4f} {p2[0]:.4f},{p2[1]:.4f}"
            )

        parts.append("Z")
        return " ".join(parts)


if __name__ == "__main__":
    NoiseShape().run()