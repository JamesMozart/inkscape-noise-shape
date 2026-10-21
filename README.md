# Inkscape Noise Shape

An Inkscape 1.x extension that applies 2D noise displacement to the paths of selected objects, producing organic, irregular, or torn-looking outlines.

Use it for:
- Turning clean rectangles, ellipses, and polygons into clouds, ink blots, cells, or island silhouettes
- Icon backplates, sticker outlines, and background decorative blobs
- Explosion or spike effects (large amplitude + high frequency)
- Torn paper, rough edges, fire/smoke contours
- Low-poly / crystalline styles (low sample count + smooth off)

## How it works

The extension does not generate new shapes — it **modifies the objects you select**:

1. Selected rectangles, ellipses, and polygons are first converted to paths automatically.
2. Each path is sampled at `Sample points` evenly spaced positions along its arc length.
3. At each sample point, the normal direction is computed and a noise value is used to push the point outward or inward along that normal.
4. The path is rebuilt from the displaced points (smooth curve or polyline) and written back to the original object or added as a new object.

Closed paths sample the noise around a circle so the start and end match seamlessly. Open paths sample noise linearly along arc length, so the two ends are independent.

## Features

- **Three noise types**: single-layer Simplex, fractal FBM (fractal Brownian motion), and Turbulence (absolute-value fractal).
- **Automatic shape conversion**: rectangles, ellipses, circles, and polygons need no manual conversion.
- **Closed and open paths**: seamless for closed paths, independent endpoints for open paths.
- **Arc-length sampling**: `Sample points` are evenly distributed along the total path length, independent of the original Bézier segment lengths.
- **Smooth or polyline output**: Catmull-Rom splines through every sample point, or straight-segment polylines.
- **No external dependencies**: includes a lightweight Simplex Noise implementation, compatible with Inkscape's bundled Python.

## Install

Copy `noise_shape.py` and `noise_shape.inx` into your Inkscape extensions folder:

- **Linux:** `~/.config/inkscape/extensions/`
- **Windows:** `%APPDATA%\inkscape\extensions\`
- **macOS:** `~/Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions/`

Restart Inkscape. Requires Inkscape 1.x (Python 3 + `inkex`).

## Usage

1. Select one or more objects on the canvas (paths, rectangles, ellipses, or polygons).
2. Open **Extensions → Generate from Path → Noise Shape**.
3. Adjust the parameters and click **Apply**.

## Parameters

| Parameter | Effect |
|---|---|
| Sample points | Number of samples evenly spaced along the total path length. Higher = smoother, more detail, slower; lower = more angular. Range 8–500. |
| Noise amplitude | Maximum displacement along the normal, in document units. Larger = more irregular and spiky; smaller = closer to the original outline. |
| Noise scale | Noise frequency along the path. Larger = finer, more fragmented detail; smaller = broader, smoother undulations. |
| Random seed | Same seed + same settings reproduce the same result; change it for a different shape. |
| Noise type | `Simplex` = single smooth layer; `FBM` = stacked octaves for natural multi-scale detail (clouds, terrain); `Turbulence` = stacked absolute-value octaves for sharp, creased, fiery or smoky outlines. |
| Octaves | Number of noise layers stacked for `FBM` / `Turbulence`. Higher = more detail, slower. Typically 3–6. |
| Lacunarity | Frequency multiplier between octaves for `FBM` / `Turbulence`. Usually 2.0. Higher = finer detail scales. |
| Gain | Amplitude decay between octaves for `FBM` / `Turbulence`. Usually 0.5. Higher = rougher, stronger high-frequency contribution. |
| Smooth output | On = Catmull-Rom spline through every sample point (organic, no visible corners); Off = straight-segment polyline (crisp angular look, good for low-poly / crystalline styles). |

> **Note:** `Octaves`, `Lacunarity`, and `Gain` only take effect when the noise type is `FBM` or `Turbulence`. They are ignored for `Simplex`.

### Noise type quick guide

| Type | Best for | Suggested settings |
|---|---|---|
| Simplex | Blobs, cells, simple organic outlines | Defaults |
| FBM | Clouds, islands, terrain-like shapes with layered detail | Octaves 4–5, Gain 0.5 |
| Turbulence | Fire, smoke, torn paper, spiky/creased edges | Octaves 3–4, Gain 0.5–0.6, reduce amplitude slightly |

### Style suggestions

- **Organic clouds / ink blots**: Sample points 80–150, Amplitude 15–30, Scale 1.0–2.0, FBM, Smooth on.
- **Torn paper / rough edges**: Sample points 150–300, Amplitude 5–15, Scale 3–6, Turbulence, Smooth on.
- **Explosion / spikes**: Sample points 60–120, Amplitude 40–100, Scale 2–4, Turbulence, Smooth off.
- **Low-poly / crystalline**: Sample points 12–20, Amplitude 20–50, Scale 1–2, Smooth off.
- **Sticker outlines / icon backplates**: Sample points 100–200, Amplitude 8–20, Scale 1.5–2.5, FBM, Smooth on.

## Known limitations

- Groups (`<g>`) are not recursively expanded. Operate on the members individually, or ungroup first.
- If the original object has a `transform`, it is baked into the path coordinates and the `transform` attribute is cleared. The visual position is preserved, but the `transform` attribute is lost.

## License

MIT