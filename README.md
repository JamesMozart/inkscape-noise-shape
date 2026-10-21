# Inkscape Noise Tiling

Scatter randomized copies of selected objects over a region, with placement probability driven by 2D Simplex noise. Higher noise values → higher spawn chance. Overlaps are avoided via AABB collision detection; each copy gets random rotation and scale.

![Example output](example.png)

## Features

- Simplex-noise–driven spawn probability
- No-overlap placement (AABB + padding)
- Random rotation & scale (configurable ranges)
- Tiling area anchored to the selected object's top-left corner
- Output objects have no `transform` (path `d` is rewritten directly)

## Install

Copy both files into your Inkscape extensions folder:

- **Linux:** `~/.config/inkscape/extensions/`
- **Windows:** `%APPDATA%\inkscape\extensions\`
- **macOS:** `~/Library/Application Support/org.inkscape.Inkscape/config/inkscape/extensions/`

Restart Inkscape.

Requires Inkscape 1.x (Python 3 + `inkex`).

## Usage

1. Select one or more objects to use as templates.
2. Go to **Extensions → Generate from Path → Simplex Noise Tiling**.
3. Adjust parameters and click **Apply**.

## Parameters

| Param | Description |
|---|---|
| Tile Area Width/Height | Size of the tiling region (starts at template's top-left corner) |
| Noise Scale | Smaller = smoother/larger clusters |
| Probability Threshold | Filter out low-noise positions (0–1) |
| Noise Seed | `0` = random each run; any other value reproduces the same layout |
| Max Placement Attempts | Upper bound on placement tries |
| Min/Max Random Scale | Scale range for each copy |
| Min/Max Random Rotation (deg) | Rotation range for each copy |
| Padding | Extra gap enforced between objects |

## License

MIT