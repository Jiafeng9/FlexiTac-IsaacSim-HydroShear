# Bump strip STL outputs

Generated from the uploaded 16 x 16 bump-strip dimensions, with the array reduced to 4 x 8 and 8 x 16 raised bumps.

## Preserved external bump geometry

- Bump pitch: 8.75 mm
- Bump footprint diameter: 8.00 mm
- Bump top flat diameter: 5.30 mm
- Bump height above base top: 3.73 mm
- Circular tessellation: 126 segments, matching the uploaded STL's bump resolution

The STL coordinates are re-zeroed so the lowest underside point is at Z = 0.

## Base / underside defaults

- Trimmed border outside bump footprint: 2.00 mm
- Current-style stepped underside:
  - thick perimeter/rim thickness: 3.00 mm
  - main sheet thickness under the bump array: 1.00 mm
  - raised underside plane is at Z = 2.00 mm
  - base top plane is at Z = 3.00 mm

## Generated positive strip files

### 4 rows x 8 columns

- `bump_strip_4rows_8cols_solid_reference.stl` — trimmed solid/reference version.
- `bump_strip_4rows_8cols_medium_hollow.stl` — 1.6 mm side wall, 1.2 mm top skin.
- `bump_strip_4rows_8cols_soft_hollow.stl` — 1.2 mm side wall, 1.0 mm top skin.
- `bump_strip_4rows_8cols_extra_soft_hollow.stl` — 0.8 mm side wall, 0.8 mm top skin.
- `bump_strip_4rows_8cols_soft_hollow_relief_grooves.stl` — soft hollow preset plus underside relief grooves.

Footprint: 73.25 x 38.25 x 6.73 mm.

### 8 rows x 16 columns

- `bump_strip_8rows_16cols_solid_reference.stl` — trimmed solid/reference version.
- `bump_strip_8rows_16cols_medium_hollow.stl` — 1.6 mm side wall, 1.2 mm top skin.
- `bump_strip_8rows_16cols_soft_hollow.stl` — 1.2 mm side wall, 1.0 mm top skin.
- `bump_strip_8rows_16cols_extra_soft_hollow.stl` — 0.8 mm side wall, 0.8 mm top skin.

Footprint: 143.25 x 73.25 x 6.73 mm.

## Mold files

- `open_face_mold_4rows_8cols.stl`
- `open_face_mold_8rows_16cols.stl`

These are one-piece open-face negative molds intended for casting solid silicone bump strips. The exposed pour side becomes the flat back of the cast strip. They do not make hollow silicone strips; hollow cast silicone would need a core or two-part mold.

## Script dependencies

The generator script is included as `bump_strip_generator.py`.

Recommended install:

```bash
python -m pip install numpy shapely trimesh triangle mapbox_earcut
```

`triangle` is recommended for the relief-groove option. `mapbox_earcut` is fast for the non-groove variants.

## Example commands

Generate a 4 x 8 extra-soft hollow strip:

```bash
python bump_strip_generator.py --rows 4 --cols 8 --variant extra-soft --output bump_4x8_extra_soft.stl
```

Generate a custom hollow version with a 1.0 mm side wall and 0.8 mm top skin:

```bash
python bump_strip_generator.py --rows 4 --cols 8 --variant custom --side-wall 1.0 --top-skin 0.8 --output bump_custom.stl
```

Generate a 4 x 8 soft hollow strip with relief grooves:

```bash
python bump_strip_generator.py --rows 4 --cols 8 --variant soft --relief-grooves --groove-axis both --output bump_4x8_soft_grooved.stl
```

Generate an open-face mold:

```bash
python bump_strip_generator.py --mode mold --rows 4 --cols 8 --output mold_4x8.stl
```
