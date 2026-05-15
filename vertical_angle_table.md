# Vertical Angle Tables for LiDAR Range Images

This note collects dataset-specific LiDAR vertical-angle lookup tables for replacing a linear elevation mapping with a per-row / per-channel lookup.

Conventions used below:

- Angles are in degrees.
- Positive elevation points upward, negative elevation points downward.
- `channel` / `ring` is the raw sensor channel index when the dataset exposes one.
- `row_top0` is the recommended range-image row convention for rendering: row `0` is the highest elevation and row increases downward.
- For datasets whose calibration is stored per sequence/frame, the correct table must be read from the dataset calibration rather than hard-coded globally.

## Summary

| Dataset | Main LiDAR | Point ring/channel in released points? | Can we hard-code a table? | Recommended source of table |
|---|---:|---:|---:|---|
| nuScenes | Velodyne HDL-32E | Yes, `.pcd.bin` stores `x,y,z,intensity,ring`; devkit drops ring by default | Yes, nominal HDL-32E table is usable | HDL-32E firing table, then sort by elevation |
| Waymo Open Dataset | 1 top mid-range + 4 short-range LiDARs | Data is already range image rows, not ordinary point ring | No single global table | Read `LaserCalibration.beam_inclinations`; if empty, interpolate `beam_inclination_min/max` |
| ONCE | Hesai Pandar40P, 40 beams | Public point cloud format is typically `x,y,z,intensity`; no reliable ring in released point cloud | Yes for design-value table; exact unit table requires Hesai angle correction file | Pandar40P Appendix channel distribution |
| KITTI raw / odometry / object | Velodyne HDL-64E | No, `.bin` is `x,y,z,reflectance` | Not exactly; use sensor `db.xml` if available, otherwise nominal HDL-64E/S2 style table only | Velodyne `db.xml` / calibration stream; fallback: nominal VFOV |
| KITTI-360 | Velodyne HDL-64E | Public `.bin`/point cloud paths do not expose ring as a standard field | Not exactly; same issue as KITTI | Velodyne `db.xml` / calibration stream; fallback: nominal VFOV |

## nuScenes: Velodyne HDL-32E

> **Empirical note (2026-05-14)**: the `ring` field nuScenes ships in `.pcd.bin`
> is **NOT** the HDL-32E manufacturer firing order shown in the table just below.
> Verified by computing the mean `atan2(z, sqrt(x²+y²))` per ring on a real
> sample — ring `k` has the `k`-th lowest elevation (ascending). Use the sorted
> table further down (`row_top0`/row_bottom0 — same values, just choose a row
> convention) as the ring → elevation source of truth for nuScenes. The QGS
> config keeps this sorted table in `ring_to_elevation_deg`, so `ring_to_row`
> is identity.

Raw ring order from the HDL-32E manual (kept for reference only; do NOT use
directly with the nuScenes `ring` channel):

| ring | elevation_deg |
|---:|---:|
| 0 | -30.67 |
| 1 | -9.33 |
| 2 | -29.33 |
| 3 | -8.00 |
| 4 | -28.00 |
| 5 | -6.66 |
| 6 | -26.66 |
| 7 | -5.33 |
| 8 | -25.33 |
| 9 | -4.00 |
| 10 | -24.00 |
| 11 | -2.67 |
| 12 | -22.67 |
| 13 | -1.33 |
| 14 | -21.33 |
| 15 | 0.00 |
| 16 | -20.00 |
| 17 | 1.33 |
| 18 | -18.67 |
| 19 | 2.67 |
| 20 | -17.33 |
| 21 | 4.00 |
| 22 | -16.00 |
| 23 | 5.33 |
| 24 | -14.67 |
| 25 | 6.67 |
| 26 | -13.33 |
| 27 | 8.00 |
| 28 | -12.00 |
| 29 | 9.33 |
| 30 | -10.67 |
| 31 | 10.67 |

Recommended renderer / GT-row order for a normal image layout, `row_top0`.

| row_top0 | ring | elevation_deg |
|---:|---:|---:|
| 0 | 31 | 10.67 |
| 1 | 29 | 9.33 |
| 2 | 27 | 8.00 |
| 3 | 25 | 6.67 |
| 4 | 23 | 5.33 |
| 5 | 21 | 4.00 |
| 6 | 19 | 2.67 |
| 7 | 17 | 1.33 |
| 8 | 15 | 0.00 |
| 9 | 13 | -1.33 |
| 10 | 11 | -2.67 |
| 11 | 9 | -4.00 |
| 12 | 7 | -5.33 |
| 13 | 5 | -6.66 |
| 14 | 3 | -8.00 |
| 15 | 1 | -9.33 |
| 16 | 30 | -10.67 |
| 17 | 28 | -12.00 |
| 18 | 26 | -13.33 |
| 19 | 24 | -14.67 |
| 20 | 22 | -16.00 |
| 21 | 20 | -17.33 |
| 22 | 18 | -18.67 |
| 23 | 16 | -20.00 |
| 24 | 14 | -21.33 |
| 25 | 12 | -22.67 |
| 26 | 10 | -24.00 |
| 27 | 8 | -25.33 |
| 28 | 6 | -26.66 |
| 29 | 4 | -28.00 |
| 30 | 2 | -29.33 |
| 31 | 0 | -30.67 |

Implementation note: for nuScenes GT range map, use `row_top0 = row_from_ring[ring]` instead of computing row from linearly spaced elevation. The renderer should also use `elevation_by_row[row_top0]`, not `linspace(-30.67, 10.67, 32)`.

## ONCE: Hesai Pandar40P

The ONCE public docs describe a 40-beam LiDAR with vertical FOV `[-25, +15]`. The sensor matches the Hesai Pandar40P design table. The values below are design values; Hesai says exact values are stored in the unit's angle correction file.

| channel | row_top0 | elevation_deg |
|---:|---:|---:|
| 1 | 0 | 15.00 |
| 2 | 1 | 11.00 |
| 3 | 2 | 8.00 |
| 4 | 3 | 5.00 |
| 5 | 4 | 3.00 |
| 6 | 5 | 2.00 |
| 7 | 6 | 1.67 |
| 8 | 7 | 1.33 |
| 9 | 8 | 1.00 |
| 10 | 9 | 0.67 |
| 11 | 10 | 0.33 |
| 12 | 11 | 0.00 |
| 13 | 12 | -0.33 |
| 14 | 13 | -0.67 |
| 15 | 14 | -1.00 |
| 16 | 15 | -1.33 |
| 17 | 16 | -1.67 |
| 18 | 17 | -2.00 |
| 19 | 18 | -2.33 |
| 20 | 19 | -2.67 |
| 21 | 20 | -3.00 |
| 22 | 21 | -3.33 |
| 23 | 22 | -3.67 |
| 24 | 23 | -4.00 |
| 25 | 24 | -4.33 |
| 26 | 25 | -4.67 |
| 27 | 26 | -5.00 |
| 28 | 27 | -5.33 |
| 29 | 28 | -5.67 |
| 30 | 29 | -6.00 |
| 31 | 30 | -7.00 |
| 32 | 31 | -8.00 |
| 33 | 32 | -9.00 |
| 34 | 33 | -10.00 |
| 35 | 34 | -11.00 |
| 36 | 35 | -12.00 |
| 37 | 36 | -13.00 |
| 38 | 37 | -14.00 |
| 39 | 38 | -19.00 |
| 40 | 39 | -25.00 |

Implementation note: if ONCE points do not include channel/ring, GT row assignment cannot be exact from point records alone. The practical fallback is nearest-neighbor assignment by elevation angle against this table:

```python
row_top0 = argmin(abs(elevation_deg(point) - elevation_by_row_top0))
```

This is still better than a linear `linspace(-25, 15, 40)` because the Pandar40P vertical spacing is intentionally nonuniform.

## Waymo Open Dataset

Do not hard-code a single Waymo vertical angle table.

Waymo stores LiDAR as range images. Each row already corresponds to an inclination. For the top LiDAR, `LaserCalibration.beam_inclinations` can contain the exact nonuniform row inclinations. The Waymo docs also state that row `0` is the maximum inclination. For LiDARs whose `beam_inclinations` field is empty, the protobuf provides `beam_inclination_min` and `beam_inclination_max`; construct the table by interpolation for that range-image height.

Recommended extraction logic:

```python
def waymo_elevation_by_row_top0(laser_calibration, height):
    if len(laser_calibration.beam_inclinations) > 0:
        # Waymo stores inclinations in radians.
        elev = list(laser_calibration.beam_inclinations)
    else:
        lo = laser_calibration.beam_inclination_min
        hi = laser_calibration.beam_inclination_max
        elev = np.linspace(hi, lo, height).tolist()
    # Ensure row 0 is max inclination.
    if elev[0] < elev[-1]:
        elev = elev[::-1]
    return np.rad2deg(np.asarray(elev))
```

So Waymo's "table" is sequence/frame calibration data, not a repo-level constant.

## KITTI and KITTI-360: Velodyne HDL-64E

KITTI and KITTI-360 use Velodyne HDL-64E-class LiDAR. Their released point clouds generally store `x,y,z,reflectance` without ring/channel. Also, Velodyne HDL-64E manuals state that per-laser vertical corrections are stored in the unit-specific `db.xml` / calibration data.

Therefore, an exact channel table for KITTI/KITTI-360 should be generated from the original sensor calibration file if available. If the calibration XML is not available, use a nominal HDL-64E/S2 vertical FOV table only as an approximation.

Nominal fallback for row lookup, top-to-bottom, assuming a simple 64-row discretization over the commonly cited HDL-64E/S2 vertical FOV `[+2.0, -24.8]`:

| row_top0 | elevation_deg_nominal |
|---:|---:|
| 0 | 2.000 |
| 1 | 1.575 |
| 2 | 1.149 |
| 3 | 0.724 |
| 4 | 0.298 |
| 5 | -0.127 |
| 6 | -0.552 |
| 7 | -0.978 |
| 8 | -1.403 |
| 9 | -1.829 |
| 10 | -2.254 |
| 11 | -2.679 |
| 12 | -3.105 |
| 13 | -3.530 |
| 14 | -3.956 |
| 15 | -4.381 |
| 16 | -4.806 |
| 17 | -5.232 |
| 18 | -5.657 |
| 19 | -6.083 |
| 20 | -6.508 |
| 21 | -6.933 |
| 22 | -7.359 |
| 23 | -7.784 |
| 24 | -8.210 |
| 25 | -8.635 |
| 26 | -9.060 |
| 27 | -9.486 |
| 28 | -9.911 |
| 29 | -10.337 |
| 30 | -10.762 |
| 31 | -11.187 |
| 32 | -11.613 |
| 33 | -12.038 |
| 34 | -12.463 |
| 35 | -12.889 |
| 36 | -13.314 |
| 37 | -13.740 |
| 38 | -14.165 |
| 39 | -14.590 |
| 40 | -15.016 |
| 41 | -15.441 |
| 42 | -15.867 |
| 43 | -16.292 |
| 44 | -16.717 |
| 45 | -17.143 |
| 46 | -17.568 |
| 47 | -17.994 |
| 48 | -18.419 |
| 49 | -18.844 |
| 50 | -19.270 |
| 51 | -19.695 |
| 52 | -20.121 |
| 53 | -20.546 |
| 54 | -20.971 |
| 55 | -21.397 |
| 56 | -21.822 |
| 57 | -22.248 |
| 58 | -22.673 |
| 59 | -23.098 |
| 60 | -23.524 |
| 61 | -23.949 |
| 62 | -24.375 |
| 63 | -24.800 |

Use this fallback only when there is no access to the original `db.xml` and when exact ring reconstruction is not critical. For exact KITTI/KITTI-360 range images, infer row by nearest elevation to the calibration-derived table, or recover ring from scanline ordering / packet data if raw packets are available.

## Sources

- nuScenes point format and ring: nuScenes forum says LiDAR files store `(x, y, z, intensity, ring index)` and the devkit drops ring by default: https://forum.nuscenes.org/t/split-data-from-lidar/164
- HDL-32E vertical angles: Velodyne HDL-32E manual firing sequence table: https://manualsdump.com/en/manuals/velodyne_acoustics-hdl-32e/275446/13
- Waymo LiDAR range-image rows and beam inclinations: https://waymo.com/intl/fil/open/data/perception/
- Waymo protobuf calibration fields: `LaserCalibration.beam_inclinations`, `beam_inclination_min`, `beam_inclination_max`: https://protodoc.io/waymo-research/waymo-open-dataset/waymo.open_dataset
- ONCE LiDAR FOV and 40 beams: https://once-for-auto-driving.github.io/
- Hesai Pandar40P design channel table and exact-calibration caveat: https://device.report/manuals/pandar40p-user-manual-setup-data-structure-web-control
- KITTI sensor setup uses Velodyne HDL-64E: https://www.cvlibs.net/datasets/kitti/setup.php
- KITTI-360 sensor setup uses Velodyne HDL-64E: https://www.cvlibs.net/datasets/kitti-360/
- HDL-64E per-laser vertical correction comes from unit `db.xml`: https://www.manualslib.com/manual/533049/Velodyne-Hdl-64e.html?page=10
