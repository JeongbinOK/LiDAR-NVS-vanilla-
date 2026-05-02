# LiDAR QGS Renderer Math Contract

This contract combines the QGS quadric/geodesic surface model with the
panoramic LiDAR range-map renderer used in this repository. CUDA and Python
tests should be compared against these equations before tolerances are changed.

Reference baselines:

- QGS paper and official CUDA: local quadric intersection, geodesic Gaussian
  support, alpha falloff, and front-to-back compositing.
- GS-LiDAR paper and official CUDA: panoramic LiDAR range-map rays and
  alpha-composited range/intensity/raydrop-style rendering.

## Sensor Model

Sensor frame axes are locked as `x=right, y=forward, z=up`.

For range-map pixel center `(u, v)` with width `W`, height `H`, and intrinsics
`[el_min, el_max, W / 2pi, H / (el_max - el_min)]`:

```text
az = (u + 0.5) / (W / 2pi) - pi
el = (v + 0.5) / (H / (el_max - el_min)) + el_min
d  = (sin(az) cos(el), cos(az) cos(el), sin(el))
```

The ray is unit length:

```text
||d||^2 = sin^2(az) cos^2(el) + cos^2(az) cos^2(el) + sin^2(el) = 1
```

Therefore a sensor-origin hit `p=t*d` has Euclidean LiDAR range `||p||=t`.
This is why the ray-root parameter is the rendered range sample; it is not a
mean depth or middepth convention.

Projection of a sensor-frame point `p=(x,y,z)` is:

```text
r  = ||p||
az = atan2(x, y)
el = atan2(z, sqrt(x^2 + y^2))
u  = (az + pi) * W / 2pi
v  = (el - el_min) * H / (el_max - el_min)
```

The supported frustum gate is vertical FOV plus `[r_near, r_far]`. Azimuth is a
full 360 degree wrap, so primitives crossing `-pi/pi` must remain visible.

## Local QGS Surface

The local Gaussian frame uses the QGS implicit surface:

```text
kx*x^2 + ky*y^2 - kz*z = 0
```

where:

```text
kx = sign(s1) / |s1|^2
ky = sign(s2) / |s2|^2
kz = 1 / s3
```

Equivalently:

```text
z = s3 * (kx*x^2 + ky*y^2)
```

`s1` and `s2` are signed tangent support scales. `s3` is the quadratic height
amplitude used by the geodesic metric and the implicit `z` coefficient; it is
not another tangent support radius. The CUDA precompute stores
`rscale_o={1/s1^2, 1/s2^2, 1/s3, opacity}` and keeps signs for `s1,s2`
separately.

The local normal before orientation is the implicit gradient:

```text
n_raw = (2*kx*x, 2*ky*y, -kz)
n_local = normalize(n_raw)
```

For LiDAR rendering the emitted normal is oriented against the incoming ray:

```text
if dot(n_local, d_local) > 0: n_local = -n_local
```

This gradient exactly matches the normal form implied by the QGS CUDA
intersection coefficients.

## Ray Intersection

The sensor ray is transformed by `view2gaussian`:

```text
o = view2gaussian translation
d = view2gaussian rotation * ray_direction
```

Substitution of `p=o+t*d` into the implicit surface gives:

```text
A = kx * dx^2 + ky * dy^2
B = 2*kx*ox*dx + 2*ky*oy*dy - kz*dz
C = kx*ox^2 + ky*oy^2 - kz*oz
```

The forward root equation is:

```text
A*t^2 + B*t + C = 0
```

If `|A|` is small, the supported linear root is `t=-C/B` when `|B|` is not
small. Otherwise candidates are the real roots of the quadratic. A candidate is
eligible only if:

```text
t > 0
r_near <= t <= r_far
QGS geodesic support accepts the local hit point
alpha survives clamp/skip thresholds
```

Degenerate all-zero equations, negative discriminants, roots behind the sensor,
unsupported points outside the geodesic support, and alpha values below the CUDA
skip threshold are no-hit cases. Tangent roots are valid forward hits but are
excluded from strict gradient tests.

## Geodesic Support and Alpha

For a local hit point `p=(x,y,z)`, define its tangent-plane polar quantities:

```text
l = sqrt(x^2 + y^2)
cos2 = x^2 / (x^2 + y^2)
sin2 = y^2 / (x^2 + y^2)
```

At direction angle `theta`, the quadratic curve coefficient is:

```text
a(theta) = s3 * (kx*cos^2(theta) + ky*sin^2(theta))
```

The QGS geodesic length from the center to tangent radius `l` is Eq. 10:

```text
u = 2*a*l
S(l,a) = log(sqrt(u^2 + 1) + u) / (4*a) + 0.5*l*sqrt(u^2 + 1)
```

For `a -> 0`, `S(l,a) -> l`, so the metric continuously reduces to the
tangent-plane Euclidean / 2DGS-like plane case.

The directional tangent support variance is:

```text
r0(theta)^2 = 1 / (cos^2(theta)/|s1|^2 + sin^2(theta)/|s2|^2)
```

The hit is within support when:

```text
S(l,a)^2 <= sigma^2 * r0(theta)^2
```

Alpha uses the same geodesic Gaussian:

```text
power = -S(l,a)^2 / (2*r0(theta)^2)
alpha = min(0.99, opacity * exp(power))
skip if alpha < 1/255
```

This replaces a simple ellipse cutoff. The ellipse only appears as the
`a -> 0` limiting case.

## LiDAR Adaptation

The pinhole QGS derivation does not depend on how the ray direction was
generated once the ray is in camera/sensor coordinates. For LiDAR, only the ray
generator changes from a pinhole pixel ray to the spherical ray above. The same
local transform, quadric root solve, geodesic support, and alpha computation
then apply.

Because `ray_direction` is unit length, sorting by root `t` is sorting by LiDAR
range. The range-map pixel itself lives in panoramic angle space, matching the
GS-LiDAR convention.

## Blending

Hits are composited front to back by increasing `t`:

```text
weight_i = T_i * alpha_i
T_{i+1}  = T_i * (1 - alpha_i)
alpha_accum = sum_i weight_i
range       = sum_i weight_i * t_i
intensity   = sum_i weight_i * intensity_i
latent      = sum_i weight_i * latent_i
normal      = sum_i weight_i * normal_i
curvature   = sum_i weight_i * curvature_i
```

This is compatible with both QGS and GS-LiDAR alpha compositing.

Here `curvature_i` is the QGS signed Gaussian curvature, not an absolute
curvature magnitude. With height coefficients
`cx=s3*kx` and `cy=s3*ky`, CUDA computes
`curvature_i = 4*cx*cy / (1 + 4*((cx*x)^2 + (cy*y)^2))^2`. Saddle patches
therefore keep negative curvature.

`range` is the alpha-weighted expected range. Divide by `alpha_accum` when a
physical single-return estimate is needed. `middepth` is a separate
transmittance-crossing / first-return-like convention and must not be silently
treated as the alpha-weighted mean.

## Tiling and AABB

The original QGS screen-space pinhole tiling is not the LiDAR answer by itself.
LiDAR needs a spherical/panoramic tile envelope. A tile candidate set is
mathematically sufficient if it is a superset of every primitive that could pass
the per-pixel root, geodesic support, range/FOV, and alpha tests.

The current CUDA spherical ball/AABB path should be read as a conservative
candidate generator that relies on exact per-pixel rejection later. It must not
be documented as an exact QGS geodesic support bound unless a separate proof or
test establishes that equality.

## Backward Contract

The oracle for differentiable regions is PyTorch double autograd. CUDA backward
must match it for `mean`, `scale`, `opacity`, and `view2gaussian`.

For quaternion parameters, raw component gradients are not the public
acceptance surface. Tests should compare gradients projected onto unit
quaternion tangent perturbations, e.g. small local axis-angle deltas.

Strict gradient tests exclude discontinuities caused by root changes, support
cutoff crossings, FOV/range gates, azimuth seam bin changes, alpha clamp
saturation, and alpha skip thresholds. Forward stability is still required for
those cases.

## Current Implementation Checks

The CUDA renderer should be checked against the following contract points when
it is next edited:

- `rscale_o.z` is `1/s3`, not `sign(s3)/s3^2`.
- Root coefficients use `kx`, `ky`, and `kz` exactly as above.
- Support and alpha use QGS Eq. 10 geodesic length, not a plain ellipse cutoff.
- The spherical candidate AABB is only a conservative tile superset unless an
  exact geodesic-support envelope is proven.
