/*
 * Spherical (panoramic LiDAR) projection helpers.
 *
 * QGS-Flow A3.2.  Author: J. (2026-04-18).
 *
 * Convention
 * ----------
 *   p_view = (x, y, z)  in sensor frame (x=right, y=forward, z=up).
 *   r      = ||p_view||             ≥ 0
 *   az     = atan2(x, y)            ∈ [-π, π]   (0 = +y forward, +π/2 = +x right)
 *   el     = atan2(z, sqrt(x² + y²)) ∈ [-π/2, π/2]
 *
 *   cam_intr[4] (re-purposed from camera mode):
 *     [0] el_min_rad           — bottom of vertical FOV
 *     [1] el_max_rad           — top    of vertical FOV
 *     [2] w_per_rad_az = W / (2π)
 *     [3] h_per_rad_el = H / (el_max - el_min)
 *
 * Pixel coords:
 *   u = (az + π) * w_per_rad_az             ∈ [0, W)
 *   v = (el - el_min) * h_per_rad_el        ∈ [0, H)
 *
 * Azimuth wraparound is the caller's responsibility — when the bbox crosses ±π,
 * emit two image-space rectangles instead of one.
 *
 * Header-only: include this from forward.cu / preprocess kernels.
 * The same math is mirrored in `tests/_spherical_python_ref.py` for validation.
 */

#pragma once

#include <cmath>

#ifdef __CUDACC__
#define SPH_HOSTDEV __host__ __device__ __forceinline__
#else
#define SPH_HOSTDEV inline
#endif

namespace qgs_lidar
{

constexpr float PI_F        = 3.14159265358979323846f;
constexpr float TWO_PI_F    = 6.28318530717958647692f;
constexpr float HALF_PI_F   = 1.57079632679489661923f;
constexpr float SPH_EPS     = 1e-8f;

/* ------------------------------------------------------------------ */
/* Project (x,y,z) → (r, az, el).                                      */
/* ------------------------------------------------------------------ */
SPH_HOSTDEV
void project_to_sphere(float x, float y, float z,
                       float& r, float& az, float& el)
{
    r  = sqrtf(x * x + y * y + z * z);
    // az: 0 = +y (forward), grows clockwise toward +x (right).
    //     atan2(x, y) gives the desired convention directly.
    az = atan2f(x, y);
    const float xy = sqrtf(x * x + y * y);
    el = atan2f(z, xy);
}

/* ------------------------------------------------------------------ */
/* Map (az, el) → (u, v) pixel coordinates given LiDAR intrinsics.    */
/* ------------------------------------------------------------------ */
SPH_HOSTDEV
void spherical_to_pixel(float az, float el,
                        const float* cam_intr,
                        float& u, float& v)
{
    const float el_min       = cam_intr[0];
    const float w_per_rad_az = cam_intr[2];
    const float h_per_rad_el = cam_intr[3];
    u = (az + PI_F) * w_per_rad_az;
    v = (el - el_min) * h_per_rad_el;
}

/* ------------------------------------------------------------------ */
/* Frustum test: is the primitive within the LiDAR vertical FOV?      */
/* (Azimuth has full 360° coverage so no horizontal frustum to test.) */
/*                                                                    */
/* near/far range gating is also done here.                            */
/* ------------------------------------------------------------------ */
SPH_HOSTDEV
bool is_in_spherical_frustum(float x, float y, float z,
                             const float* cam_intr,
                             float r_near, float r_far)
{
    float r, az, el;
    project_to_sphere(x, y, z, r, az, el);
    if (r < r_near || r > r_far) return false;
    const float el_min = cam_intr[0];
    const float el_max = cam_intr[1];
    return (el >= el_min) && (el <= el_max);
}

/* ------------------------------------------------------------------ */
/* Spherical AABB of a ball of radius R_eff centred at p_view.        */
/*                                                                    */
/* For a primitive of effective extent R_eff (e.g. sigma·max(scale)),  */
/* compute the angular bounding box subtended at the sensor origin.    */
/*                                                                    */
/* Returns: az_min, az_max, el_min, el_max  (all radians).             */
/*                                                                    */
/* NOTE: az can wrap around ±π. If az_min > az_max in the result, the  */
/* caller must split the rect at the seam (az_min..π and -π..az_max).  */
/* The split itself is handled outside; we set the `wrapped` flag.     */
/* ------------------------------------------------------------------ */
SPH_HOSTDEV
void aabb_spherical(float x, float y, float z,
                    float R_eff,
                    float& az_min, float& az_max,
                    float& el_min, float& el_max,
                    bool& wrapped)
{
    const float r = sqrtf(x * x + y * y + z * z);
    wrapped = false;
    if (r < SPH_EPS) {
        // Primitive at sensor origin → covers full sphere.
        az_min = -PI_F;  az_max = PI_F;
        el_min = -HALF_PI_F; el_max = HALF_PI_F;
        return;
    }
    // Half-angle subtended by the ball:  sin(θ_half) = R_eff / r.
    // Clamp to π/2 (ball encloses the origin).
    const float sin_half = fminf(R_eff / r, 1.0f);
    const float theta_half = asinf(sin_half);

    // Centre angles
    float az_c, el_c;
    {
        float r_unused;
        project_to_sphere(x, y, z, r_unused, az_c, el_c);
    }

    // Elevation extent: simple ±θ_half (clamped to [-π/2, π/2]).
    el_min = fmaxf(el_c - theta_half, -HALF_PI_F);
    el_max = fminf(el_c + theta_half,  HALF_PI_F);

    // Azimuth extent of the spherical cap. The exact longitude half-width is
    // asin(sin(theta_half) / cos(el_c)) unless the cap reaches a pole, in
    // which case every azimuth has at least one supported ray.
    const float cos_el = fmaxf(fabsf(cosf(el_c)), SPH_EPS);
    const bool full_circle = (sin_half >= cos_el);
    const float az_half = full_circle ? PI_F : asinf(fminf(sin_half / cos_el, 1.0f));

    az_min = az_c - az_half;
    az_max = az_c + az_half;

    // If the cap reaches a pole, no wrap; just cover the full azimuth range.
    if (full_circle) {
        az_min = -PI_F;  az_max = PI_F;
        return;
    }

    // Wraparound: bring az_min/az_max back into [-π, π] and flag the wrap.
    if (az_min < -PI_F) {
        az_min += TWO_PI_F;
        wrapped = true;
    }
    if (az_max > PI_F) {
        az_max -= TWO_PI_F;
        wrapped = true;
    }
    // After wrap, if az_min > az_max, the box is `[az_min, π] ∪ [-π, az_max]`.
}

}  // namespace qgs_lidar
