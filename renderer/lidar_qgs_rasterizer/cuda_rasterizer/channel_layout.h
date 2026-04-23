/*
 * A3.3 — Channel-layout constants shared between CUDA kernels and host-side
 * Python bindings. Kept in its own header so ext.cpp (host C++) can include
 * the offsets without dragging in CUDA-only declarations from auxiliary.h.
 */

#ifndef CUDA_RASTERIZER_CHANNEL_LAYOUT_H_INCLUDED
#define CUDA_RASTERIZER_CHANNEL_LAYOUT_H_INCLUDED

#include "config.h"  // NUM_CHANNELS

// Storage layout — colors_precomp slot first (NUM_CHANNELS wide), then 3-D
// normal, then a fixed block of metadata (depth/alpha/distortion/...). Making
// the offsets NUM_CHANNELS-relative is what lets us widen the colors_precomp
// slot for LiDAR features (intensity + 16 latent dims) without re-laying out
// the metadata. Normal is always 3 components — do NOT confuse with NUM_CHANNELS.
#define NORMAL_DIM               3
#define NORMAL_OFFSET            NUM_CHANNELS
#define DEPTH_OFFSET             (NUM_CHANNELS + 3)
#define ALPHA_OFFSET             (NUM_CHANNELS + 4)
#define DISTORTION_OFFSET        (NUM_CHANNELS + 5)
#define MIDDEPTH_OFFSET          (NUM_CHANNELS + 6)
#define MEDIAN_WEIGHT_OFFSET     (NUM_CHANNELS + 7)
#define CURVATURE_OFFSET         (NUM_CHANNELS + 8)
#define CURV_DISTORTION_OFFSET   (NUM_CHANNELS + 9)
#define OUTPUT_CHANNELS          (NUM_CHANNELS + 10)

// LiDAR logical view — see auxiliary.h header comment for the full table.
// In LiDAR mode the colors_precomp slot is reinterpreted as
//   channel 0:           intensity
//   channels 1..L:       latent[0..L-1]   (L = LIDAR_LATENT_DIM)
// and the depth slot is interpreted as range. Other metadata channels keep
// their upstream meaning.
#define LIDAR_INTENSITY_OFFSET   0
#define LIDAR_LATENT_OFFSET      1
#define LIDAR_LATENT_DIM         (NUM_CHANNELS - 1)
#define LIDAR_RANGE_OFFSET       DEPTH_OFFSET
#define LIDAR_ALPHA_OFFSET       ALPHA_OFFSET
#define LIDAR_NORMAL_OFFSET      NORMAL_OFFSET
#define LIDAR_CURVATURE_OFFSET   CURVATURE_OFFSET

#endif
