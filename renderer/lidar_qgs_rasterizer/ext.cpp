/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/channel_layout.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);

  // A3.3 — channel-layout constants. Python wrappers read these so the named
  // LiDAR view never drifts from the CUDA storage layout.
  m.attr("OUTPUT_CHANNELS")        = OUTPUT_CHANNELS;
  m.attr("NUM_CHANNELS")           = NUM_CHANNELS;
  m.attr("NORMAL_OFFSET")          = NORMAL_OFFSET;
  m.attr("DEPTH_OFFSET")           = DEPTH_OFFSET;
  m.attr("ALPHA_OFFSET")           = ALPHA_OFFSET;
  m.attr("CURVATURE_OFFSET")       = CURVATURE_OFFSET;
  m.attr("MIDDEPTH_OFFSET")        = MIDDEPTH_OFFSET;
  m.attr("LIDAR_INTENSITY_OFFSET") = LIDAR_INTENSITY_OFFSET;
  m.attr("LIDAR_LATENT_OFFSET")    = LIDAR_LATENT_OFFSET;
  m.attr("LIDAR_LATENT_DIM")       = LIDAR_LATENT_DIM;
}
