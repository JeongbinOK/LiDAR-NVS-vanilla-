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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// LiDAR mode: 1 intensity + L latent feature dims = NUM_CHANNELS.
// Default L=16 → NUM_CHANNELS=17. Override at build time with
//   LIDAR_LATENT_DIM=<L> pip install -e renderer/lidar_qgs_rasterizer
// which makes setup.py forward -DLIDAR_RASTER_NUM_CHANNELS=<L+1>.
//
// We use the indirection via LIDAR_RASTER_NUM_CHANNELS rather than defining
// NUM_CHANNELS directly on the command line because CUB headers (pulled in
// transitively by torch/cub) use the identifier `NUM_CHANNELS` as a template
// parameter — a global -DNUM_CHANNELS=N would replace the template name and
// break CUB. Defining inside this header (after CUB is already parsed) is safe.
#ifdef LIDAR_RASTER_NUM_CHANNELS
#define NUM_CHANNELS LIDAR_RASTER_NUM_CHANNELS
#else
#define NUM_CHANNELS 17
#endif
#define BLOCK_X 16
#define BLOCK_Y 16

#endif