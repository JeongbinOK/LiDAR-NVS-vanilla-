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

// LiDAR mode (A3.4): 1 intensity + 16 latent feature dims = 17.
// Camera mode keeps using channels 0..2 as RGB; channels 3..16 are zero-padded
// in upstream tests (colors_precomp widened from 3 to NUM_CHANNELS).
#define NUM_CHANNELS 17
#define BLOCK_X 16
#define BLOCK_Y 16

#endif