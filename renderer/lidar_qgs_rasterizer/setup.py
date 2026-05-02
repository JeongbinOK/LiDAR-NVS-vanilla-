#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_GLM_INC = os.path.join(_HERE, "third_party/glm/")

# Build-time latent feature dimension for LiDAR mode.
#   NUM_CHANNELS = 1 (intensity) + LIDAR_LATENT_DIM (latent)
# Override per build:  LIDAR_LATENT_DIM=32 pip install -e renderer/lidar_qgs_rasterizer
_latent_dim = int(os.environ.get("LIDAR_LATENT_DIM", "16"))
if _latent_dim < 0:
    raise ValueError(f"LIDAR_LATENT_DIM must be >= 0, got {_latent_dim}")
_num_channels = _latent_dim + 1
_channel_macro = [f"-DLIDAR_RASTER_NUM_CHANNELS={_num_channels}"]

setup(
    name="diff_quadratic_rasterization",
    packages=['diff_quadratic_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_quadratic_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-Xcompiler", "-fno-gnu-unique",
                    "-I" + _GLM_INC,
                ] + _channel_macro,
                "cxx": _channel_macro,
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
