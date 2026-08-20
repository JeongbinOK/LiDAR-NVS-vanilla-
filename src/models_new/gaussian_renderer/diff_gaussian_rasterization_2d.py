import os
import sys
from pathlib import Path
from typing import NamedTuple
import torch.nn as nn
import torch

environment_bin_path = Path(sys.executable).resolve().parent
environment_root = environment_bin_path.parent
path_entries = os.environ.get("PATH", "").split(os.pathsep)
if str(environment_bin_path) not in path_entries:
    # Absolute interpreter invocations do not necessarily put their conda
    # environment's Ninja/NVCC executables on PATH.
    os.environ["PATH"] = os.pathsep.join([
        str(environment_bin_path), *path_entries,
    ])
if (environment_bin_path / "nvcc").is_file():
    # Set this before importing cpp_extension: CUDA_HOME is detected at module
    # import time. This avoids accidentally selecting an older system toolkit.
    os.environ["CUDA_HOME"] = str(environment_root)
# Build for the GPU actually present rather than a hardcoded architecture:
# an sm_89 cubin does not run on sm_80, and compute_89 PTX cannot JIT backward
# to older hardware, so a fixed value silently breaks on a different machine.
if torch.cuda.is_available():
    _cuda_major, _cuda_minor = torch.cuda.get_device_capability()
else:
    # No visible device (docs build, CPU-only import): fall back to the torch
    # default rather than guessing an architecture.
    _cuda_major, _cuda_minor = None, None
if _cuda_major is not None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cuda_major}.{_cuda_minor}"
    _gencode_flags = [
        f"-gencode=arch=compute_{_cuda_major}{_cuda_minor},"
        f"code=sm_{_cuda_major}{_cuda_minor}"
    ]
else:
    _gencode_flags = []
# torch.utils.cpp_extension feeds $CC to nvcc as -ccbin for every .cu source
# in this extension, so CC needs the same complete C++ toolchain as CXX/
# CUDAHOSTCXX (cc1plus, from the g++ package) even though it is nominally the
# C compiler; gcc-12 alone (no g++-12) has the driver but not the backend nvcc
# needs, and fails with "cannot execute 'cc1plus'".
for compiler_variable, compiler_path in (
    ("CC", "/usr/bin/gcc-12"),
    ("CXX", "/usr/bin/g++-12"),
    ("CUDAHOSTCXX", "/usr/bin/g++-12"),
):
    if Path("/usr/bin/g++-12").is_file() and Path(compiler_path).is_file():
        os.environ[compiler_variable] = compiler_path
# nvcc rejects a --compiler-bindir that does not exist, so only pin the host
# compiler when this machine actually has it. --compiler-bindir compiles the
# C++ host code nvcc generates from these .cu files, which needs cc1plus (the
# g++ package); a gcc-only install (gcc present, g++ absent) has the driver
# but not the C++ backend, so check for g++-12 specifically.
_host_compiler_flags = (
    ["--compiler-bindir", "/usr/bin/gcc-12"]
    if Path("/usr/bin/g++-12").is_file() else []
)

from torch.utils.cpp_extension import load

parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diff-gaussian-rasterization-2d")
repository_root = Path(__file__).resolve().parents[3]
extension_build_dir = (
    repository_root
    / ".cache"
    / "torch_extensions"
    / "diff_gaussian_rasterization"
)
extension_build_dir.mkdir(parents=True, exist_ok=True)

# Keep the binary coupled to this worktree's source tree. Passing an explicit
# build_directory prevents a machine-level TORCH_EXTENSIONS_DIR from silently
# loading a rasterizer compiled by another checkout. cpp_extension.load uses
# Ninja's dependency checks, so unchanged runs reuse the binary and edited
# CUDA/C++ sources are rebuilt automatically.
_C = load(
    name='diff_gaussian_rasterization',
    build_directory=str(extension_build_dir),
    extra_cuda_cflags=[
        "-I " + os.path.join(parent_dir, "third_party/glm/"),
        "-g",
        *_gencode_flags,
        *_host_compiler_flags,
    ],
    sources=[
        os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(parent_dir, "rasterize_points.cu"),
        os.path.join(parent_dir, "ext.cpp")],
    verbose=True)


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
        means3D,
        means2D,
        sh,
        colors_precomp,
        features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        mask,
        raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        mask,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            means2D,
            sh,
            colors_precomp,
            features,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            mask,
            raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            features,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            mask,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.vfov[0],
            raster_settings.vfov[1],
            raster_settings.hfov[0],
            raster_settings.hfov[1],
            raster_settings.row_to_theta,
            raster_settings.scale_factor
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, contrib, color, feature, depth, T, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, contrib, color, feature, depth, T, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, features, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, contrib)
        return contrib, color, feature, depth, 1 - T, radii

    @staticmethod
    def backward(ctx, grad_out_contrib, grad_out_color, grad_out_feature, grad_depth, grad_alpha, _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, features, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, contrib = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                features,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_depth,
                grad_alpha,
                grad_out_feature,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                contrib,
                raster_settings.debug,
                raster_settings.vfov[0],
                raster_settings.vfov[1],
                raster_settings.hfov[0],
                raster_settings.hfov[1],
                raster_settings.row_to_theta,
                raster_settings.scale_factor)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_features,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    vfov: tuple
    hfov: tuple
    row_to_theta: torch.Tensor
    scale_factor: float


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, features=None, scales=None, rotations=None, cov3D_precomp=None, mask=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        device = means3D.device
        dtype = means3D.dtype

        if shs is None:
            shs = torch.empty((0,), device=device, dtype=dtype)
        else:
            shs = shs.contiguous()
        if colors_precomp is None:
            colors_precomp = torch.empty((0,), device=device, dtype=dtype)
        else:
            colors_precomp = colors_precomp.contiguous()
        if features is None:
            features = torch.empty_like(means3D[..., :0])

        if scales is None:
            scales = torch.empty((0,), device=device, dtype=dtype)
        else:
            scales = scales.contiguous()
        if rotations is None:
            rotations = torch.empty((0,), device=device, dtype=dtype)
        else:
            rotations = rotations.contiguous()
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty((0,), device=device, dtype=dtype)
        else:
            cov3D_precomp = cov3D_precomp.contiguous()
        if mask is None:
            mask = torch.ones((means3D.shape[0],), device=device, dtype=torch.bool)
        else:
            mask = mask.to(device=device, dtype=torch.bool).reshape(-1).contiguous()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            features,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            mask,
            raster_settings,
        )
