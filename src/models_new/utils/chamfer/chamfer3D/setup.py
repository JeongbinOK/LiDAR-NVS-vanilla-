from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --compiler-bindir needs cc1plus (the g++ package, not just gcc); only pin it
# when this machine actually has a full g++-12 install.
_nvcc_flags = ["-O3"]
if Path("/usr/bin/g++-12").is_file():
    _nvcc_flags += ["--compiler-bindir", "/usr/bin/gcc-12"]  # nvcc가 사용할 호스트 컴파일러 강제 지정

setup(
    name="chamfer_3D",
    ext_modules=[
        CUDAExtension(
            "chamfer_3D",
            [
                "/".join(__file__.split("/")[:-1] + ["chamfer_cuda.cpp"]),
                "/".join(__file__.split("/")[:-1] + ["chamfer3D.cu"]),
            ],
            extra_compile_args={
                            "cxx": ["-O3"],
                            "nvcc": _nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
