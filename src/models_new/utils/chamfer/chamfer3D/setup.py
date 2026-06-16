from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
                            "nvcc": [
                                "-O3", 
                                "--compiler-bindir", "/usr/bin/gcc-12"  # nvcc가 사용할 호스트 컴파일러 강제 지정
                            ]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
