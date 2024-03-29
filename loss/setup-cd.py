import os
import runpy

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    include_dirs = ["_ext_src/chamfer_distance"]
    main_source = os.path.join("_ext_src/chamfer_distance", "ext.cpp")
    sources = [os.path.join("_ext_src/chamfer_distance", "knn_cpu.cpp")]
    source_cuda = [os.path.join("_ext_src/chamfer_distance", "knn.cu")]
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++14"]}
    define_macros = []

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            nvcc_args.extend(nvcc_flags_env.split(" "))

        CC = os.environ.get("CC", None)
        if CC is not None:
            CC_arg = "-ccbin={}".format(CC)
            if CC_arg not in nvcc_args:
                if any(arg.startswith("-ccbin") for arg in nvcc_args):
                    raise ValueError("Inconsistent ccbins")
                nvcc_args.append(CC_arg)

        extra_compile_args["nvcc"] = nvcc_args

    ext_modules = [
        extension(
            "chamfer_distance._ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if os.getenv("NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)


else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension


setup(
    name="chamfer_distance",
    description="Pytorch Chamfer distance",
    packages=find_packages(),
    package_data={'_ext_src/chamfer_distance': ['*.cu', '*.cuh', '*.h']},
    install_requires=[],
    extras_require={
        "dev": ["black", "flake8", "isort"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
