from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


if torch.cuda.is_available():
    ext_modules=[
        CUDAExtension(
            name = 'embedding_dot_cuda', 
            sources = [
                'embedding_dot.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-march=native'],
                'nvcc': ['-O3', '-Xptxas', '-O3,-v'],
            },
        ),
    ]

else:
    ext_modules = []

setup(
    name='embedding_dot',
    ext_modules = ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    package_dir={"":"src"},
    packages=find_packages(where="src"),
)
