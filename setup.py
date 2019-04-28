from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='embedding_dot',
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
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
