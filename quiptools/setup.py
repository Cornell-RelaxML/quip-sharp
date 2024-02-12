from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name='quiptools_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'quiptools_cuda',
            ['quiptools_wrapper.cpp', 'quiptools.cu', 'quiptools_e8p_gemv.cu'],
            extra_compile_args={
                'cxx': ['-g', '-lineinfo'],
                'nvcc': ['-O2', '-g', '-Xcompiler', '-rdynamic', '-lineinfo']
            })
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
