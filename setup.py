from setuptools import setup
import os,glob
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

torch_path = torch.__path__[0]

setup(
    name='dummy_backend',
    version='0.1',
    ext_modules=[CppExtension('dummy_backend',
                              include_dirs=[os.path.abspath('./')],
                              sources=glob.glob('./torch_dummy/**/*.cpp'),
                              library_dirs=[torch_path + '/lib'],
                              extra_compile_args=['-O0', '-g'],
                              runtime_library_dirs=[torch_path + '/lib'])],
    cmdclass={'build_ext': BuildExtension},
    )
