from setuptools import setup
import os,glob
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

torch_path = torch.__path__[0]

setup(
    name='torch_dpu',
    version='0.1',
    license="BSD License",
    ext_modules=[CppExtension('torch_dpu._C',
                              include_dirs=[os.path.abspath('./')],
                              sources=glob.glob('./torch_dpu/csrc/**.cpp'),
                              library_dirs=[torch_path + '/lib'],
                              extra_compile_args=['-O0', '-g'],
                              runtime_library_dirs=[torch_path + '/lib'])],
    cmdclass={'build_ext': BuildExtension},
    )
