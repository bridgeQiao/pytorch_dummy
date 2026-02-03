from setuptools import setup
import os
import glob
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

torch_path = torch.__path__[0]
print("torch path: ", torch_path)

ext_modules = [
    CppExtension(
        'torch_dpu._C',
        include_dirs=[os.path.abspath('./')],
        sources=glob.glob('./torch_dpu/csrc/*.cpp')
        + glob.glob('./torch_dpu/csrc/*/*.cpp')
        + glob.glob('./torch_dpu/csrc/*/*/*.cpp'),
        library_dirs=[torch_path + '/lib'],
        extra_compile_args=['-O0', '-g'],
        runtime_library_dirs=[torch_path + '/lib'],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
