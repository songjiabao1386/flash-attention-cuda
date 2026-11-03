import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

include_dir = os.path.abspath("include")

setup(
    name='custom_flash_attn',
    ext_modules=[
        CUDAExtension(
            'custom_flash_attn_cuda',
            [
                'src/attention_naive.cu',
                'src/attention_flash.cu',
            ],
            include_dirs=[include_dir],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_89,code=sm_89',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
