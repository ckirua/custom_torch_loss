"""
Setup script for custom_loss_cpp Python extension
Build optimized C++ loss functions as a Python module
"""

import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CppExtension(
        name='custom_loss_cpp',
        sources=[
            'cpp/custom_loss_bindings.cpp',
            'cpp/custom_loss_functions.cpp',
        ],
        include_dirs=['cpp'],
        extra_compile_args=['-O3', '-march=native'] if os.name != 'nt' else ['/O2'],
    )
]

_readme = os.path.join(os.path.dirname(__file__), 'docs', 'README_PYTHON.md')
_long_desc = open(_readme, encoding='utf-8').read() if os.path.exists(_readme) else ''

setup(
    name='custom_loss_cpp',
    version='1.0.0',
    author='JDE65 (Original), Converted to C++',
    author_email='j.dessain@navagne.com',
    description='High-performance custom loss functions for financial forecasting',
    long_description=_long_desc,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['custom_loss']),
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
        'torch>=1.8.0',
    ],
    python_requires='>=3.14',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.14',
    ],
)
