from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "fish2bird",
        sources=["fish2bird.pyx"],
        include_dirs=[numpy.get_include()],  # Include the NumPy headers
    ),
]
setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()], 
)
# -I/media/cyrille/Data/linux_program/conda-env/ros_env/lib/python3.9/site-packages/numpy/core/include