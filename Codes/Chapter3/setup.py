from setuptools import setup
from Cython.Build import cythonize
import numpy

module_name = 'tm_sinusoid_trap_pack_save_every'

setup(
    name=module_name,
    ext_modules=cythonize(f"{module_name}.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)



#Run the line below in terminal


#python3 setup.py build_ext --inplace