from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = 'nms_module',
      ext_modules = cythonize('cpu_nms.pyx'),
      )
