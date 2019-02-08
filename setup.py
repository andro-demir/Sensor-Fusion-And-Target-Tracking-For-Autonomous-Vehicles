from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["kalmanFilter.pyx",
                             "readData.pyx"]))

'''
    Run the built-in extension
    $python setup.by build_ext -i  
'''