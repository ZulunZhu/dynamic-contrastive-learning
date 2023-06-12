from distutils.core import setup,Extension
from Cython.Build import cythonize
import eigency

setup(
	ext_modules=cythonize(Extension(
		name='propagation',
		author='anonymous',
		version='0.0.1',
		sources=['propagation.pyx'],
		language='c++',
		extra_compile_args=["-std=c++14"],
		include_dirs=[".", "module-dir-name"] + eigency.get_includes()+["./eigen3"],
		#If you have installed eigen, you can configure your own path. When numpy references errors, you need to import its header file
		install_requires=['Cython>=0.2.15','eigency>=1.77'],
		packages=['little-try'],
		python_requires='>=3'
	))
)