from setuptools import setup, find_packages

setup(
   name='PyAL',
   version='0.1',
   description='Active Learning in Python',
   author='Mirko Fischer',
   author_email='mirko.fischer@uni-muenster.de',
   packages=find_packages(where='src/PyAL'),
   package_dir={'': 'src'},
   install_requires=['matplotlib', 'scikit-learn', 'pandas', 'openpyxl', 'pyswarms', 'scipy'], #external packages as dependencies
)