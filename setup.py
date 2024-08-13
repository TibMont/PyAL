from setuptools import setup, find_packages

PACKAGES = find_packages()

INSTALL_REQUIRES= [
    "matplotlib", 
    "scikit-learn", 
    "pandas", 
    "openpyxl", 
    "pyswarms", 
    "scipy"
]

setup(
   name="PyAL",
   version="0.1",
   description="Active Learning in Python",
   author="Mirko Fischer",
   author_email="mirko.fischer@uni-muenster.de",
   packages=PACKAGES,
   install_requires=INSTALL_REQUIRES, #external packages as dependencies
)