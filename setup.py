from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Amplemarket ML Solution Package'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="api", 
        version=VERSION,
        author="Chris Swart",
        author_email="cswart@outlook.com",
        description=DESCRIPTION,
        packages=find_packages()
)