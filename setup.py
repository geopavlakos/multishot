from setuptools import setup, find_packages

print(find_packages())
setup(
    description='Regression as a package',
    name='regression',
    packages=find_packages()
)
