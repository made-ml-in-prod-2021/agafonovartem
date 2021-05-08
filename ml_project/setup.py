from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(include=['ml_project', 'ml_project.*']),
    version='0.1.0',
    description='A short description of the project.',
    author='Artem Agafonov',
    license='MIT',
)
