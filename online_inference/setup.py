from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(include=['online_inference', 'online_inference.*']),
    version='0.1.0',
    author='Artem Agafonov',
    license='MIT',
)
