# setup.py

from setuptools import setup, find_packages

setup(
    name="local_diffusion_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    author="Your Name",
    description="A package for local diffusion analysis in MD simulations",
    license="MIT",
    keywords="molecular dynamics diffusion analysis",
    url="https://github.com/yourusername/local_diffusion_analysis",
)

