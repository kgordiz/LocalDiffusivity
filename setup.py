from setuptools import setup, find_packages

setup(
    name="local_diffusion_analysis",
    version="2.0",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    author="Kiarash Gordiz",
    description="A package for local diffusion analysis in MD simulations",
    license="MIT",
    keywords="molecular dynamics diffusion analysis",
    url="https://github.com/kgordiz/LocalDiffusivity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

