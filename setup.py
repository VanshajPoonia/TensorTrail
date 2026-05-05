"""Compatibility setup.py for older editable-install tooling."""

from setuptools import find_packages, setup


setup(
    name="tensortrail",
    version="0.1.0",
    description="TensorTrail: an educational NumPy neural network framework built from scratch.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=["numpy"],
)
