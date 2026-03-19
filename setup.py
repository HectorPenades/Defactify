"""Setup file for Defactify package."""

from setuptools import setup, find_packages

setup(
    name="defactify",
    version="0.1.0",
    description="Framework for AI image detection experiments",
    author="Hector",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "pandas>=1.0.0",
        "PyYAML>=5.0",
        "pillow>=8.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.0.0",
        "datasets>=2.0.0",
        "transformers>=4.0.0",
    ],
)
