"""Setup script for TSInfluenceScoring package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tsinfluencescoring",
    version="0.1.0",
    author="TSInfluenceScoring Contributors",
    description="Framework for selecting influential timestamps from time-series using attention-based models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
    ],
    extras_require={
        "notebook": [
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
)
