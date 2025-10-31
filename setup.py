"""
abstractNN Setup Configuration
===============================

Installation script for the abstractNN library.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
VERSION = "0.1.0"

setup(
    name="abstractNN",
    version=VERSION,
    author="Guillaume Berthelot",
    author_email="contact@flyworthi.ai",
    description="Formal verification of neural networks using abstract interpretation and affine arithmetic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flyworthi/abstractNN",
    project_urls={
        "Bug Tracker": "https://github.com/flyworthi/abstractNN/issues",
        "Documentation": "https://abstractnn.readthedocs.io",
        "Source Code": "https://github.com/flyworthi/abstractNN",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "abstractnn-verify=abstractnn.cli:verify",
            "abstractnn-info=abstractnn.cli:info",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "neural networks",
        "formal verification",
        "abstract interpretation",
        "affine arithmetic",
        "robustness",
        "adversarial",
        "soundness",
        "deep learning",
        "AI safety",
    ],
)
