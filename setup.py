from setuptools import setup, find_packages

setup(
    name="numpy-nn",              # Package name on PyPI
    version="1.0.0",             # Semantic versioning
    description="Neural Network Implementation in NumPy",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),     # Automatically find all Python packages
    install_requires=[           # Dependencies that pip will install
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.9",     # Minimum Python version
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)