from setuptools import find_packages, setup

__version__ = "0.1.0"
__author__ = "Kalen Cantrell"
__email__ = "kcantrel@ucsd.edu"

required_packages = [
    "numpy",
    "pandas",
    "cython",
    "biom-format",
    "scikit-bio",
    "scikit-learn",
    "scipy",
    "unifrac",
]

classes = """
    Development Status :: 3 - Alpha
    License :: OSI Approved :: BSD License
"""
classifiers = [s.strip() for s in classes.split("\n") if s]

with open("README.md") as f:
    long_description = f.read()


setup(
    name="amplicon_gpt",
    version=__version__,
    description="Deep Learning Method for Microbial Sequencing Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    classifiers=classifiers,
    packages=find_packages(),
    install_requires=required_packages,
    python_requires="~=3.9",
    license="BSD-3-Clause",
)
