"""
cclib_custom
Customized quantum chemistry parsers for arbitrary attributes, built on top of cclib
"""
import sys

import versioneer

from setuptools import find_packages, setup

short_description = __doc__.split("\n")

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

with open("README.md") as handle:
    long_description = handle.read()


setup(
    name="cclib_custom",
    author="Eric Berquist",
    author_email="eric.berquist@gmail.com",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="BSD-3-Clause",
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,  # type: ignore
    install_requires=["cclib"],
    python_requires=">=3.6",
    # due to py.typed and mypy
    zip_safe=False,
)
