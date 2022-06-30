"""
cclib_custom
Customized quantum chemistry parsers for arbitrary attributes, built on top of cclib
"""

# Handle versioneer
from ._version import get_versions
from .data import ccDataKeepall
from .logfileparser import LogfileKeepall

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
