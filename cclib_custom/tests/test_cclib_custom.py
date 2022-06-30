"""
Unit and regression test for the cclib_custom package.
"""

import sys

# Import package, test suite, and other packages as needed
import cclib_custom

import pytest


def test_cclib_custom_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cclib_custom" in sys.modules
