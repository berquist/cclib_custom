"""logfileparser.py: Generic output file parser and related tools.

The abstract Logfile here doesn't delete any attributes.
"""

import logging
import sys

from cclib_custom.data import ccDataKeepall

from cclib.parser.logfileparser import Logfile


class LogfileKeepall(Logfile):
    """Abstract class for logfile objects.

    Unlike cclib.parser.logfileparser.Logfile, this abstract Logfile doesn't
    delete any attributes.
    """

    def __init__(
        self,
        source,
        loglevel=logging.INFO,
        logname="Log",
        logstream=sys.stdout,
        datatype=ccDataKeepall,
        **kwds,
    ):
        super().__init__(source, loglevel, logname, logstream, datatype, **kwds)
        # Prevent `optdone_as_list` from overriding the ccData type, while
        # still allowing the use of the `future` keyword
        self.datatype = datatype
