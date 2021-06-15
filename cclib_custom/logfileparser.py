"""logfileparser.py: Generic output file parser and related tools.

The abstract Logfile here doesn't delete any attributes.
"""

import logging
import sys

from cclib.parser.logfileparser import Logfile

from cclib_custom.data import ccDataKeepall


class LogfileKeepall(Logfile):
    """Abstract class for logfile objects.

    Subclasses defined by cclib:
        ADF, DALTON, GAMESS, GAMESSUK, Gaussian, Jaguar, Molpro, MOPAC, NWChem,
          ORCA, Psi, QChem

    This abstract Logfile doesn't delete any attributes.
    """

    def __init__(
        self,
        source,
        loglevel=logging.INFO,
        logname="Log",
        logstream=sys.stdout,
        datatype=ccDataKeepall,
        **kwds
    ):
        super().__init__(source, loglevel, logname, logstream, datatype, **kwds)
        # Prevent `optdone_as_list` from overriding the ccData type, while
        # still allowing the use of the `future` keyword
        self.datatype = datatype
