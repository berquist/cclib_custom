# -*- coding: utf-8 -*-

"""logfileparser.py: Generic output file parser and related tools.

The abstract Logfile here doesn't delete any attributes.
"""

import logging
import sys

from cclib.parser.logfileparser import Logfile

from . import ccDataKeepall


class LogfileKeepall(Logfile):
    """Abstract class for logfile objects.

    Subclasses defined by cclib:
        ADF, DALTON, GAMESS, GAMESSUK, Gaussian, Jaguar, Molpro, NWChem, ORCA,
          Psi, QChem

    This abstract Logfile doesn't delete any attributes.
    """

    def __init__(self, source, loglevel=logging.INFO, logname="Log",
                 logstream=sys.stdout, datatype=ccDataKeepall, **kwds):
        # Call the __init__ method of the superclass
        super(LogfileKeepall, self).__init__(source, loglevel, logname,
                                             logstream, datatype, **kwds)
