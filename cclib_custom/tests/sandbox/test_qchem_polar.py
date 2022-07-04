from cclib_custom.sandbox.qchem_polar import QChemPolar
from cclib_custom.tests import _testdatadir

from cclib.io import ccopen


def test_qchem_polar() -> None:
    outputfile = _testdatadir / "Trp_bromine.out"
    assert outputfile.exists()

    job_default = ccopen(str(outputfile))
    data_default = job_default.parse()

    assert len(data_default.polarizabilities) == 1

    job_qcp = QChemPolar(str(outputfile))
    data_qcp = job_qcp.parse()

    assert len(data_qcp.polarizabilities) == 2
