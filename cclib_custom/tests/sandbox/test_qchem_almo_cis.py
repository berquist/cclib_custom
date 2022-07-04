from cclib_custom.sandbox.qchem_almo_cis import QChemALMOCIS
from cclib_custom.tests import _testdatadir

from cclib.io import ccopen


def test_qchem_almo_cis() -> None:
    outputfile = _testdatadir / "almo_cis_noct_2_2.45.out"
    assert outputfile.exists()

    job_default = ccopen(str(outputfile))
    data_default = job_default.parse()

    # assert not hasattr(data_default, "etenergies")
    # assert not hasattr(data_default, "etoscs")
    assert not hasattr(data_default, "matrices")

    job_almo = QChemALMOCIS(str(outputfile))
    data_almo = job_almo.parse()

    # assert hasattr(data_almo, "etenergies")
    # assert hasattr(data_almo, "etoscs")
    assert hasattr(data_almo, "matrices")
    ref_keys = {
        "J using almo-motran",
        "K half-transformed (oo)",
        "Almo K",
        "superFS",
        "superS",
        "A",
    }
    assert set(data_almo.matrices.keys()) == ref_keys
