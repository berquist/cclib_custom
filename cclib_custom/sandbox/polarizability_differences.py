import numpy as np

formatter = {
    "float_kind": lambda x: "{:8.4f}".format(x),
}
np.set_printoptions(linewidth=200, formatter=formatter)  # type: ignore

from cclib_custom import LogfileKeepall, ccDataKeepall

# import pandas as pd
# from cclib.io import ccopen
from cclib.parser.qchemparser import QChem


class QChemPolar(QChem, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        # Static polarizability from responseman/libresponse.
        if line.strip() == "Calculating the static polarizability using libresponse.":
            if not hasattr(self, "polarizabilities"):
                self.polarizabilities = []
            polarizability = []
            while line.strip() != "Static polarizability":
                line = next(inputfile)
            for _ in range(3):
                line = next(inputfile)
                polarizability.append(line.split())
            self.polarizabilities.append(np.array(polarizability))


def tensor_printer(tensor) -> None:
    assert len(tensor.shape) == 2
    assert tensor.shape[0] == tensor.shape[1]
    print(tensor)
    eigvals = np.linalg.eigvals(tensor)
    print(eigvals)
    print(np.average(eigvals))


if __name__ == "__main__":

    ions = (
        # 'fluorine',
        # 'chlorine',
        "bromine",
    )

    isotropic_polarizability_differences = dict()

    # parsing for responseman isn't in cclib yet
    # job_trp = ccopen('Trp.out')
    job_trp = QChemPolar("Trp.out")
    data_trp = job_trp.parse()
    # print('\n'.join(dir(data_trp)))

    polar_trp = data_trp.polarizabilities[0]
    pv_trp = np.linalg.eigvals(polar_trp)
    iso_trp = np.average(pv_trp)

    print("Trp")
    tensor_printer(polar_trp)

    for ion in ions:

        # job_ion = ccopen('{}.out'.format(ion))
        # job_super = ccopen('Trp_{}.out'.format(ion))
        job_ion = QChemPolar("{}.out".format(ion))
        job_super = QChemPolar("Trp_{}.out".format(ion))
        data_ion = job_ion.parse()
        data_super = job_super.parse()

        polar_ion = data_ion.polarizabilities[0]
        pv_ion = np.linalg.eigvals(polar_ion)
        iso_ion = np.average(pv_ion)

        polar_super = data_super.polarizabilities[0]
        pv_super = np.linalg.eigvals(polar_super)
        iso_super = np.average(pv_super)

        diff = iso_super - (iso_trp + iso_ion)

        isotropic_polarizability_differences[ion] = diff

        print(ion)
        tensor_printer(polar_ion)
        print("Trp + {} (supersystem)".format(ion))
        tensor_printer(polar_super)
        print("Trp + {} (separate)".format(ion))
        tensor_printer(polar_trp + polar_ion)

        job_super_almo = QChemPolar("almo_0/Trp_{}.out".format(ion))
        data_super_almo = job_super_almo.parse()
        # assert len(data_super_almo.polarizabilities) == 2
        polar_super_almo_mopropman = data_super_almo.polarizabilities[0]
        polar_super_almo_libresponse = data_super_almo.polarizabilities[1]
        print("Trp + {} (ALMO mopropman)".format(ion))
        tensor_printer(polar_super_almo_mopropman)
        print("Trp + {} (ALMO libresponse)".format(ion))
        tensor_printer(polar_super_almo_libresponse)

    # print(isotropic_polarizability_differences)

#    print(pd.DataFrame(isotropic_polarizability_differences))
