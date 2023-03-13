import numpy as np

np_formatter = {"float_kind": lambda x: "{:14.8f}".format(x)}
np.set_printoptions(linewidth=240, formatter=np_formatter)  # type: ignore

from cclib_custom import LogfileKeepall, ccDataKeepall

from cclib.parser.qchemparser import QChem


class QChemFock(QChem, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        # aooverlaps
        if line.strip() == "Overlap Matrix":
            self.aooverlaps = QChem.parse_matrix(inputfile, self.nbasis, self.nbasis, 6)

        # if 'Final Alpha MO Eigenvalues' in line:
        #     self.moenergies = []
        #     # TODO should be nmo!
        #     moenergies = QChem.parse_matrix(inputfile, 1, self.nbasis, 6)
        #     self.moenergies.append(moenergies)

        # if 'Final Beta MO Eigenvalues' in line:
        #     moenergies = QChem.parse_matrix(inputfile, 1, self.nbasis, 6)
        #     self.moenergies.append(moenergies)

        if "Final Alpha Fock Matrix" in line:
            fockmat = QChem.parse_matrix(inputfile, self.nbasis, self.nbasis, 4)
            if not hasattr(self, "fockao"):
                self.fockao = []
            self.fockao.append(fockmat)

        if "Final Beta Fock Matrix" in line:
            fockmat = QChem.parse_matrix(inputfile, self.nbasis, self.nbasis, 4)
            self.fockao.append(fockmat)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("outputfilename", nargs="+")
    args = parser.parse_args()

    for outputfilename in args.outputfilename:
        print(outputfilename)
        job = QChemFock(outputfilename)
        data = job.parse()
        # print(dir(data))

        fock_matrix = data.fockao[0]
        moenergies = data.moenergies[0] / 27.21138505
        mocoeffs = data.mocoeffs[0].T
        C = mocoeffs
        S = data.aooverlaps

        F_MO = np.dot(C.T, np.dot(fock_matrix, C))
        F_AO = np.dot(C, np.dot(np.diag(moenergies), C.T))

        print(moenergies)
        print(F_MO)
        print(fock_matrix)
        print(F_AO)
