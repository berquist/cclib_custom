from cclib_custom import LogfileKeepall, ccDataKeepall

from cclib.parser.qchemparser import QChem


class QChemALMOCIS(QChem, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

        self.matrices = dict()

        matrix_headers_1 = [
            "Coulomb J",
            # why are there two of these? wouldn't need the list
            # append otherwise
            "K half-transformed",
            "Exchange K",
            "superFS",
            "superS",
            "A",
        ]

        matrix_headers_2 = [
            "J using almo-motran",
            # first half transform, IJmunu
            "K half-transformed (oo)",
            # second half transform, IJAB
            "Almo K",
            "superFS",
            "superS",
            "A",
        ]

        self.matrix_headers = matrix_headers_1 + matrix_headers_2

    def extract(self, inputfile, line: str) -> None:
        super().extract(inputfile, line)

        # check $rem for the local_cis option
        # 1 -> no RI (incorrect) version
        # 2 -> RI (correct) version
        #
        # This actually doesn't work, since this is a rem keyword, and the
        # $rem section is parsed by the main QChem parser.  For things inside
        # sections that are already parsed in parent parsers, if they call
        # `next(inputfile)` over the line(s) of interest, there's no way to
        # access it.
        #
        # if "local_cis" in line:
        #     self.set_attribute("local_cis", int(line.split()[-1]))

        if any(line.strip() == x for x in self.matrix_headers):
            header = line.strip()
            if header not in self.matrices:
                self.matrices[header] = []
            # append the matrix
