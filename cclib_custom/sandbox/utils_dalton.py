import re

import numpy as np

RE_ELEMENT = re.compile(r"([+-]?[0-9]*\.?[0-9]*|[+-]?\.[0-9]+)[EeDd]?([+-]?[0-9]+)?")


def parse_element_dalton(element):
    """Given a number that might appear in a DALTON output, especially one
    printed in a matrix, convert it to a float.

    The 'expt' regex specifically captures the exponent for when a 'D'
    isn't present in the number.

    Parameters
    ----------
    element : str

    Returns
    -------
    float
    """

    if any(x in element for x in ["e", "E", "d", "D"]):
        match = [(p, q) for (p, q) in RE_ELEMENT.findall(element) if p != "" if q != ""][0]
        return float("e".join(match))
    return float(element)


def dalton_parse_line(line):
    """Unpack a '@G' line from a DALTON output into a matrix."""

    # each field is 7 characters long
    xx, yy, zz = line[9:16], line[16:23], line[23:30]
    xy, yx, xz = line[30:37], line[37:44], line[44:51]
    zx, yz, zy = line[51:58], line[58:65], line[65:72]

    arr = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]], dtype=float)

    return arr


def parse_matrix_dalton(outputfile):
    """Parse a matrix from a DALTON output file. `outputfile` is the file
    object being read from.

    Since DALTON doesn't print every line or column if all its
    elements fall below a certain threshold (is there an option for
    this?), we will parse and form a "sparse" (dictionary of
    dictionaries) representation, and separately/later convert this
    sparse representation to dense array format.

    ...which is why there's no matrix passed as an argument.

    Parameters
    ----------
    outputfile : file handle
        The file handle should be at a position near the desired matrix.

    Returns
    -------
    dict
        A sparse matrix of floats packed into a 2D dictionary, where each
        keys are single indices into non-zero matrix elements.
    """

    spmat = dict()

    line = ""
    while "Column" not in line:
        line = next(outputfile)
    while "==== End of matrix output ====" not in line:
        chomp = line.split()
        if chomp == []:
            pass
        elif chomp[0] == "Column":
            colindices = list(map(int, chomp[1::2]))
        else:
            rowindex = int(chomp[0])
            if rowindex not in spmat:
                spmat[rowindex] = dict()
            for colindex, element in zip(colindices, chomp[1:]):
                spmat[rowindex][colindex] = parse_element_dalton(element)
        line = next(outputfile)
    return spmat
