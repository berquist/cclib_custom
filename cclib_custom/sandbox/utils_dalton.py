import numpy as np


def parse_element_dalton(element):
    """Given a number that might appear in a DALTON output, especially one
    printed in a matrix, convert it to a float.
    """

    return float(element.lower().replace("d", "e"))


def dalton_parse_line(line):
    """Unpack a '@G' line from a DALTON output into a matrix."""

    # each field is 7 characters long
    xx, yy, zz = line[9:16], line[16:23], line[23:30]
    xy, yx, xz = line[30:37], line[37:44], line[44:51]
    zx, yz, zy = line[51:58], line[58:65], line[65:72]

    arr = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]], dtype=float)

    return arr
