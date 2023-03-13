from cclib_custom import LogfileKeepall, ccDataKeepall
from cclib_custom.sandbox.utils_dalton import dalton_parse_line, parse_element_dalton

import numpy as np
from cclib.io import ccopen
from cclib.parser.daltonparser import DALTON
from cclib.parser.gaussianparser import Gaussian
from cclib.parser.nwchemparser import NWChem
from cclib.parser.orcaparser import ORCA
from cclib.parser.qchemparser import QChem
from cclib.parser.utils import convertor


def g_eigvals(g_matrix):
    """Return the eigenvalues of a g-matrix."""

    return np.sqrt(np.linalg.eigvalsh(np.dot(g_matrix.T, g_matrix)))


def g_tensor_populate_eigvals(d):
    """Assume that all values are consistently in ppm."""

    keys = (
        "gc_ppm",
        "gc_1e_ppm",
        "gc_2e_ppm",
        "oz_soc_ppm",
        "oz_soc_1e_ppm",
        "oz_soc_2e_ppm",
    )

    for k in keys:
        if k in d:
            d[k + "_eig"] = g_eigvals(d[k])

    return d


class CFOURNMR(LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        # no super call, since a CFOUR parser doesn't exist yet

        if "SCF has converged." in line:
            if not hasattr(self, "scfenergies"):
                self.scfenergies = []
            while "E(SCF)=" not in line:
                line = next(inputfile)
            scfenergy = convertor(float(line.split()[1]), "hartree", "eV")
            self.scfenergies.append(scfenergy)

        if "Total shielding tensor" in line:
            self.skip_lines(inputfile, ["d", "b", "header", "b"])
            line = next(inputfile)
            self.nmr_shielding_tensors = []
            num_old = ""
            num_new = ""
            while "Calculation of total derivative of f" not in line:
                if line.strip() != "":
                    sline = line.strip()
                    # LI#1    x   92.694602    0.000000    0.000000
                    atomsym = sline[:2]
                    assert sline[2:3] == "#"
                    num_new = sline[3:8]
                    coord = sline[8:9]
                    v1 = sline[9:21]
                    v2 = sline[21:33]
                    v3 = sline[33:45]
                    # chomp = line.split()
                    # assert len(chomp) == 6
                    # num_new = chomp[1]
                    if num_new != num_old:
                        if num_old != "":
                            tensor = np.array(tensor, dtype=float)
                            self.nmr_shielding_tensors.append(tensor)
                        tensor = []
                    # tensor.append(chomp[3:])
                    tensor.append([v1, v2, v3])
                    num_old = num_new
                line = next(inputfile)
            tensor = np.array(tensor, dtype=float)
            self.nmr_shielding_tensors.append(tensor)


class DALTONNMR(DALTON, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        if "ABACUS - CHEMICAL SHIELDINGS" in line:
            self.nmr_shielding_tensors = []

            self.skip_lines(inputfile, ["s", "b", "b", "b"])
            line = next(inputfile)
            assert "Shielding tensors in symmetry coordinates (ppm)" in line
            self.skip_lines(inputfile, ["d", "b"])
            line = next(inputfile)
            assert "Bx             By             Bz" in line
            self.skip_line(inputfile, "b")
            line = next(inputfile)

            for atomno in self.atomnos:
                tensor = []

                chomp_x = line.split()
                assert chomp_x[1] == "x"
                assert len(chomp_x) == 5
                tensor.append(chomp_x[2:])
                line = next(inputfile)
                chomp_y = line.split()
                assert chomp_y[1] == "y"
                assert len(chomp_y) == 5
                tensor.append(chomp_y[2:])
                line = next(inputfile)
                chomp_z = line.split()
                assert chomp_z[1] == "z"
                assert len(chomp_z) == 5
                tensor.append(chomp_z[2:])
                self.skip_line(inputfile, "b")
                line = next(inputfile)

                tensor = np.array(tensor, dtype=float)
                # print(np.trace(tensor) / 3)
                self.nmr_shielding_tensors.append(tensor)

        if line[:2] == "@2":
            assert (
                "Definitions from Smith, Palke, and Grieg, Concepts in Mag. Res. 4 (1992), 107"
                in line
            )
            self.skip_lines(inputfile, ["blank", "header", "blank", "header", "header"])
            line = next(inputfile)
            while line.strip():
                shielding = float(line[9:19])
                dia = float(line[19:29])
                para = float(line[29:39])
                aniso = float(line[39:49])
                asym = float(line[49:59])
                S = float(line[59:69])
                A = float(line[69:79])
                line = next(inputfile)

        if line.strip() == "Electronic g-tensor (ppm)":
            self.skip_lines(inputfile, ["=", "b"])
            line = next(inputfile)
            assert "Gauge origin (electronic charge centroid)" in line
            gauge_origin = np.array(line.split()[5:], dtype=float)
            self.skip_lines(inputfile, ["b", "b", "b"])
            line = next(inputfile)
            assert line.strip() == "Relativistic mass contribution"
            self.skip_lines(inputfile, ["d", "b"])
            line = next(inputfile)
            g_rmc_ppm = float(line.strip())
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "One-electron gauge correction"
            self.skip_lines(inputfile, ["d", "b"])
            g_gc_1e_ppm = []
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm = np.array(g_gc_1e_ppm)
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "Two-electron gauge correction"
            self.skip_lines(inputfile, ["d", "b"])
            g_gc_2e_ppm = []
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm = np.array(g_gc_2e_ppm)
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "One-electron spin-orbit+orbital-Zeeman contribution"
            self.skip_lines(inputfile, ["d", "b"])
            g_oz_soc_1e_ppm = []
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm = np.array(g_oz_soc_1e_ppm)
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "Two-electron spin-orbit+orbital-Zeeman contribution"
            self.skip_lines(inputfile, ["d", "b"])
            g_oz_soc_2e_ppm = []
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm = np.array(g_oz_soc_2e_ppm)
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "Total g-tensor shift"
            self.skip_lines(inputfile, ["d", "b"])
            g_tot_ppm = []
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm = np.array(g_tot_ppm)
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "Total g-tensor"
            self.skip_lines(inputfile, ["d", "b"])
            g_tot = []
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot = np.array(g_tot)
            self.skip_lines(inputfile, ["b", "b"])
            line = next(inputfile)
            assert line.strip() == "G-shift components (ppm)"
            self.skip_lines(inputfile, ["d", "b", "@G header"])
            line_g_rmc_ppm = dalton_parse_line(next(inputfile))
            line_g_gc_1e_ppm = dalton_parse_line(next(inputfile))
            line_g_gc_2e_ppm = dalton_parse_line(next(inputfile))
            line_g_oz_soc_1e_ppm = dalton_parse_line(next(inputfile))
            line_g_oz_soc_2e_ppm = dalton_parse_line(next(inputfile))
            line_g_tot_ppm = dalton_parse_line(next(inputfile))

            self.g_tot = g_tot

            self.g_tensor = dict()
            self.g_tensor["rmc_ppm"] = g_rmc_ppm
            self.g_tensor["gc_1e_ppm"] = g_gc_1e_ppm
            self.g_tensor["gc_2e_ppm"] = g_gc_2e_ppm
            self.g_tensor["gc_ppm"] = g_gc_1e_ppm + g_gc_2e_ppm
            self.g_tensor["oz_soc_1e_ppm"] = g_oz_soc_1e_ppm
            self.g_tensor["oz_soc_2e_ppm"] = g_oz_soc_2e_ppm
            self.g_tensor["oz_soc_ppm"] = g_oz_soc_1e_ppm + g_oz_soc_2e_ppm

            self.g_tensor = g_tensor_populate_eigvals(self.g_tensor)

            g_free = 2.002319
            # This is the g-shift that appears under "G-shift/tensor eigenvalues and cosines"
            print((np.sqrt(np.linalg.eigvals(np.dot(g_tot.T, g_tot))) - g_free) * 1.0e6)

            g_sum_ppm = (
                g_rmc_ppm * np.eye(3)
                + g_gc_1e_ppm
                + g_gc_2e_ppm
                + g_oz_soc_1e_ppm
                + g_oz_soc_2e_ppm
            )
            # print(g_tot_ppm)
            # print(g_sum_ppm)
            # This allows for 1 ppm error in every position.
            assert np.sum(abs(g_tot_ppm - g_sum_ppm)) <= 9.0

        #     g_rmc_abs = g_rmc_ppm / 1.0e6
        #     g_gc_1_abs = g_gc_1_ppm / 1.0e6
        #     g_gc_2_abs = g_gc_2_ppm / 1.0e6
        #     g_oz_soc_1_abs = g_oz_soc_1_ppm / 1.0e6
        #     g_oz_soc_2_abs = g_oz_soc_2_ppm / 1.0e6
        #     g_tot_abs = g_tot_ppm / 1.0e6

        #     g_gc_1_eigvals_abs = g_eigvals(g_gc_1_abs)
        #     g_gc_2_eigvals_abs = g_eigvals(g_gc_2_abs)
        #     g_oz_soc_1_eigvals_abs = g_eigvals(g_oz_soc_1_abs)
        #     g_oz_soc_2_eigvals_abs = g_eigvals(g_oz_soc_2_abs)
        #     g_tot_eigvals_abs = g_eigvals(g_tot_abs)

        #     print('One-electron gauge correction')
        #     print_matrix_ppm(g_gc_1_ppm)
        #     print('Two-electron gauge correction')
        #     print_matrix_ppm(g_gc_2_ppm)
        #     print('One-electron spin-orbit+orbital-Zeeman contribution')
        #     print_matrix_ppm(g_oz_soc_1_ppm)
        #     print('Two-electron spin-orbit+orbital-Zeeman contribution')
        #     print_matrix_ppm(g_oz_soc_2_ppm)

        #     print('\delta g^{{RMC}}        :  {:12.7f}'.format(g_rmc_abs))
        #     print('\delta g^{GC(1e)}     :', return_eigval_string(g_gc_1_eigvals_abs))
        #     print('\delta g^{GC(2e)}     :', return_eigval_string(g_gc_2_eigvals_abs))
        #     print('\delta g^{OZ/SOC(1e)} :', return_eigval_string(g_oz_soc_1_eigvals_abs))
        #     print('\delta g^{OZ/SOC(2e)} :', return_eigval_string(g_oz_soc_2_eigvals_abs))
        #     print('\delta g              :', return_eigval_string(g_tot_eigvals_abs))


class GaussianNMR(Gaussian, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        if "SCF GIAO Magnetic shielding tensor (ppm):" in line:
            self.nmr_shielding_tensors = []
            line = next(inputfile)
            for atomno in self.atomnos:
                tensor = []

                chomp_summary = line.split()
                isotropic_printed = float(chomp_summary[4])
                anisotropic_printed = float(chomp_summary[7])

                line = next(inputfile)

                for _ in range(3):
                    chomp = line.split()
                    assert len(chomp) == 6
                    tensor.append(chomp[1::2])
                    line = next(inputfile)

                tensor = np.array(tensor, dtype=float)
                isotropic = np.trace(tensor) / 3
                assert abs(isotropic - isotropic_printed) < 1.0e-4
                self.nmr_shielding_tensors.append(tensor)

                assert "Eigenvalues:" in line
                line = next(inputfile)
                if "Eigenvectors:" in line:
                    for _ in range(4):
                        line = next(inputfile)

        if "g value of the free electron" in line:
            g_free = parse_element_dalton(line.split()[-1])
            line = next(inputfile)
            assert "relativistic mass correction" in line
            g_rmc = parse_element_dalton(line[38:])
            line = next(inputfile)
            assert line.strip() == "diamagnetic correction to g tensor [g_DC]:"
            line = next(inputfile)
            g_dc = []
            for _ in range(3):
                elements = [
                    parse_element_dalton(line[6:21]),
                    parse_element_dalton(line[27:42]),
                    parse_element_dalton(line[48:63]),
                ]
                g_dc.append(elements)
                line = next(inputfile)
            g_dc = np.array(g_dc)
            assert (
                line.strip()
                == "orbital Zeeman and spin-orbit coupling contribution to g tensor [g_OZ/SOC]:"
            )
            line = next(inputfile)
            g_oz_soc = []
            for _ in range(3):
                elements = [
                    parse_element_dalton(line[6:21]),
                    parse_element_dalton(line[27:42]),
                    parse_element_dalton(line[48:63]),
                ]
                g_oz_soc.append(elements)
                line = next(inputfile)
            g_oz_soc = np.array(g_oz_soc)
            assert line.strip() == "g tensor [g = g_e + g_RMC + g_DC + g_OZ/SOC]:"
            line = next(inputfile)
            g_tot = []
            for _ in range(3):
                elements = [
                    parse_element_dalton(line[6:21]),
                    parse_element_dalton(line[27:42]),
                    parse_element_dalton(line[48:63]),
                ]
                g_tot.append(elements)
                line = next(inputfile)
            g_tot = np.array(g_tot)
            assert line.strip() == "g shifts relative to the free electron (ppm):"
            line = next(inputfile)
            g_shifts_printed = np.array([float(x) for x in line.split()[1::2]])

            g_tot_calculated = g_free * np.eye(3) + g_rmc * np.eye(3) + g_dc + g_oz_soc
            # test this < 1.0e-7
            # print(g_tot - g_tot_calculated)
            g_eigvals = np.sqrt(np.linalg.eigvals(np.dot(g_tot, g_tot.T)))
            idx = g_eigvals.argsort()[::1]
            g_eigvals = g_eigvals[idx]
            ppm = 1.0e6
            assert abs(((g_eigvals - g_free) * ppm).all() - g_shifts_printed.all()) < 1.0e-2

            self.g_tot = g_tot

            self.g_tensor = dict()
            self.g_tensor["rmc_abs"] = g_rmc
            self.g_tensor["gc_abs"] = g_dc
            self.g_tensor["oz_soc_abs"] = g_oz_soc
            self.g_tensor["rmc_ppm"] = g_rmc * 1.0e6
            self.g_tensor["gc_ppm"] = g_dc * 1.0e6
            self.g_tensor["oz_soc_ppm"] = g_oz_soc * 1.0e6

            self.g_tensor = g_tensor_populate_eigvals(self.g_tensor)


class NWChemNMR(NWChem, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        if "Chemical Shielding Tensors (GIAO, in ppm)" in line:
            self.nmr_shielding_tensors = []

            for atomno in self.atomnos:
                while "Atom:" not in line:
                    line = next(inputfile)
                tensor = []
                while "Total Shielding Tensor" not in line:
                    line = next(inputfile)
                for _ in range(3):
                    line = next(inputfile)
                    chomp = line.split()
                    assert len(chomp) == 3
                    tensor.append(chomp)
                tensor = np.array(tensor, dtype=float)
                self.nmr_shielding_tensors.append(tensor)


class ORCANMR(ORCA, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        if "CHEMICAL SHIFTS" in line:
            self.nmr_shielding_tensors = []

            self.skip_lines(inputfile, ["d", "b"])
            line = next(inputfile)
            for atomno in self.atomnos:
                assert list(set(line.strip())) == ["-"]
                line = next(inputfile)
                assert "Nucleus" in line
                line = next(inputfile)
                assert list(set(line.strip())) == ["-"]
                line = next(inputfile)
                assert "Tensor is right-handed." in line
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)
                assert "Raw-matrix :" in line
                tensor = []
                for _ in range(3):
                    line = next(inputfile)
                    chomp = line.split()
                    tensor.append(chomp)
                tensor = np.array(tensor, dtype=float)
                self.nmr_shielding_tensors.append(tensor)
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)
                assert "Diagonalized sT*s matrix:" in line
                line = next(inputfile)
                line = next(inputfile)
                line = next(inputfile)
                line = next(inputfile)
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)
                assert "Orientation:" in line
                for _ in range(3):
                    line = next(inputfile)
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)

        if "Coordinates of the origin" in line:
            origin = np.array(line.split()[5:8], dtype=float)

        if "ELECTRONIC G-MATRIX" in line:
            self.skip_lines(inputfile, ["d", "b"])
            line = next(inputfile)
            assert line.strip() == "The g-matrix:"
            xx, xy, xz = next(inputfile).split()
            yx, yy, yz = next(inputfile).split()
            zx, zy, zz = next(inputfile).split()
            g_matrix = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]], dtype=float)

            self.g_tot = g_matrix

            self.skip_line(inputfile, "b")

            line_gel = next(inputfile)
            line_grmc = next(inputfile)
            line_gdso1 = next(inputfile)
            line_gdso2 = next(inputfile)
            line_gdsot = next(inputfile)
            line_gpso1 = next(inputfile)
            line_gpso2 = next(inputfile)
            line_gpsot = next(inputfile)
            self.skip_line(inputfile, "d")
            line_gtot = next(inputfile)
            line_deltag = next(inputfile)
            self.skip_line(inputfile, "Orientation:")
            line_orix = next(inputfile)
            line_oriy = next(inputfile)
            line_oriz = next(inputfile)

            # Array-ify and sanity checks.
            g_rmc_abs = float(line_grmc.split()[1])
            g_dso1_abs_eig = np.array(line_gdso1.split()[1:], dtype=float)
            g_dso2_abs_eig = np.array(line_gdso2.split()[1:], dtype=float)
            g_dsot_abs_eig = np.array(line_gdsot.split()[1:], dtype=float)
            # print(g_dso1_abs_eig + g_dso2_abs_eig, g_dsot_abs_eig)
            g_pso1_abs_eig = np.array(line_gpso1.split()[1:], dtype=float)
            g_pso2_abs_eig = np.array(line_gpso2.split()[1:], dtype=float)
            g_psot_abs_eig = np.array(line_gpsot.split()[1:], dtype=float)
            # print(g_pso1_abs_eig + g_pso2_abs_eig, g_psot_abs_eig)
            g_tot_abs_eig = np.array(line_gtot.split()[1:4], dtype=float)
            deltag_abs_eig = np.array(line_deltag.split()[1:4], dtype=float)
            # print((g_rmc_abs * np.ones(3)) + g_dsot_abs_eig + g_psot_abs_eig, deltag_abs_eig)
            # print(deltag_abs_eig - ((g_rmc_abs * np.ones(3)) + g_dsot_abs_eig + g_psot_abs_eig))

            # Conversions.
            g_rmc_ppm = g_rmc_abs * 1.0e6
            g_dso1_ppm_eig = g_dso1_abs_eig * 1.0e6
            g_dso2_ppm_eig = g_dso2_abs_eig * 1.0e6
            g_dsot_ppm_eig = g_dsot_abs_eig * 1.0e6
            g_pso1_ppm_eig = g_pso1_abs_eig * 1.0e6
            g_pso2_ppm_eig = g_pso2_abs_eig * 1.0e6
            g_psot_ppm_eig = g_psot_abs_eig * 1.0e6

            g_gc_1e_ppm_eig = g_dso1_ppm_eig
            g_gc_2e_ppm_eig = g_dso2_ppm_eig
            g_oz_soc_1e_ppm_eig = g_pso1_ppm_eig
            g_oz_soc_2e_ppm_eig = g_pso2_ppm_eig

            self.g_tensor = dict()
            self.g_tensor["rmc_ppm"] = g_rmc_ppm
            self.g_tensor["gc_1e_ppm_eig"] = g_gc_1e_ppm_eig
            self.g_tensor["gc_2e_ppm_eig"] = g_gc_2e_ppm_eig
            self.g_tensor["gc_ppm_eig"] = g_gc_1e_ppm_eig + g_gc_2e_ppm_eig
            self.g_tensor["oz_soc_1e_ppm_eig"] = g_oz_soc_1e_ppm_eig
            self.g_tensor["oz_soc_2e_ppm_eig"] = g_oz_soc_2e_ppm_eig
            self.g_tensor["oz_soc_ppm_eig"] = g_oz_soc_1e_ppm_eig + g_oz_soc_2e_ppm_eig

            self.g_tensor = g_tensor_populate_eigvals(self.g_tensor)


class QChemNMR(QChem, LogfileKeepall):
    def __init__(self, *args, **kwargs):
        super().__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

    def extract(self, inputfile, line):
        super().extract(inputfile, line)

        if "Final Alpha Fock Matrix" in line:
            fockmat = np.empty(shape=(self.nbasis, self.nbasis))
            self.parse_matrix(inputfile, fockmat, maxncolsblock=4)
            if not hasattr(self, "fockao"):
                self.fockao = []
            self.fockao.append(fockmat)

        if "Final Beta Fock Matrix" in line:
            fockmat = np.empty(shape=(self.nbasis, self.nbasis))
            self.parse_matrix(inputfile, fockmat, maxncolsblock=4)
            self.fockao.append(fockmat)

        if "NMR-SHIELDING TENSORS (SCF)" in line:
            self.nmr_shielding_tensors = []

            self.skip_lines(inputfile, ["h", "h", "b", "b"])
            line = next(inputfile)

            for atomno in self.atomnos:
                assert list(set(line.strip())) == ["-"]
                line = next(inputfile)
                assert "ATOM" in line
                line = next(inputfile)
                assert list(set(line.strip())) == ["-"]
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)
                chomp_summary = line.split()
                isotropic_printed = float(chomp_summary[1])
                anisotropic_printed = float(chomp_summary[3])
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)
                assert "diamagnetic (undisturbed density) part of shielding tensor  (EFS)" in line
                for _ in range(6):
                    line = next(inputfile)
                assert "paramagnetic (undisturbed density) part of shielding tensor (SOILP)" in line
                for _ in range(6):
                    line = next(inputfile)
                assert "paramagnetic (disturbed density) part of shielding tensor   (SOI)" in line
                for _ in range(6):
                    line = next(inputfile)
                assert "total shielding tensor" in line
                line = next(inputfile)
                assert "Trace =" in line
                line = next(inputfile)
                assert "Full Tensor:" in line
                tensor = []
                line = next(inputfile)
                tensor.append(line.split())
                line = next(inputfile)
                tensor.append(line.split())
                line = next(inputfile)
                tensor.append(line.split())
                line = next(inputfile)
                assert line.strip() == ""
                line = next(inputfile)
                if "Summary of detailed contributions" in line:
                    pass
                else:
                    assert line.strip() == ""
                    line = next(inputfile)

                tensor = np.array(tensor, dtype=float)
                self.nmr_shielding_tensors.append(tensor)

        if "ELECTRONIC G-TENSOR" in line:
            self.skip_line(inputfile, "d")
            line = next(inputfile)

            assert line.strip() == "Relativistic mass correction [ppm]"
            line = next(inputfile)
            g_rmc_ppm = float(line.strip())
            line = next(inputfile)
            assert line.strip() == "One-electron gauge correction [ppm]"
            g_gc_1e_ppm = []
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm = np.array(g_gc_1e_ppm)
            line = next(inputfile)
            assert line.strip() == "Two-electron gauge correction [ppm]"
            g_gc_2e_ppm = []
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm = np.array(g_gc_2e_ppm)
            line = next(inputfile)
            assert line.strip() == "One-electron spin-orbit+orbital-Zeeman contribution [ppm]"
            g_oz_soc_1e_ppm = []
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm = np.array(g_oz_soc_1e_ppm)
            line = next(inputfile)
            assert line.strip() == "Two-electron spin-orbit+orbital-Zeeman contribution [ppm]"
            g_oz_soc_2e_ppm = []
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm = np.array(g_oz_soc_2e_ppm)
            line = next(inputfile)
            assert line.strip() == "Total shift [ppm]"
            g_tot_ppm = []
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm = np.array(g_tot_ppm)
            line = next(inputfile)
            assert line.strip() == "The electronic g-matrix:"
            g_tot = []
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot = np.array(g_tot)
            line = next(inputfile)

            self.g_tot = g_tot

            self.g_tensor = dict()
            self.g_tensor["rmc_ppm"] = g_rmc_ppm
            self.g_tensor["gc_1e_ppm"] = g_gc_1e_ppm
            self.g_tensor["gc_2e_ppm"] = g_gc_2e_ppm
            self.g_tensor["gc_ppm"] = g_gc_1e_ppm + g_gc_2e_ppm
            self.g_tensor["oz_soc_1e_ppm"] = g_oz_soc_1e_ppm
            self.g_tensor["oz_soc_2e_ppm"] = g_oz_soc_2e_ppm
            self.g_tensor["oz_soc_ppm"] = g_oz_soc_1e_ppm + g_oz_soc_2e_ppm

            self.g_tensor = g_tensor_populate_eigvals(self.g_tensor)


def parser_dispatch(outputfile):
    if "cfour" in outputfile.lower():
        return CFOURNMR
    job = ccopen(outputfile)
    program_types = (
        (DALTON, DALTONNMR),
        (Gaussian, GaussianNMR),
        (NWChem, NWChemNMR),
        (ORCA, ORCANMR),
        (QChem, QChemNMR),
    )
    for program_type, parser_class in program_types:
        if isinstance(job, program_type):
            return parser_class
    return


def getargs():
    """Get command-line arguments."""

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("outputfilename", nargs="+")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = getargs()

    for outputfilename in args.outputfilename:
        print("-" * 70)
        print(outputfilename)
        parser_class = parser_dispatch(outputfilename)
        job = parser_class(outputfilename)
        data = job.parse()

        print(convertor(data.scfenergies[0], "eV", "hartree"))
        # isotropic_shieldings = [np.trace(nmr_shielding_tensor) / 3
        #                         for nmr_shielding_tensor in data.nmr_shielding_tensors]
        # print(isotropic_shieldings)
        # if hasattr(data, 'g_tot'):
        # print(data.g_tot)
        # print(sorted(g_eigvals(data.g_tot)))

        if hasattr(data, "g_tensor"):
            for k in sorted(data.g_tensor.keys()):
                if "eig" in k:
                    print(k, data.g_tensor[k], np.mean(data.g_tensor[k]))

    print("-" * 70)
