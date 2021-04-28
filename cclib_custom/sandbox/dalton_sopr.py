"""Tools for working with second-order (linear response properties) from
DALTON calculations.
"""

import re
import numpy as np

from cclib.parser.daltonparser import DALTON

from cclib_custom import ccDataKeepall
from cclib_custom import LogfileKeepall

from extract_nmr import dalton_parse_line

from parse_matrices_dalton import \
    (parse_matrix_dalton, parse_element_dalton)


proplist_angmom = (
    'XANGMOM',
    'YANGMOM',
    'ZANGMOM',
)
proplist_angecc = (
    'XANGECC',
    'YANGECC',
    'ZANGECC',
)
proplist_spnorb1 = (
    'X1SPNORB',
    'Y1SPNORB',
    'Z1SPNORB',
)
proplist_spnorb2 = (
    'X1SPNSCA',
    'Y1SPNSCA',
    'Z1SPNSCA',
)
proplist_diplen = (
    'XDIPLEN',
    'YDIPLEN',
    'ZDIPLEN',
)


def getargs():
    """Get command-line arguments."""

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()

    return args


def get_uhf_values(mat_uhf_a, mat_uhf_b, pair_rohf, nocc_a, nvirt_a, nocc_b, nvirt_b):
    """For a pair ROHF 1-based indices, find the corresponing alpha- and
    beta-spin UHF values.
    """

    # TODO there has to be a better way than including this here...
    range_uhf_a_closed = list(range(0, nocc_a))
    range_uhf_a_virt = list(range(nocc_a, nocc_a + nvirt_a))
    range_uhf_b_closed = list(range(0, nocc_b))
    range_uhf_b_virt = list(range(nocc_b, nocc_b + nvirt_b))
    indices_uhf_a = [(i, a) for i in range(nocc_a) for a in range(nvirt_a)]
    indices_uhf_b = [(i, a) for i in range(nocc_b) for a in range(nvirt_b)]
    # These are the indices for unique pairs considering the full
    # dimensionality of the system (correct orbital window), [norb,
    # norb], starting from 1.
    indices_display_uhf_a = [(p+1, q+1) for p in range_uhf_a_closed for q in range_uhf_a_virt]
    indices_display_uhf_b = [(p+1, q+1) for p in range_uhf_b_closed for q in range_uhf_b_virt]

    values = []
    if pair_rohf in indices_display_uhf_a:
        idx_uhf_a = indices_display_uhf_a.index(pair_rohf)
        p_a, q_a = indices_uhf_a[idx_uhf_a]
        val_uhf_a = mat_uhf_a[p_a, q_a]
        values.append(val_uhf_a)
    if pair_rohf in indices_display_uhf_b:
        idx_uhf_b = indices_display_uhf_b.index(pair_rohf)
        p_b, q_b = indices_uhf_b[idx_uhf_b]
        val_uhf_b = mat_uhf_b[p_b, q_b]
        values.append(val_uhf_b)
    return values


def sparse_to_dense_matrix_dalton(spmat, dim1, dim2, antisymmetric=False, mocoeffs=False):
    """Turn the sparse dictionary representation of a matrix into a dense
    NumPy array.

    If the two dimensions are the same, assume the matrix is meant to
    be symmetric.

    If the two dimensions differ, assume nothing.
    """

    densemat = np.zeros(shape=(dim1, dim2))
    # print(dim1, dim2, antisymmetric, mocoeffs)
    if mocoeffs or dim1 != dim2:
        for ridx in spmat:
            for cidx in spmat[ridx]:
                # print(ridx, cidx)
                densemat[ridx-1][cidx-1] = spmat[ridx][cidx]
    else:
        assert dim1 == dim2
        for ridx in spmat:
            for cidx in spmat[ridx]:
                densemat[ridx-1][cidx-1] = densemat[cidx-1][ridx-1] = spmat[ridx][cidx]

    # If the integrals are supposed to be antisymmetric, flip the sign
    # of the upper triangle.
    if antisymmetric and dim1 == dim2:
        triu_indices = np.triu_indices(dim1)
        densemat[triu_indices] *= -1.0

    return densemat


class DALTONExt(DALTON, LogfileKeepall):

    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the superclass
        super(DALTONExt, self).__init__(datatype=ccDataKeepall, future=True, *args, **kwargs)

        ## Put things in the constructor that are static so we aren't
        ## forced to rewrite the before parsing/after parsing methods.

        # This is how DALTON prints true/false values.
        self.boolmap = {'T': True, 'F': False}

        # We could parse these, but it's pointless since we don't have
        # anything to compare them to from any other program package.
        self.ignore_these_matrices = (
            'HUCKOVLP',
            'HUCKEL',
            'HJPOPOVL',
        )

        # These labels correspond to integrals that need to be
        # antisymmetrized. Look in DALTON/abacus/her1pro.F/PR1IN1 at
        # the ANTI flag.
        self.antisymmetric_integral_labels = (
            'DIPVEL',
            'SPNORB',
            'PSO',
            'S1MAG', 'dS/dB',
            'ANGMOM',
            'MAGMOM', 'dh/dB',
            'CM1',
            'LONSOL1',
            'DPLGRA',
            'QUAGRA',
            'OCTGRA',
            'ROTSTR',
            'PVIOLA',
            'QDB',
            'SOSCALE', 'SPNSCA',
            'ANGECC',
            'OZKE',
            'PSOKE',
            'B[XYZ]\sEF\s[XYZ]',
            'DERAM',
            'DIPANH',
        )

    def extract(self, inputfile, line):

        if 'Integrals of operator:' in line:
            matchline = 'Integrals of operator:'
            if not hasattr(self, 'AOPROPER'):
                self.AOPROPER = dict()
            if not any(ignore in line
                       for ignore in self.ignore_these_matrices):
                # need to handle headers with spaces/multiple parts
                start = line.index(matchline) + len(matchline)
                matname = line[start:][:-2].strip()
                # Store antisymmetric matrices as their antisymmetric
                # representation!
                is_antisymmetric = any(re.search(ail, matname)
                                       for ail in self.antisymmetric_integral_labels)
                mat_sparse = parse_matrix_dalton(inputfile)
                mat = sparse_to_dense_matrix_dalton(mat_sparse, self.nbasis, self.nbasis,
                                                    antisymmetric=is_antisymmetric)
                self.AOPROPER[matname] = mat


        if 'KZYWOP,KZYCON,KZYVAR' in line:
            chomp = line.split()
            assert len(chomp) == 4
            self.KZYWOP = int(chomp[1])
            self.KZYCON = int(chomp[2])
            self.KZYVAR = int(chomp[3])
            assert self.KZYWOP == (self.KZYCON + self.KZYVAR)
            self.KZVAR = self.KZYVAR // 2
            # assume there are no configuration space parameters!
            self.KZRED = self.KZVAR
            self.KZYRED = self.KZYVAR

        if 'KZYVAR, KZCONF, KZVAR' in line:
            chomp = line.split()
            assert len(chomp) == 6
            self.KZYVAR = int(chomp[3])
            self.KZCONF = int(chomp[4])
            self.KZVAR = int(chomp[5])
            assert self.KZYVAR == 2 * self.KZVAR
            # assume there are no configuration space parameters!
            self.KZRED = self.KZVAR
            self.KZYRED = self.KZYVAR

        if line.strip() == 'FOCK VALENCE MATRIX':
            self.skip_lines(inputfile, ['blank', '*** Block'])
            line = next(inputfile)
            FV_sparse = parse_matrix_dalton(inputfile)
            FV = sparse_to_dense_matrix_dalton(FV_sparse, self.nbasis, self.nbasis)
            self.FV = FV

        if line.strip() == 'FOCK CORE MATRIX':
            self.skip_lines(inputfile, ['blank', '*** Block'])
            line = next(inputfile)
            FC_sparse = parse_matrix_dalton(inputfile)
            FC = sparse_to_dense_matrix_dalton(FC_sparse, self.nbasis, self.nbasis)
            self.FC = FC

        if line.strip() == 'TOTAL FOCK  MATRIX':
            self.skip_lines(inputfile, ['blank', '*** Block'])
            line = next(inputfile)
            FT_sparse = parse_matrix_dalton(inputfile)
            FT = sparse_to_dense_matrix_dalton(FT_sparse, self.nbasis, self.nbasis)
            self.FT = FT

        if line.strip() == 'MO. COEFFICIENTS FOR SYMMETRY    1':
            self.skip_line(inputfile, 'blank')
            mocoeffs_sparse = parse_matrix_dalton(inputfile)
            mocoeffs_list = []
            mocoeffs_full = sparse_to_dense_matrix_dalton(mocoeffs_sparse, self.nbasis, self.nbasis, mocoeffs=True)
            mocoeffs_list.append(mocoeffs_full.T)
            # set_attribute isn't working properly...
            self.mocoeffs = mocoeffs_list

        if ' Perturbation symmetry.     KSYMOP:' in line:
            self.KSYMOP = int(line.split()[-1])
            line = next(inputfile)
            assert ' Perturbation spin symmetry.TRPLET:' in line
            self.TRPLET = self.boolmap[line.split()[-1]]
            line = next(inputfile)
            assert ' Orbital variables.         KZWOPT:' in line
            self.KZWOPT = int(line.split()[-1])
            line = next(inputfile)
            assert ' Configuration variables.   KZCONF:' in line
            self.KZCONF = int(line.split()[-1])
            line = next(inputfile)
            assert ' Total number of variables. KZVAR :' in line
            self.KZVAR = int(line.split()[-1])
            assert self.KZWOPT + self.KZCONF == self.KZVAR

        if '  DIAGONAL ORBITAL PART OF E(2)' in line:
            self.skip_line(inputfile, 'blank')
            e2_diag_orb_sparse = parse_matrix_dalton(inputfile)
            e2_diag_orb = sparse_to_dense_matrix_dalton(e2_diag_orb_sparse, self.KZYVAR, 1)
            self.e2_diag_orb = np.reshape(e2_diag_orb, (self.KZVAR, 2), order='F')

        if '  DIAGONAL ORBITAL PART OF S(2)' in line:
            self.skip_line(inputfile, 'blank')
            s2_diag_orb_sparse = parse_matrix_dalton(inputfile)
            s2_diag_orb = sparse_to_dense_matrix_dalton(s2_diag_orb_sparse, self.KZYVAR, 1)
            self.s2_diag_orb = np.reshape(s2_diag_orb, (self.KZVAR, 2), order='F')

        if 'Linear Response single residue calculation' in line:
            # avoid StopIteration in the superclass
            line = next(inputfile)

        if line[0:14] == ' E(2) MATRIX :':
            e2_full_dim = int(line.split()[-1])
            assert e2_full_dim == self.KZYVAR
            line = next(inputfile)
            e2_full_sparse = parse_matrix_dalton(inputfile)
            e2_full = sparse_to_dense_matrix_dalton(e2_full_sparse, e2_full_dim, e2_full_dim)
            # if hasattr(self, 'e2_diag_orb'):
            #     print(self.e2_diag_orb.flatten(order='A'))
            #     print(np.diag(e2_full))
            self.E2_full = e2_full

        if line[0:14] == ' S(2) MATRIX :':
            s2_full_dim = int(line.split()[-1])
            assert s2_full_dim == self.KZYVAR
            line = next(inputfile)
            s2_full_sparse = parse_matrix_dalton(inputfile)
            s2_full = sparse_to_dense_matrix_dalton(s2_full_sparse, s2_full_dim, s2_full_dim)
            self.S2_full = s2_full

        # if 'KZRED' in line:
        #     self.KZRED = int(line.split()[-1])
        #     self.KZYRED = self.KZRED * 2

        # if line.strip() == 'RSPRED: REDUCED HESSIAN MATRIX:':
        #     if not hasattr(self, 'E2_reduced'):
        #         self.E2_reduced = dict()
        #     line = next(inputfile)
        #     e2_reduced_sparse = parse_matrix_dalton(inputfile)
        #     e2_reduced = sparse_to_dense_matrix_dalton(e2_reduced_sparse, self.KZYRED, self.KZYRED)
        #     if self.section_label not in self.E2_reduced:
        #         self.E2_reduced[self.section_label] = []
        #     self.E2_reduced[self.section_label].append(e2_reduced)

        # if 'REDUCED GRADIENT VECTORS DIM (2,KZRED)' in line:
        #     KZRED = int(line.split()[-1])
        #     assert KZRED == self.KZRED
        #     if not hasattr(self, 'GP_reduced'):
        #         self.GP_reduced = dict()
        #     line = next(inputfile)
        #     gp_reduced_sparse = parse_matrix_dalton(inputfile)
        #     gp_reduced = sparse_to_dense_matrix_dalton(gp_reduced_sparse, 2, KZRED)
        #     if self.section_label not in self.GP_reduced:
        #         self.GP_reduced[self.section_label] = []
        #     self.GP_reduced[self.section_label].append(gp_reduced)

        if 'Gradient vectors in RSPLR for operator' in line:
            label = line[55:-2].strip()
            if not hasattr(self, 'GP'):
                self.GP = dict()
            self.skip_lines(inputfile, ['d', 'b', 'b'])
            line = next(inputfile)
            assert 'Norm :' in line
            norm = parse_element_dalton(line.split()[2])
            line = next(inputfile)
            GP_sparse = parse_matrix_dalton(inputfile)
            GP = sparse_to_dense_matrix_dalton(GP_sparse, self.KZVAR, 2, False, False)
            assert abs(norm - np.linalg.norm(GP)) < 1.0e-2
            self.GP[label] = GP

        if 'Solution vectors in RSPLR for operator' in line:
            label = line[55:-2].strip()
            if not hasattr(self, 'RSPVEC'):
                self.RSPVEC = dict()
            self.skip_lines(inputfile, ['d', 'b', 'b'])
            RSPVEC_sparse = parse_matrix_dalton(inputfile)
            RSPVEC = sparse_to_dense_matrix_dalton(RSPVEC_sparse, self.KZVAR, 2)
            line = next(inputfile)
            norm = parse_element_dalton(line[6:])
            assert abs(norm - np.linalg.norm(RSPVEC)) < 1.0e7
            self.RSPVEC[label] = RSPVEC

        if 'Residual vectors in RSPLR for operator' in line:
            label = line[55:-2].strip()
            if not hasattr(self, 'RESVEC'):
                self.RESVEC = dict()
            self.skip_lines(inputfile, ['d', 'b', 'b'])
            RESVEC_sparse = parse_matrix_dalton(inputfile)
            RESVEC = sparse_to_dense_matrix_dalton(RESVEC_sparse, self.KZVAR, 2)
            line = next(inputfile)
            norm = parse_element_dalton(line[6:])
            assert abs(norm - np.linalg.norm(RESVEC)) < 1.0e7
            self.RESVEC[label] = RESVEC

        # These are ...
        if '@ FREQUENCY INDEPENDENT SECOND ORDER PROPERTIES' in line:
            if not hasattr(self, 'propdict_lr'):
                self.propdict_lr = dict()
            self.skip_line(inputfile, 'blank')
            while line.strip() != '':
                line = next(inputfile)
                if '@ -<<' in line:
                    lab1, lab2 = line[5:15].strip(), line[16:26].strip()
                    chomp = line.split()
                    val = parse_element_dalton(chomp[-1])
                    self.propdict_lr[(lab1, lab2)] = val
            self._proparr_angecc()
            self._proparr_angmom_1spnorb()

        # One way to get property vectors. Required a modification to
        # DALTON/rsp/rspqrx3.F/QRGP by removing debug pragmas.
        if 'QRGP:Operator' in line:
            assert self.section == 'PRPGET'
            if not hasattr(self, 'PROPVEC'):
                self.PROPVEC = dict()
            # TODO
            label = line.split()[-1].strip()
            while 'Property vector' not in line:
                line = next(inputfile)
            self.skip_lines(inputfile, ['d', 'b'])
            line = next(inputfile)
            PROPVEC_sparse = parse_matrix_dalton(inputfile)
            PROPVEC = sparse_to_dense_matrix_dalton(PROPVEC_sparse, self.KZVAR, 2)
            self.PROPVEC[label] = PROPVEC

        if 'PRPGET: PROPERTY MATRIX' in line:
            self.section = 'PRPGET'
            if not hasattr(self, 'PROPVEC'):
                self.PROPVEC = dict()
            self.section_label = line[25:33].strip()

        if line.strip().endswith('IN MO. BASIS'):
            assert self.section == 'PRPGET'
            if not hasattr(self, 'MOPROPER'):
                self.MOPROPER = dict()
            label = line[10:20].strip()
            mat_sparse = parse_matrix_dalton(inputfile)
            mat = sparse_to_dense_matrix_dalton(mat_sparse, self.nbasis, self.nbasis)
            self.MOPROPER[label] = mat

        if '*** Individual non-zero orbital contributions' in line:
            assert self.section == 'PRPGET'
            line = next(inputfile)
            assert '*** to the expectation value for property' in line
            label = line[43:51].strip()
            assert self.section_label == label
            while 'inactive part' not in line:
                line = next(inputfile)
            # do nothing with these for now
            inactive = parse_element_dalton(line[28:])
            line = next(inputfile)
            assert 'active part' in line
            active = parse_element_dalton(line[28:])
            line = next(inputfile)
            assert 'total' in line
            total = parse_element_dalton(line[28:])
            assert abs(inactive + active - total) < 1.0e16

        # Another way to get property vectors. This just requires a
        # high print level in **RESPONSE and no code modifications.
        if 'Sigma vector from one electron operator' in line:
            assert self.section == 'PRPGET'
            assert self.section_label is not None
            line = next(inputfile)
            PROPVEC_sparse = parse_matrix_dalton(inputfile)
            PROPVEC = sparse_to_dense_matrix_dalton(PROPVEC_sparse, self.KZYVAR, 1)
            self.PROPVEC[self.section_label] = np.reshape(PROPVEC, (self.KZVAR, 2), order='F')

        if line.strip() == 'Electronic g-tensor (ppm)':
            self.skip_lines(inputfile, ['=', 'b'])
            line = next(inputfile)
            assert 'Gauge origin (electronic charge centroid)' in line
            gauge_origin = np.array(line.split()[5:], dtype=float)
            self.skip_lines(inputfile, ['b', 'b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'Relativistic mass contribution'
            self.skip_lines(inputfile, ['d', 'b'])
            line = next(inputfile)
            g_rmc_ppm = float(line.strip())
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'One-electron gauge correction'
            self.skip_lines(inputfile, ['d', 'b'])
            g_gc_1e_ppm = []
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_1e_ppm = np.array(g_gc_1e_ppm)
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'Two-electron gauge correction'
            self.skip_lines(inputfile, ['d', 'b'])
            g_gc_2e_ppm = []
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_gc_2e_ppm = np.array(g_gc_2e_ppm)
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'One-electron spin-orbit+orbital-Zeeman contribution'
            self.skip_lines(inputfile, ['d', 'b'])
            g_oz_soc_1e_ppm = []
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_1e_ppm = np.array(g_oz_soc_1e_ppm)
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'Two-electron spin-orbit+orbital-Zeeman contribution'
            self.skip_lines(inputfile, ['d', 'b'])
            g_oz_soc_2e_ppm = []
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm.append([float(x) for x in next(inputfile).split()])
            g_oz_soc_2e_ppm = np.array(g_oz_soc_2e_ppm)
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'Total g-tensor shift'
            self.skip_lines(inputfile, ['d', 'b'])
            g_tot_ppm = []
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm.append([float(x) for x in next(inputfile).split()])
            g_tot_ppm = np.array(g_tot_ppm)
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'Total g-tensor'
            self.skip_lines(inputfile, ['d', 'b'])
            g_tot = []
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot.append([float(x) for x in next(inputfile).split()])
            g_tot = np.array(g_tot)
            self.skip_lines(inputfile, ['b', 'b'])
            line = next(inputfile)
            assert line.strip() == 'G-shift components (ppm)'
            self.skip_lines(inputfile, ['d', 'b', '@G header'])
            line_g_rmc_ppm = dalton_parse_line(next(inputfile))
            line_g_gc_1e_ppm = dalton_parse_line(next(inputfile))
            line_g_gc_2e_ppm = dalton_parse_line(next(inputfile))
            line_g_oz_soc_1e_ppm = dalton_parse_line(next(inputfile))
            line_g_oz_soc_2e_ppm = dalton_parse_line(next(inputfile))
            line_g_tot_ppm = dalton_parse_line(next(inputfile))

            self.g_oz_soc_1e_ppm = g_oz_soc_1e_ppm
            self.g_tot = g_tot

        # Call the extract method of the superclass, get all the
        # regular parsed attributes for free!
        # Placed all the way at the bottom so we avoid lack of features
        # in the superclass.
        super(DALTONExt, self).extract(inputfile, line)

    def _proparr_angecc(self):
        if hasattr(self, 'propdict_lr'):
            propmap_angecc = {'XANGECC': 0, 'YANGECC': 1, 'ZANGECC': 2}
            self.proparr_angecc = np.zeros(shape=(3, 3))
            for (lab1, lab2) in self.propdict_lr:
                if 'ANGECC' in lab1 and 'ANGECC' in lab2:
                    r, c = propmap_angecc[lab1], propmap_angecc[lab2]
                    self.proparr_angecc[r, c] = \
                        self.proparr_angecc[c, r] = \
                        self.propdict_lr[(lab1, lab2)]

    # TODO fixme
    def _proparr_angmom_1spnorb(self):
        if hasattr(self, 'propdict_lr'):
            propmap_angmom = {'XANGMOM': 0, 'YANGMOM': 1, 'ZANGMOM': 2}
            propmap_1spnorb = {'X1SPNORB': 0, 'Y1SPNORB': 1, 'Z1SPNORB': 2}
            self.proparr_angmom_1spnorb = np.zeros(shape=(3, 3))
            for (lab1, lab2) in self.propdict_lr:
                if 'ANGMOM' in lab1 and '1SPNORB' in lab2:
                    r, c = propmap_angmom[lab1], propmap_1spnorb[lab2]
                    self.proparr_angmom_1spnorb[r, c] = \
                        self.propdict_lr[(lab1, lab2)]


def orbsx(zymat, D, norb, nclosed, nact, nvirt, TRPLET=False):
    """Collect the terms from zymat (a DALTON MO-basis matrix) that
    contribute to the orbital part of the property gradient. Store
    them in a matrix of the same shape (they are turned into a vector
    later).

    A direct translation from DALTON/rsp/rspqrx3.F/ORBSX.
    """
    # k(l,k) = <0| [ E(k,l), zymat ] |0>
    # 1) k(a,i) = 2 * OVLAP * zymat(a,i)
    # 2) k(i,a) = - 2 * OVLAP * zymat(i,a)
    # 3) k(t,i) = 2 * OVLAP * zymat(t,i)- sum(x) zymat(x,i) D(x,t)
    # 4) k(i,t) = sum(x) zymat(i,x) D(t,x) - 2 * OVLAP * zymat(i,t)
    # 5) k(t,a) = - sum(x) zymat(x,a) D(t,x)
    # 6) k(a,t) = sum(x) zymat(a,x) D(t,x)
    # 7) k(u,t) = sum(x) zymat(u,x) D(t,x) - zymat(x,t) D(x,u)
    OVLAP = 1.0
    assert nclosed + nact + nvirt == norb
    # [i,j,k,l]   inactive (closed)
    # [t,u,v,x,y] active (open)
    # [a,b,c,d]   secondary (virtual)
    # [p,q,r,s]   general (all)
    r_inactive = list(range(0, nclosed))
    r_active = list(range(nclosed, nclosed + nact))
    r_secondary = list(range(nclosed + nact, nclosed + nact + nvirt))
    # r_general = list(range(norb))
    k = np.zeros((norb, norb))
    if not TRPLET:
        for i in r_inactive:
            for a in r_secondary:
                k[a, i] += 2 * OVLAP * zymat[a, i]
                k[i, a] += -2 * OVLAP * zymat[i, a]
    for t in r_active:
        for i in r_inactive:
            k[t, i] += 2 * OVLAP * zymat[t, i] - sum(zymat[x, i] * D[x, t] for x in r_active)
            k[i, t] += sum(zymat[i, x] * D[t, x] for x in r_active) - 2 * OVLAP * zymat[i, t]
        for a in r_secondary:
            k[t, a] += -sum(zymat[x, a] * D[t, x] for x in r_active)
            k[a, t] += sum(zymat[a, x] * D[t, x] for x in r_active)
        for u in r_active:
            k[u, t] += sum((zymat[u, x] * D[t, x])  - (zymat[x, t] * D[x, u]) for x in r_active)
    return k


def orbdia(UDV, FOCK, FC, FV, norb, nclosed, nact, nvirt, AVDIA=False):
    """A direct translation from DALTON/rsp/rspe2c.F/ORBDIA.

     CALCULATE FOCK CONTRIBUTIONS TO DIAGONAL E(2) AND S(2) MATRICES

        EODIA(L,K) = < [E(K,L),H,E(L,K)] > L>K
        SODIA(L,K) = < [E(K,L),E(L,K)] >   L>K

     SECONDARY - INACTIVE (L,K) = (A,I)
     EODIA(A,I) = -FOCK(I,I) + 2*FC(A,A) +2FV(A,A)
     SODIA(A,I) = 2

     SECONDARY - ACTIVE (L,K) = (A,M)
     EODIA(A,M) = UDV(M,M)*FC(A,A) - FOCK(M,M)
     SODIA(A,M) = UDV(M,M)

     ACTIVE - INACTIVE (L,K) = (M,I)
     EODIA(M,I) = 2*FC(M,M) + UDV(M,M)*FC(I,I) - FOCK(I,I) - FOCK(M,M) + 2*FV(M,M)
     SODIA(M,I) = 2 - UDV(M,M)

     ACTIVE - ACTIVE (L,K) = (M,N)
     EODIA(M,N) = UDV(N,N)*FC(M,M) + UDV(M,M)*FC(N,N) - FOCK(M,M) -FOCK(N,N)
     SODIA(M,N) = UDV(N,N) - UDV(M,M)

     AVDIA = .TRUE. , ADD FV CONTRIBUTIONS TO EODIA
             WHICH ORIGINATE FROM FOCK TYPE DECOUPLING OF THE TWO
             ELECTRON DENSITY MATRIX.
             ALL UDV(*,*)*FC(*,*) CONTRIBUTIONS THEN BECOME
                 UDV(*,*)*(FC(*,*)+FV(*,*))
    """
    assert nclosed + nact + nvirt == norb
    # Original:
    # [i,j,k,l]   inactive (closed)
    # [t,u,v,x,y] active (open)
    # [a,b,c,d]   secondary (virtual)
    # [p,q,r,s]   general (all)
    # This routine:
    # [i]   inactive
    # [m,n] active
    # [a]   secondary
    r_inactive = list(range(0, nclosed))
    r_active = list(range(nclosed, nclosed + nact))
    r_secondary = list(range(nclosed + nact, nclosed + nact + nvirt))
    r_general = list(range(norb))
    eodia = np.zeros((norb, norb))
    sodia = np.zeros((norb, norb))
    for a in r_secondary:
        for i in r_inactive:
            eodia[a, i] += -FOCK[i, i] + 2*FC[a, a] + 2*FV[a, a]
            sodia[a, i] += 2.0
        for m in r_active:
            eodia[a, m] += UDV[m, m]*FC[a, a] - FOCK[m, m]
            if AVDIA:
                eodia[a, m] += UDV[m, m]*FV[a, a]
            sodia[a, m] += UDV[m, m]
    for m in r_active:
        for i in r_inactive:
            eodia[m, i] += 2*FC[m, m] + UDV[m, m]*FC[i, i] - FOCK[i, i] - FOCK[m, m] + 2*FV[m, m]
            if AVDIA:
                eodia[m, i] += UDV[m, m]*FV[i, i]
            sodia[m, i] += 2 - UDV[M, M]
        for n in r_active:
            eodia[m, n] += UDV[n, n]*FC[m, m] + UDV[m, m]*FC[n, n] - FOCK[m, m] - FOCK[n, n]
            if AVDIA:
                eodia[m, n] += UDV[n, n]*FV[m, m] + UDV[m, m]*FV[n, n]
            sodia[m, n] += UDV[n, n] - UDV[m, m]
    return eodia, sodia


def repack_orb_to_dalton(A, norb, nclosed, nact, nvirt):
    """Repack a [norb, norb] matrix into a [(nclosed*nact) +
    (nclosed*nvirt) + (nact*nvirt)] vector for contraction with the CI
    Hamiltonian.
    """

    assert norb == nclosed + nact + nvirt
    assert A.shape == (norb, norb)

    # These might be available in the global namespace, but this
    # function should work on its own.
    range_closed = list(range(0, nclosed))
    range_act = list(range(nclosed, nclosed + nact))
    range_virt = list(range(nclosed + nact, nclosed + nact + nvirt))
    indices_rohf_closed_act = [(i, t) for i in range_closed for t in range_act]
    indices_rohf_closed_virt = [(i, a) for i in range_closed for a in range_virt]
    indices_rohf_act_virt = [(t, a) for t in range_act for a in range_virt]

    B = np.zeros(len(indices_rohf_closed_act) + \
                 len(indices_rohf_closed_virt) + \
                 len(indices_rohf_act_virt))

    for (i, t) in indices_rohf_closed_act:
        it = (t - nclosed)*nclosed + i
        B[it] += A[i, t]
    for (i, a) in indices_rohf_closed_virt:
        ia = i*nvirt + a - nclosed - nact + (nclosed*nact)
        B[ia] += A[i, a]
    for (t, a) in indices_rohf_act_virt:
        ta = (t - nclosed)*nvirt + a - nclosed - nact + (nclosed*nact) + (nclosed*nvirt)
        B[ta] += A[t, a]

    return B



if __name__ == '__main__':

    args = getargs()

    logfile = DALTONExt(args.filename)
    data = logfile.parse()
    if args.verbose >= 1:
        print('\n'.join(dir(data)))

    if args.verbose >= 1:
        if hasattr(data, 'RSPVEC'):
            print('=== RSPVEC ===')
            for k in sorted(data.RSPVEC):
                print(k)
                print(data.RSPVEC[k])
        if hasattr(data, 'GP'):
            print('=== GP ===')
            for k in sorted(data.GP):
                print(k)
                print(data.GP[k])
        if hasattr(data, 'PROPVEC'):
            print('=== PROPVEC ===')
            for k in sorted(data.PROPVEC):
                print(k)
                print(data.PROPVEC[k])
        if hasattr(data, 'e2_diag_orb'):
            print('=== E[2] (diagonal) ===')
            print(data.e2_diag_orb)
        print('=' * 78)

    # Contract all possible response vectors with all possible
    # property vectors to get the final LR values.
    if hasattr(data, 'RSPVEC') and hasattr(data, 'GP'):
        for k in sorted(data.RSPVEC):
            for l in sorted(data.GP):
                left = np.dot(data.RSPVEC[k][:, 0], data.GP[l][:, 0])
                right = np.dot(data.RSPVEC[k][:, 1], data.GP[l][:, 1])
                both = np.sum(data.RSPVEC[k] * data.GP[l])
                assert (left + right) == both
                print(k, l, both)
