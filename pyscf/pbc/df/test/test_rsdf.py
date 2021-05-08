# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import scipy.linalg
import h5py
from pyscf import lib
import pyscf.pbc
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import rsdf, ft_ao
from pyscf.pbc.tools import pbc as pbctools

def gdf_via_aft(cell, auxcell, mesh, q, kpts):
    nao = cell.nao
    naux = auxcell.nao
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = pbctools.get_coulG(cell, q, False, None, mesh, Gv)
    coulG *= kws
    aoao = ft_ao.ft_aopair_kpts(cell, Gv, aosym='s1', q=q, kptjs=kpts)
    Gaux = ft_ao.ft_ao(auxcell, Gv, kpt=q)
    ref = lib.einsum('ngij,gk->nijk', aoao, coulG[:,None] * Gaux.conj())
    j2c = lib.dot(Gaux.conj().T*coulG, Gaux)
    ref = scipy.linalg.solve_triangular(
        scipy.linalg.cholesky(j2c, lower=True), ref.reshape(-1,naux).T, lower=True)
    ref = ref.reshape(naux, len(kpts), nao, nao).transpose(1,0,2,3)
    return ref

class KnownValues(unittest.TestCase):
    def test_RangeSeperationCell(self):
        cell = pgto.M(a = '''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329''',
                      atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391''',
                      basis = 'ccpvdz')

        rs_cell = ft_ao._RangeSeperationCell.from_cell(
            cell, ke_cut_threshold=20, rcut_threshold=3.2)
        self.assertEqual(rs_cell.nao, 38)
        self.assertEqual(rs_cell.nbas, 14)

        s_rs = rs_cell.intor('int1e_ovlp')
        s0 = cell.intor('int1e_ovlp')
        fcontract = rs_cell.recontract()
        s1 = fcontract(fcontract(s_rs).T)
        self.assertAlmostEqual(abs(s0 - s1).max(), 0, 9)

        d_idx = rs_cell.get_ao_indices(
            np.where(rs_cell.bas_type == rsdf.SMOOTH_BASIS)[0], rs_cell.ao_loc)
        s_dd = s_rs[d_idx[:,None], d_idx].copy()
        self.assertEqual(s_dd.shape, (8, 8))
        s_rs[d_idx[:,None], d_idx] = 0
        s1 = fcontract(fcontract(s_rs).T)

        fmerge = rs_cell.merge_diffused_block()
        s2 = fmerge(s1[:,:,None], s_dd[:,:,None], (0, cell.nbas, 0, cell.nbas))
        self.assertAlmostEqual(abs(s0 - s2[:,:,0]).max(), 0, 9)

    def test_bas_mask_to_sh_loc(self):
        cell = pgto.M(a='''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329''',
                      atom='He', basis=[[0, [1, 1]]])
        rs_cell = ft_ao._RangeSeperationCell.from_cell(cell, 10.0, 3.2)
        np.random.seed(10)
        rs_cell.bas_type = np.array([0, 2, 0, 2, 1])
        rs_cell.sh_loc = rs_cell._reverse_bas_map([0, 0, 1, 1, 2])
        bas_mask = np.random.random((3, 5, 3)) < .5
        sh_loc = rsdf._ExtendedMoleSR.bas_mask_to_sh_loc(rs_cell, bas_mask)
        self.assertTrue(np.array_equiv(sh_loc, [0, 3, 6, 7, 8, 12, 12, 15, 17, 19]))

    def test_ft_ao(self):
        cell = pgto.M(output='/dev/null',
                      a='''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329
                        ''',
                      atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391''',
                      basis='''
C S
4.3362376436      0.1490797872
1.2881838513      -0.0292640031
C P
1.2881838513      -0.27755603
''')
        cell.verbose = 8

        kpts = cell.make_kpts([3,1,1])
        auxcell = cell
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).build()
        mesh = [5, 5, 5]
        ft_kern = df1.supmol_ft.gen_ft_kernel()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = None
        q = kpts[1]
        kptjs = kpts
        gijR, gijI = ft_kern(Gv, gxyz, None, q, kptjs)
        self.assertAlmostEqual(lib.fp(gijR), 6.039584880263869, 9)
        self.assertAlmostEqual(lib.fp(gijI), 6.511927635915370, 9)
        ref = ft_ao.ft_aopair_kpts(cell, Gv, q=q, kptjs=kptjs)
        self.assertAlmostEqual(abs(gijR - ref.real).max(), 0, 9)
        self.assertAlmostEqual(abs(gijI - ref.imag).max(), 0, 9)

        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        gijR, gijI = ft_kern(Gv, gxyz, None, q, kptjs)
        self.assertAlmostEqual(lib.fp(gijR), 6.039584880263869, 9)
        self.assertAlmostEqual(lib.fp(gijI), 6.511927635915370, 9)
        self.assertAlmostEqual(abs(gijR - ref.real).max(), 0, 9)
        self.assertAlmostEqual(abs(gijI - ref.imag).max(), 0, 9)

        # TODO: test GTO_Gv_nonorth

    def test_sr_3c2e(self):
        cell = pgto.M(output='/dev/null',
                      a='''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329''',
                      atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391''',
                      basis='''
C S
4.3362376436      0.1490797872
1.2881838513      -0.0292640031
C P
1.2881838513      -0.27755603
''')
        cell.verbose = 8
        kpts = cell.make_kpts([4,1,1])
        auxcell = cell
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).build()
        #int3c = df1.gen_int3c_kernel(aosym='s2', j_only=True, kpts=kpts)
        int3c = df1.gen_int3c_kernel(aosym='s1', j_only=True, kpts=kpts)
        outR, outI = int3c()
        mesh = [30] * 3
        q = np.zeros(3)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = df1.weighted_coulG(q, False, mesh) - df1.weighted_coulG_LR(q, False, mesh)
        coulG[0] += np.pi/df1.omega**2 * kws
        aoao = ft_ao.ft_aopair_kpts(cell, Gv, aosym='s1', q=q, kptjs=kpts)
        Gaux = ft_ao.ft_ao(auxcell, Gv, kpt=q)
        ref = lib.einsum('ngij,gk->nijk', aoao, coulG[:,None] * Gaux.conj())
        self.assertAlmostEqual(abs(ref.real - outR.reshape(ref.shape)).max(), 0, 9)
        self.assertAlmostEqual(abs(ref.imag - outI.reshape(ref.shape)).max(), 0, 9)

    def test_smooth_block_3c2e(self):
        cell = pgto.M(output='/dev/null',
                      a=np.eye(3)*2.4,
                      atom='''
He     0.      0.      0.
He     0.4917  0.4917  0.4917''',
                      basis = {'He': [[0, [1.2, 1]],
                                      [0, [0.2, 1]], ]})
        kpts = cell.make_kpts([3,1,1])
        auxcell = cell
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).set(omega=0.6).build()
        with h5py.File('cderi.h5', 'w') as f:
            df1.outcore_smooth_block(f, kpts=kpts)
            val = [f['j3cR-dd/0'][:],
                   None,
                   f['j3cR-dd/2'][:],
                   f['j3cR-dd/3'][:],
                   f['j3cR-dd/4'][:],
                   None,
                   None,
                   f['j3cR-dd/7'][:],
                   f['j3cR-dd/8'][:]]

        def _via_aft(cell, auxcell, mesh, q, kpts):
            Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
            coulG = pbctools.get_coulG(cell, q, False, None, mesh, Gv)
            coulG *= kws
            aoao = ft_ao.ft_aopair_kpts(cell, Gv, aosym='s1', q=q, kptjs=kpts)
            Gaux = ft_ao.ft_ao(auxcell, Gv, kpt=q)
            ref = lib.einsum('ngij,gk->nijk', aoao, coulG[:,None] * Gaux.conj())
            return ref

        cell_d = df1.rs_cell.smooth_basis_cell()
        ref = _via_aft(cell_d, cell, cell_d.mesh, kpts[0], kpts)
        self.assertAlmostEqual(abs(val[0] - ref[0].real).max(), 0, 9)
        self.assertAlmostEqual(abs(val[4] - ref[1].real).max(), 0, 9)
        self.assertAlmostEqual(abs(val[8] - ref[2].real).max(), 0, 9)
        ref = _via_aft(cell_d, cell, cell_d.mesh, kpts[2], kpts)
        self.assertAlmostEqual(abs(val[3] - ref[0].real).max(), 0, 8)
        self.assertAlmostEqual(abs(val[7] - ref[1].real).max(), 0, 8)
        self.assertAlmostEqual(abs(val[2] - ref[2].real).max(), 0, 8)

    def test_2c2e(self):
        cell = pgto.M(output='/dev/null',
                      a='''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329''',
                      atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391''',
                      basis='''
C S
4.3362376436      0.1490797872
1.2881838513      -0.0292640031
0.4037767149      -0.688204051
0.1187877657      -0.3964426906
C P
4.3362376436      -0.0878123619
1.2881838513      -0.27755603
0.4037767149      -0.4712295093
0.1187877657      -0.4058039291
''')
        cell.verbose = 8
        kpts = cell.make_kpts([5,1,1])
        auxcell = cell
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).build()
        j2c = df1.get_2c2e(kpts)

        auxcell = df1.auxcell
        mesh = [30]*3
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        b = auxcell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ref = []
        for k, kpt in enumerate(kpts):
            coulG = df1.weighted_coulG(kpt, False, mesh)
            auxG = ft_ao.ft_ao(auxcell, Gv, None, b, gxyz, Gvbase, kpt).T
            ref.append(lib.dot(auxG.conj()*coulG, auxG.T))
        self.assertAlmostEqual(abs(np.array(j2c) - np.array(ref)).max(), 0, 11)

    def test_gdf_gamma_point(self):
        cell = pgto.M(output='/dev/null',
                      a='''
3.370137329, 0.000000000, 0.000000000
0.000000000, 3.370137329, 0.000000000
0.000000000, 0.000000000, 3.370137329''',
                      atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391''',
                      basis='''
C S
4.3362376436      0.1490797872
1.2881838513      -0.0292640031
0.2037767149      -0.688204051
C P
4.3362376436      -0.0878123619
1.2881838513      -0.27755603
0.2037767149      -0.4712295093
''')
        cell.verbose = 8

        auxbasis = [
            [0, [1.0, 1]],
            [0, [.15, 1]],
        ]
        auxcell = cell.copy()
        auxcell.basis = auxbasis
        auxcell.build(0, 0)
        kpts = cell.make_kpts([1,1,1])
        q = np.zeros(3)
        mesh = [15] * 3
        ref = gdf_via_aft(cell, auxcell, mesh, q, kpts)

        cderi = 'test_rsdf.h5'
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).set(omega=0.6).build()
        df1.make_j3c(cderi, aosym='s2')
        nao = cell.nao
        tril_mask = np.tril(np.ones((nao, nao), dtype=bool))
        with h5py.File(cderi, 'r') as val:
            self.assertAlmostEqual(abs(val['j3c/0/0'][:] - ref[0][:,tril_mask].real).max(), 0, 9)
            self.assertTrue(val['j3c/0/0'][:].dtype == np.double)

        df1.make_j3c(cderi, aosym='s1')
        with h5py.File(cderi, 'r') as val:
            self.assertAlmostEqual(abs(val['j3c/0/0'][:].ravel() - ref[0].ravel().real).max(), 0, 9)

    def test_gdf_j_only(self):
        cell = pgto.M(output='/dev/null',
                      a=np.eye(3)*3.4,
                      atom = '''
He     0.      0.      0.
He     0.4917  0.4917  0.4917''',
                      basis={'He':
                             [[1, [1.0, 1, -.2],
                                  [0.15, .5, .5]],
                             ]})
        cell.verbose = 8

        auxbasis = [
            [0, [1.1, 1],
                [0.5, 1],
                [.15, 1]],
            [1, [0.8, 1]],
        ]
        auxcell = cell.copy()
        auxcell.basis = auxbasis
        auxcell.build(0, 0)
        kpts = cell.make_kpts([3,1,1])
        q = np.zeros(3)
        mesh = [16] * 3
        ref = gdf_via_aft(cell, auxcell, mesh, q, kpts)

        cderi = 'test_rsdf.h5'
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).set(omega=0.6).build()
        df1.make_j3c(cderi, aosym='s2', j_only=True)
        nao = cell.nao
        tril_mask = np.tril(np.ones((nao, nao), dtype=bool))
        with h5py.File(cderi, 'r') as val:
            self.assertAlmostEqual(abs(val['j3c/0/0'][:] - ref[0][:,tril_mask]).max(), 0, 10)
            self.assertAlmostEqual(abs(val['j3c/4/0'][:] - ref[1][:,tril_mask]).max(), 0, 10)
            self.assertAlmostEqual(abs(val['j3c/8/0'][:] - ref[2][:,tril_mask]).max(), 0, 10)

        df1.make_j3c(cderi, aosym='s1', j_only=True)
        with h5py.File(cderi, 'r') as val:
            self.assertAlmostEqual(abs(val['j3c/0/0'][:].ravel() - ref[0].ravel()).max(), 0, 10)
            self.assertAlmostEqual(abs(val['j3c/4/0'][:].ravel() - ref[1].ravel()).max(), 0, 10)
            self.assertAlmostEqual(abs(val['j3c/8/0'][:].ravel() - ref[2].ravel()).max(), 0, 10)

    def test_gdf_kpt(self):
        cell = pgto.M(output='/dev/null',
                      a=np.eye(3)*3.4,
                      atom = '''
He     0.      0.      0.
He     0.4917  0.4917  0.4917''',
                      basis={'He':
                             [[1, [1.1, 1, -.2],
                                  [0.15, .5, .5]],
                             ]})
        cell.verbose = 8

        auxbasis = [
            [0, [1.4, 1],
                [0.5, 1],
                [.15, 1]],
            [1, [1.0, 1]],
        ]
        auxcell = cell.copy()
        auxcell.basis = auxbasis
        auxcell.build(0, 0)
        kpts = cell.get_abs_kpts(np.array([[0.4, .25, .2]]))
        q = np.zeros(3)
        mesh = [15] * 3
        ref = gdf_via_aft(cell, auxcell, mesh, q, kpts)

        cderi = 'test_rsdf.h5'
        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).set(omega=0.6).build()
        df1.make_j3c(cderi, aosym='s2')
        nao = cell.nao
        tril_mask = np.tril(np.ones((nao, nao), dtype=bool))
        with h5py.File(cderi, 'r') as val:
            self.assertAlmostEqual(abs(val['j3c/0/0'][:] - ref[0][:,tril_mask]).max(), 0, 8)

        val = rsdf.make_j3c(cell, auxcell, cderi, kpts=kpts, aosym='s1')
        with h5py.File(cderi, 'r') as val:
            self.assertAlmostEqual(abs(val['j3c/0/0'][:].ravel() - ref[0].ravel()).max(), 0, 8)

    def test_gdf_full(self):
        cell = pgto.M(output='/dev/null',
                      a=np.eye(3)*3.4,
                      atom = '''
He     0.      0.      0.
He     0.4917  0.4917  0.4917''',
                      basis={'He':
                             [[1, [1.0, 1, -.2],
                                  [0.15, .5, .5]],
                             ]})
        cell.verbose = 8

        auxbasis = [
            [0, [1.0, 1],
                [0.5, 1],
                [.15, 1]],
            [1, [0.7, 1]],
        ]
        auxcell = cell.copy()
        auxcell.basis = auxbasis
        auxcell.build(0, 0)
        kpts = cell.make_kpts([3,2,1])
        mesh = [16] * 3
        nao = cell.nao

        df1 = rsdf._RangeSeparationDFBuilder(cell, auxcell, kpts).set(omega=0.6).build()
        cderi = 'test_rsdf.h5'
        df1.make_j3c(cderi, aosym='s2')
        cderi1 = 'test_rsdf1.h5'
        df1.make_j3c(cderi1, aosym='s1')
        tril_mask = np.tril(np.ones((nao, nao), dtype=bool))

        def check(k_idx, ref):
            with h5py.File(cderi, 'r') as val:
                for i,k in enumerate(k_idx):
                    self.assertAlmostEqual(
                        abs(val[f'j3c/{k}/0'] - ref[i][:,tril_mask]).max(), 0, 8)
            with h5py.File(cderi1, 'r') as val:
                for i,k in enumerate(k_idx):
                    self.assertAlmostEqual(
                        abs(val[f'j3c/{k}/0'][:].ravel() - ref[i].ravel()).max(), 0, 8)

        q = np.zeros(3)
        ref = gdf_via_aft(cell, auxcell, mesh, q, kpts)
        check([0, 7, 14, 21, 28, 35], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[1], kpts[[1,3,5]])
        check([1, 15, 29], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, -kpts[1], kpts[[0,2,4]])
        check([6, 20, 34], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[2], kpts[[2,3,4,5]])
        check([2, 9, 16, 23], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, -kpts[4], kpts[[0,1]])
        check([24, 31], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[4], kpts[[4,5]])
        check([4, 11], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, -kpts[2], kpts[[0,1,2,3]])
        check([12, 19, 26, 33], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[3], kpts[[3,5]])
        check([3, 17], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[2]-kpts[1], kpts[[2,4]])
        check([8, 22], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[1]-kpts[4], kpts[[1,0]])
        check([25, 30], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[5], kpts[[5]])
        check([5], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[4]-kpts[1], kpts[[4]])
        check([10], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[1]-kpts[2], kpts[[1,3]])
        check([13, 27], ref)

        ref = gdf_via_aft(cell, auxcell, mesh, kpts[0]-kpts[3], kpts[[0,2]])
        check([18, 32], ref)

if __name__ == '__main__':
    print("Full Tests for rsdf")
    unittest.main()
