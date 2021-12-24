#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import r_numint

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]
mol.basis = '6-31g'
mol.build()

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_eval_rho(self):
        n2c = mol.nao_2c()
        numpy.random.seed(10)
        ngrids = 100
        coords = numpy.random.random((ngrids,3))*20
        coords = coords[70:75]
        dm = numpy.random.random((n2c,n2c))
        dm = dm + dm.T.conj()
        aoLa, aoLb, aoSa, aoSb = r_numint.eval_ao(mol, coords, deriv=1)

        rho0a = numpy.einsum('pi,ij,pj->p', aoLa[0], dm, aoLa[0].conj())
        rho0b = numpy.einsum('pi,ij,pj->p', aoLb[0], dm, aoLb[0].conj())
        rho0 = rho0a + rho0b

        aoL = numpy.array([aoLa[0],aoLb[0]])
        m0 = numpy.einsum('api,ji,bpj,xab->xp', aoL.conj(), dm, aoL, lib.PauliMatrices)

        ni = r_numint.RNumInt()
        rho1, m1 = ni.eval_rho(mol, (aoLa[0], aoLb[0]), dm, xctype='LDA')
        self.assertAlmostEqual(abs(rho1.imag).max(), 0, 9)
        self.assertAlmostEqual(abs(rho0-rho1).max(), 0, 9)
        self.assertTrue(numpy.allclose(m0, m1))

    def test_eval_mat(self):
        numpy.random.seed(10)
        ngrids = 100
        coords = numpy.random.random((ngrids,3))*20
        rho = numpy.random.random((ngrids))
        m = numpy.random.random((3,ngrids)) * .05
        vxc = [numpy.random.random((2,ngrids)).T, None, None, None]
        weight = numpy.random.random(ngrids)
        aoLa, aoLb, aoSa, aoSb = r_numint.eval_ao(mol, coords, deriv=1)

        s = numpy.linalg.norm(m, axis=0)
        m_pauli = numpy.einsum('xp,xij,p->pij', m, lib.PauliMatrices, 1./(s+1e-300))
        aoL = numpy.array([aoLa[0],aoLb[0]])

        mat0 = numpy.einsum('pi,p,pj->ij', aoLa[0].conj(), weight*vxc[0][:,0], aoLa[0])
        mat0+= numpy.einsum('pi,p,pj->ij', aoLb[0].conj(), weight*vxc[0][:,0], aoLb[0])
        mat0+= numpy.einsum('api,p,pab,bpj->ij', aoL.conj(), weight*vxc[0][:,1], m_pauli, aoL)
        mat1 = r_numint.eval_mat(mol, (aoLa[0], aoLb[0]), weight, (rho, m), vxc, xctype='LDA')
        self.assertTrue(numpy.allclose(mat0, mat1))

    def test_rsh_omega(self):
        rho0 = numpy.array([[1., 1., 0.1, 0.1],
                            [.1, .1, 0.01, .01]]).reshape(2, 4, 1)
        ni = r_numint.RNumInt()
        ni.omega = 0.4
        omega = 0.2
        exc, vxc, fxc, kxc = ni.eval_xc('ITYH,', rho0, deriv=1, omega=omega)
        self.assertAlmostEqual(float(exc), -0.6394181669577297, 7)
        #self.assertAlmostEqual(float(vxc[0][0,0]), -0.8688965017309331, 7)  # libxc-4.3.4
        #self.assertAlmostEqual(float(vxc[0][0,1]), -0.04641346660681983, 7)  # libxc-4.3.4
        self.assertAlmostEqual(float(vxc[0][0,0]), -0.868874616534556, 7)  # libxc-5.1.2
        self.assertAlmostEqual(float(vxc[0][0,1]), -0.0465674503111865, 7)  # libxc-5.1.2
        # vsigma of GGA may be problematic?
        #?self.assertAlmostEqual(float(vxc[1][0,0]), 0, 7)
        #?self.assertAlmostEqual(float(vxc[1][0,1]), 0, 7)

        exc, vxc, fxc, kxc = ni.eval_xc('ITYH,', rho0, deriv=1)
        self.assertAlmostEqual(float(exc), -0.5439673757289064, 7)
        #self.assertAlmostEqual(float(vxc[0][0,0]), -0.7699824959456474, 7)  # libxc-4.3.4
        #self.assertAlmostEqual(float(vxc[0][0,1]), -0.04529004028228567, 7)  # libxc-4.3.4
        self.assertAlmostEqual(float(vxc[0][0,0]), -0.7698921068411966, 7)  # libxc-5.1.2
        self.assertAlmostEqual(float(vxc[0][0,1]), -0.04592601580190408, 7)  # libxc-5.1.2
        # vsigma of GGA may be problematic?
        #?self.assertAlmostEqual(float(vxc[1][0,0]), 0, 7)
        #?self.assertAlmostEqual(float(vxc[1][0,1]), 0, 7)

    def test_vxc(self):
        numpy.random.seed(10)
        nao = mol.nao_2c()
        n4c = nao * 2
        dms =(numpy.random.random((2,n4c,n4c)) +
              numpy.random.random((2,n4c,n4c)) * 1j)
        dms = dms + dms.transpose(0, 2, 1).conj()
        grids = dft.Grids(mol)
        grids.atom_grid = {"H": (50, 110), "O": (50, 110)}
        grids.prune = False

        v = r_numint.RNumInt().r_vxc(mol, grids, 'lda,', dms, hermi=1)[2]
        self.assertAlmostEqual(lib.fp(v), -0.023694712341023827+1.6530188048162782j, 12)

        v = r_numint.RNumInt().r_vxc(mol, grids, 'HF', dms, hermi=0)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)

        v = r_numint.RNumInt().r_vxc(mol, grids, '', dms, hermi=0)[2]
        self.assertAlmostEqual(abs(v).max(), 0, 9)


if __name__ == "__main__":
    print("Test r_numint")
    unittest.main()

