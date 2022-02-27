#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import copy
from pyscf import lib, gto, scf, dft
from pyscf import tdscf

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = '''
O     0.   0.       0.
H     0.   -0.757   0.587
H     0.   0.757    0.587'''
mol.spin = 2
mol.basis = '631g'
mol.build()

molsym = gto.M(
    atom='''
O     0.   0.       0.
H     0.   -0.757   0.587
H     0.   0.757    0.587''',
    spin=2,
    basis='631g',
    symmetry=True)

h3 = gto.Mole()
h3.verbose = 5
h3.output = '/dev/null'
h3.atom = '''
H     0.   0.    0.
H     0.  -0.7   0.7
H     0.   0.7   0.7'''
h3.basis = '631g'
h3.spin = 1
h3.build()

mf_ghf = scf.GHF(molsym).run()
mf_bp86 = dft.GKS(molsym).set(xc='bp86', conv_tol=1e-12).newton().run()
mf_lda = dft.GKS(mol).set(xc='lda,', conv_tol=1e-12).run()
mf_b3lyp = dft.GKS(mol).set(xc='b3lyp', conv_tol=1e-12).newton().run()
mf_m06l = dft.GKS(mol).run(xc='m06l')

mcol_lda = dft.GKS(h3).set(xc='lda,vwn', collinear='mcol', conv_tol=1e-8)
mcol_lda._numint.spin_samples = 6
mcol_lda = mcol_lda.run()
mcol_b3lyp = dft.GKS(h3).set(xc='b3lyp', collinear='mcol', conv_tol=1e-8)
mcol_b3lyp._numint.spin_samples = 6
mcol_b3lyp = mcol_b3lyp.run()
mcol_m06l = dft.GKS(h3).set(xc='m06,', collinear='mcol', conv_tol=1e-8)
mcol_m06l._numint.spin_samples = 6
mcol_m06l = mcol_m06l.run()

def tearDownModule():
    global mol, molsym, h3, mf_ghf, mf_lda, mf_b3lyp, mf_m06l, mcol_lda, mcol_b3lyp, mcol_m06l
    mol.stdout.close()
    h3.stdout.close()
    del mol, molsym, h3, mf_ghf, mf_lda, mf_b3lyp, mf_m06l, mcol_lda, mcol_b3lyp, mcol_m06l

def diagonalize(a, b, nroots=4):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = numpy.bmat([[a        , b       ],
                    [-b.conj(),-a.conj()]])
    e = numpy.linalg.eig(numpy.asarray(h))[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    lowest_e = lowest_e[lowest_e > 1e-3]
    return lowest_e

class KnownValues(unittest.TestCase):
    def test_tddft_lda(self):
        td = mf_lda.TDDFT()
        es = td.kernel(nstates=4)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 6)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 7.198586785950176, 6)

        td = mcol_lda.TDDFT()
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 6)
        es = td.kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 1.5707747356780348, 5)

    def test_tddft_b88p86(self):
        td = mf_bp86.CasidaTDDFT()
        es = td.kernel(nstates=5)[0]
        a,b = td.get_ab()
        e_ref = diagonalize(a, b, 6)
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 7.881922749754810, 6)

    def test_tddft_b3lyp(self):
        td = mf_b3lyp.TDDFT()
        es = td.kernel(nstates=4)[0]
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 7.934339053919082, 6)

        td = mcol_b3lyp.TDDFT()
        es = td.kernel(nstates=4)[0]
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), -0.8429897079277264, 5)

    def test_tda_b3lyp(self):
        td = mf_b3lyp.TDA()
        es = td.kernel(nstates=4)[0]
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 7.962133569183688, 6)

        td = mcol_b3lyp.TDA()
        es = td.kernel(nstates=4)[0]
        self.assertAlmostEqual(lib.fp(es[:3]*27.2114), 2.9234687103972155, 5)

    def test_tda_lda(self):
        td = mf_lda.TDA()
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 7.2177002168314734, 6)
        ref = [6.37845795, 6.75960766, 6.75960766, 7.17797962, 7.9042117]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 5)

        td = mcol_lda.TDA()
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 2.2501675680378717, 6)
        ref = [2.41748322, 6.79123567, 9.21942837, 16.65799482, 18.55397532]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 5)

    def test_tda_m06l(self):
        td = mf_m06l.TDA()
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 9.264754678239758, 6)
        ref = [8.18680046, 8.68229347, 8.68229347, 8.77578157, 10.20105534]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 6)

        td = mcol_m06l.TDA()
        es = td.kernel(nstates=5)[0] * 27.2114
        self.assertAlmostEqual(lib.fp(es[:3]), 1.5801208359638494, 6)
        ref = [1.82046573, 4.11185386, 5.91615459, 8.27819098, 11.96381448]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 5)

    def test_ab_hf(self):
        mf = mf_ghf
        a, b = mf.TDHF().get_ab()
        ftda = mf.TDA().gen_vind()[0]
        ftdhf = mf.TDHF().gen_vind()[0]
        nocc = numpy.count_nonzero(mf.mo_occ == 1)
        nvir = numpy.count_nonzero(mf.mo_occ == 0)
        numpy.random.seed(2)
        x, y = xy = numpy.random.random((2,nocc,nvir))

        ax = numpy.einsum('iajb,jb->ia', a, x)
        self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 9)

        ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
        ab2 =-numpy.einsum('iajb,jb->ia', b, x)
        ab2-= numpy.einsum('iajb,jb->ia', a, y)
        abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
        self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_ab_ks(self):
        for mf in (mf_lda, mf_b3lyp, mf_m06l, mcol_lda, mcol_b3lyp, mcol_m06l):
            td = tdscf.gks.TDDFT(mf)
            a, b = td.get_ab()
            ftda = mf.TDA().gen_vind()[0]
            ftdhf = td.gen_vind()[0]
            nocc = numpy.count_nonzero(mf.mo_occ == 1)
            nvir = numpy.count_nonzero(mf.mo_occ == 0)
            numpy.random.seed(2)
            x, y = xy = numpy.random.random((2,nocc,nvir))

            ax = numpy.einsum('iajb,jb->ia', a, x)
            self.assertAlmostEqual(abs(ax - ftda([x]).reshape(nocc,nvir)).max(), 0, 9)

            ab1 = ax + numpy.einsum('iajb,jb->ia', b, y)
            ab2 =-numpy.einsum('iajb,jb->ia', b, x)
            ab2-= numpy.einsum('iajb,jb->ia', a, y)
            abxy_ref = ftdhf([xy]).reshape(2,nocc,nvir)
            self.assertAlmostEqual(abs(ab1 - abxy_ref[0]).max(), 0, 9)
            self.assertAlmostEqual(abs(ab2 - abxy_ref[1]).max(), 0, 9)

    def test_tda_with_wfnsym(self):
        td = mf_bp86.TDA()
        td.wfnsym = 'B1'
        es = td.kernel(nstates=3)[0]
        self.assertAlmostEqual(lib.fp(es), 0.4523465502706356, 6)

    def test_tdhf_with_wfnsym(self):
        td = mf_ghf.TDHF()
        td.wfnsym = 'B1'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 0.48380638923581476, 6)

    def test_tddft_with_wfnsym(self):
        td = mf_bp86.TDDFT()
        td.wfnsym = 'B1'
        td.nroots = 3
        es = td.kernel()[0]
        self.assertAlmostEqual(lib.fp(es), 0.45050838461527387, 6)


if __name__ == "__main__":
    print("Full Tests for TD-GKS")
    unittest.main()
