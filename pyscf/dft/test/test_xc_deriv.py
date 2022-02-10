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

import unittest
import numpy as np
from pyscf import gto
from pyscf import dft
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft import xc_deriv

def eval_xc(xctype, ng):
    np.random.seed(1)
    outbuf = np.random.rand(220,ng)
    exc = outbuf[0]
    if xctype == 'R-LDA':
        vxc = [outbuf[1]]
        fxc = [outbuf[2]]
        kxc = [outbuf[3]]
    elif xctype == 'R-GGA':
        vxc = [outbuf[1], outbuf[2]]
        fxc = [outbuf[3], outbuf[4], outbuf[5]]
        kxc = [outbuf[6], outbuf[7], outbuf[8], outbuf[9]]
    elif xctype == 'U-LDA':
        vxc = [outbuf[1:3].T]
        fxc = [outbuf[3:6].T]
        kxc = [outbuf[6:10].T]
    elif xctype == 'U-GGA':
        vxc = [outbuf[1:3].T, outbuf[3:6].T]
        fxc = [outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T]
        kxc = [outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T]
    elif xctype == 'R-MGGA':
        vxc = [outbuf[1], outbuf[2], None, outbuf[4]]
        fxc = [
            # v2rho2, v2rhosigma, v2sigma2,
            outbuf[5], outbuf[6], outbuf[7],
            # v2lapl2, v2tau2,
            None, outbuf[9],
            # v2rholapl, v2rhotau,
            None, outbuf[11],
            # v2lapltau, v2sigmalapl, v2sigmatau,
            None, None, outbuf[14]]
        # v3lapltau2 might not be strictly 0
        # outbuf[18] = 0
        kxc = [
            # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
            outbuf[15], outbuf[16], outbuf[17], outbuf[18],
            # v3rho2lapl, v3rho2tau,
            None, outbuf[20],
            # v3rhosigmalapl, v3rhosigmatau,
            None, outbuf[22],
            # v3rholapl2, v3rholapltau, v3rhotau2,
            None, None, outbuf[25],
            # v3sigma2lapl, v3sigma2tau,
            None, outbuf[27],
            # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
            None, None, outbuf[30],
            # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
            None, None, None, outbuf[34]]
    elif xctype == 'U-MGGA':
        vxc = [outbuf[1:3].T, outbuf[3:6].T, None, outbuf[8:10].T]
        # v2lapltau might not be strictly 0
        # outbuf[39:43] = 0
        fxc = [
            # v2rho2, v2rhosigma, v2sigma2,
            outbuf[10:13].T, outbuf[13:19].T, outbuf[19:25].T,
            # v2lapl2, v2tau2,
            None, outbuf[28:31].T,
            # v2rholapl, v2rhotau,
            None, outbuf[35:39].T,
            # v2lapltau, v2sigmalapl, v2sigmatau,
            None, None, outbuf[49:55].T]
        # v3lapltau2 might not be strictly 0
        # outbuf[204:216] = 0
        kxc = [
            # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
            outbuf[55:59].T, outbuf[59:68].T, outbuf[68:80].T, outbuf[80:90].T,
            # v3rho2lapl, v3rho2tau,
            None, outbuf[96:102].T,
            # v3rhosigmalapl, v3rhosigmatau,
            None, outbuf[114:126].T,
            # v3rholapl2, v3rholapltau, v3rhotau2,
            None, None, outbuf[140:146].T,
            # v3sigma2lapl, v3sigma2tau,
            None, outbuf[158:170].T,
            # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
            None, None, outbuf[191:200].T,
            # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
            None, None, None, outbuf[216:220].T]
    return exc, vxc, fxc, kxc

def v6to5(v6):
    if v6.ndim == 2:
        v5 = v6[[0,1,2,3,5]]
    else:
        v5 = v6[:,[0,1,2,3,5]]
    return v5

def v5to6(v5):
    if v5.ndim == 2:
        v6 = np.zeros((6, v5.shape[1]))
        v6[[0,1,2,3,5]] = v5
    else:
        v6 = np.zeros((2, 6, v5.shape[2]))
        v6[:,[0,1,2,3,5]] = v5
    return v6

class KnownValues(unittest.TestCase):
    def test_gga_deriv1(self):
        ng = 7
        xctype = 'GGA'
        np.random.seed(8)
        rho = np.random.rand(2,4,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = numint._rks_gga_wv0(rho[0], vxc, weight)
        ref[0] *= 2
        v1  = xc_deriv.transform_vxc(rho[0], vxc, xctype, spin=0)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = np.array(numint._uks_gga_wv0(rho, vxc, weight))
        ref[:,0] *= 2
        v1  = xc_deriv.transform_vxc(rho, vxc, xctype, spin=1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_gga_deriv2(self):
        ng = 7
        xctype = 'GGA'
        np.random.seed(8)
        rho = np.random.rand(2,4,ng)
        rho1 = np.random.rand(2,4,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = numint._rks_gga_wv1(rho[0], rho1[0], vxc, fxc, weight)
        ref[0] *= 2
        v1 = xc_deriv.transform_fxc(rho[0], vxc, fxc, xctype, spin=0)
        v1 = np.einsum('xg,xyg->yg', rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = np.array(numint._uks_gga_wv1(rho, rho1, vxc, fxc, weight))
        ref[:,0] *= 2
        v1  = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
        v1 = np.einsum('axg,axbyg->byg', rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_gga_deriv3(self):
        ng = 7
        xctype = 'GGA'
        np.random.seed(8)
        rho = np.random.rand(2,4,ng)
        rho1 = np.random.rand(2,4,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = numint._rks_gga_wv2(rho[0], rho1[0], fxc, kxc, weight)
        ref[0] *= 2
        v1 = xc_deriv.transform_kxc(rho[0], fxc, kxc, xctype, spin=0)
        v1 = np.einsum('xg,yg,xyzg->zg', rho1[0], rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = np.array(numint._uks_gga_wv2(rho, rho1, fxc, kxc, weight))
        ref[:,0] *= 2
        v1  = xc_deriv.transform_kxc(rho, fxc, kxc, xctype, spin=1)
        v1 = np.einsum('axg,byg,axbyczg->czg', rho1, rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_mgga_deriv1(self):
        ng = 7
        xctype = 'MGGA'
        np.random.seed(8)
        rho = np.random.rand(2,5,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = v6to5(numint._rks_mgga_wv0(v5to6(rho[0]), vxc, weight))
        ref[0] *= 2
        ref[4] *= 4
        v1  = xc_deriv.transform_vxc(rho[0], vxc, xctype, spin=0)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = v6to5(np.array(numint._uks_mgga_wv0(v5to6(rho), vxc, weight)))
        ref[:,0] *= 2
        ref[:,4] *= 4
        v1  = xc_deriv.transform_vxc(rho, vxc, xctype, spin=1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_mgga_deriv2(self):
        ng = 7
        xctype = 'MGGA'
        np.random.seed(8)
        rho = np.random.rand(2,5,ng)
        rho1 = np.random.rand(2,5,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = v6to5(numint._rks_mgga_wv1(v5to6(rho[0]), v5to6(rho1[0]), vxc, fxc, weight))
        ref[0] *= 2
        ref[4] *= 4
        v1 = xc_deriv.transform_fxc(rho[0], vxc, fxc, xctype, spin=0)
        v1 = np.einsum('xg,xyg->yg', rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = v6to5(np.array(numint._uks_mgga_wv1(v5to6(rho), v5to6(rho1), vxc, fxc, weight)))
        ref[:,0] *= 2
        ref[:,4] *= 4
        v1  = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
        v1 = np.einsum('axg,axbyg->byg', rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

    def test_mgga_deriv3(self):
        ng = 7
        xctype = 'MGGA'
        np.random.seed(8)
        rho = np.random.rand(2,5,ng)
        rho1 = np.random.rand(2,5,ng)
        weight = 1

        exc, vxc, fxc, kxc = eval_xc(f'R-{xctype}', ng)
        ref = v6to5(numint._rks_mgga_wv2(v5to6(rho[0]), v5to6(rho1[0]), fxc, kxc, weight))
        ref[0] *= 2
        ref[4] *= 4
        v1 = xc_deriv.transform_kxc(rho[0], fxc, kxc, xctype, spin=0)
        v1 = np.einsum('xg,yg,xyzg->zg', rho1[0], rho1[0], v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

        exc, vxc, fxc, kxc = eval_xc(f'U-{xctype}', ng)
        ref = v6to5(np.array(numint._uks_mgga_wv2(v5to6(rho), v5to6(rho1), fxc, kxc, weight)))
        ref[:,0] *= 2
        ref[:,4] *= 4
        v1  = xc_deriv.transform_kxc(rho, fxc, kxc, xctype, spin=1)
        v1 = np.einsum('axg,byg,axbyczg->czg', rho1, rho1, v1)
        self.assertAlmostEqual(abs(v1 - ref).max(), 0, 12)

if __name__ == "__main__":
    print("Test xc_deriv")
    unittest.main()
