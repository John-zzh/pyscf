#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

'''
Numerical integration functions for (2-component) GKS with real AO basis
'''

import functools
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, _scale_ao, _tau_dot, BLKSIZE
from pyscf.dft import xc_deriv
from pyscf.dft import mcfun
from pyscf import __config__


def gks_mcol_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                 max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # Ignore density laplacian for mcfun
    with_lapl = ni.collinear[0] != 'm'
    make_rho, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi, with_lapl)
    nao = n2c // 2

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'n'): (_ncol_lda_vxc_mat , 0),
            ('LDA' , 'm'): (_mcol_lda_vxc_mat , 0),
            ('GGA' , 'm'): (_mcol_gga_vxc_mat , 1),
            ('MGGA', 'm'): (_mcol_mgga_vxc_mat, 2),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]

        if ni.collinear[0] == 'm':  # mcol
            fn_eval_xc = ni.mcfun_eval_xc_wrapper(xc_code)
            nproc = lib.num_threads()

        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                if ni.collinear[0] == 'm':
                    exc, vxc = mcfun.eval_xc_eff(
                        fn_eval_xc, rho, deriv=1, spin_samples=ni.spin_samples,
                        collinear_threshold=ni.collinear_thrd,
                        collinear_samples=ni.collinear_samples, workers=nproc)[:2]
                else:
                    exc, vxc = ni.eval_xc(xc_code, rho, spin=1,
                                          relativity=relativity, deriv=1,
                                          verbose=verbose)[:2]
                if xctype == 'LDA':
                    den = rho[0] * weight
                else:
                    den = rho[0,0] * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                vmat[i] += fmat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_fxc for functional {xc_code}')

    vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat

def gks_mcol_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                 rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    assert ni.collinear[0] == 'm'  # mcol
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # Ignore density laplacian for mcfun
    with_lapl = ni.collinear[0] != 'm'
    if rho0 is None and (xctype != 'LDA' or fxc is None):
        make_rho0 = ni._gen_rho_evaluator(mol, dm0, 1, with_lapl)[0]
    else:
        make_rho0 = None

    make_rho1, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi, with_lapl)
    nao = n2c // 2
    vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

    fn_eval_xc = ni.mcfun_eval_xc_wrapper(xc_code)
    nproc = lib.num_threads()
    if xctype == 'MGGA':
        fmat, ao_deriv = (_mcol_mgga_fxc_mat , 2)
    elif xctype == 'GGA':
        fmat, ao_deriv = (_mcol_gga_fxc_mat  , 1)
    else:
        fmat, ao_deriv = (_mcol_lda_fxc_mat  , 0)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        _rho0 = None
        _vxc = None
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            if rho0 is not None:
                if xctype == 'LDA':
                    _rho0 = np.asarray(rho0[:,p0:p1], order='C')
                else:
                    _rho0 = np.asarray(rho0[:,:,p0:p1], order='C')
            elif make_rho0 is not None:
                _rho0 = make_rho0(0, ao, mask, xctype)

            if fxc is None:
                _fxc = mcfun.eval_xc_eff(
                    fn_eval_xc, _rho0, deriv=2, spin_samples=ni.spin_samples,
                    collinear_threshold=ni.collinear_thrd,
                    collinear_samples=ni.collinear_samples, workers=nproc)[2]
            else:
                _fxc = fxc[:,:,:,:,p0:p1]

            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                vmat[i] += fmat(mol, ao, weight, _rho0, rho1, _vxc, _fxc,
                                mask, shls_slice, ao_loc)
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_fxc for functional {xc_code}')

    vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

def _ncol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc):
    '''Vxc matrix of non-collinear LDA'''
    r, mx, my, mz = rho
    vrho = vxc[0]
    vr, vs = vrho.T
    s = lib.norm(rho[1:4], axis=0)

    idx = s < 1e-20
    with np.errstate(divide='ignore',invalid='ignore'):
        ws = vs * weight / s
    ws[idx] = 0

    # * .5 because of v+v.conj().T in r_vxc
    wv = .5 * weight * vr
    ws *= .5
    aow = None
    aow = _scale_ao(ao, ws*mx, out=aow)  # Mx
    matba = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, ws*my, out=aow)  # My
    tmp = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    matba = matba + tmp * 1j
    matab = matba.conj().T
    aow = _scale_ao(ao, wv+ws*mz, out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wv-ws*mz, out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    mat = np.bmat([[mataa, matab], [matba, matbb]])
    return np.asarray(mat)

def _mcfun_eval_xc(ni, xc_code, rho, deriv, xctype=None):
    if xctype is None:
        xctype = ni._xc_type(xc_code)
    t, s = rho
    if xctype == 'MGGA':
        # Padding for laplacian
        ngrids = rho.shape[-1]
        rhop = np.empty((2, 6, ngrids))
        rhop[0,:4] = (t[:4] + s[:4]) * .5
        rhop[1,:4] = (t[:4] - s[:4]) * .5
        rhop[:,4] = 0
        rhop[0,5] = (t[4] + s[4]) * .5
        rhop[1,5] = (t[4] - s[4]) * .5
    else:
        rhop = np.empty_like(rho)
        rhop[0] = (t + s) * .5
        rhop[1] = (t - s) * .5

    exc, vxc, fxc, kxc = ni.libxc.eval_xc(xc_code, rhop, spin=1, deriv=deriv)

    if deriv > 2:
        kxc = xc_deriv.transform_kxc(rhop, fxc, kxc, xctype, spin=1)
        kxc = xc_deriv.ud2ts(kxc)
    if deriv > 1:
        fxc = xc_deriv.transform_fxc(rhop, vxc, fxc, xctype, spin=1)
        fxc = xc_deriv.ud2ts(fxc)
    if deriv > 0:
        vxc = xc_deriv.transform_vxc(rhop, vxc, xctype, spin=1)
        vxc = xc_deriv.ud2ts(vxc)
    return exc, vxc, fxc, kxc

def mcfun_eval_xc_wrapper(ni, xc_code):
    '''Wrapper to generate the eval_xc function required by mcfun'''
    xctype = ni._xc_type(xc_code)
    return functools.partial(_mcfun_eval_xc, ni, xc_code, xctype=xctype)

def _mcol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc):
    '''Vxc matrix of multi-collinear LDA'''
    # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = .5 * weight * vxc

    # einsum('g,g,xgi,xgj->ij', vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vxc, weight, ao, ao)
    aow = None
    aow = _scale_ao(ao, wmx[0], out=aow)  # Mx
    matba = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wmy[0], out=aow)  # My
    tmp = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    matba = matba + tmp * 1j
    matab = matba.conj().T
    aow = _scale_ao(ao, wr[0]+wmz[0], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wr[0]-wmz[0], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    mat = np.bmat([[mataa, matab], [matba, matbb]])
    return np.asarray(mat)

def _mcol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao, wmx[:4], out=aow)  # Mx
    matba = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wmy[:4], out=aow)  # My
    tmp = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    matba = matba + tmp * 1j
    matab = matba.conj().T
    aow = _scale_ao(ao, wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    mat = np.bmat([[mataa, matab], [matba, matbb]])
    return np.asarray(mat)

def _mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc):
    '''Vxc matrix of multi-collinear MGGA'''
    wv = weight * vxc
    wv[:,0] *= .5
    tau_idx = 4
    wv[:,tau_idx] *= .25
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao, wmx[:4], out=aow)  # Mx
    matba = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    matba += _tau_dot(mol, ao, ao, wmx[tau_idx], mask, shls_slice, ao_loc)

    aow = _scale_ao(ao, wmy[:4], out=aow)  # My
    tmp = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmp += _tau_dot(mol, ao, ao, wmx[tau_idx], mask, shls_slice, ao_loc)
    matba = matba + tmp * 1j
    matab = matba.conj().T

    aow = _scale_ao(ao, wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    mataa += _tau_dot(mol, ao, ao, wr[tau_idx]+wmz[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    matbb += _tau_dot(mol, ao, ao, wr[tau_idx]-wmz[tau_idx], mask, shls_slice, ao_loc)

    mat = np.bmat([[mataa, matab], [matba, matbb]])
    return np.asarray(mat)

def _mcol_lda_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc):
    '''Kernel matrix of multi-collinear LDA'''
    vxc1 = np.einsum('ag,abyg->byg', rho1, fxc[:,0,:])
    return _mcol_lda_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice, ao_loc)

def _mcol_gga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc):
    '''Kernel matrix of multi-collinear GGA'''
    vxc1 = np.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_gga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice, ao_loc)

def _mcol_mgga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                       mask, shls_slice, ao_loc):
    '''Kernel matrix of multi-collinear MGGA'''
    vxc1 = np.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_mgga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice, ao_loc)


def _contract_rho2x2(bra, ket):
    ket_a, ket_b = ket
    bra_a, bra_b = bra
    nao = min(ket_a.shape[1], bra_a.shape[1])
    rhoaa = np.einsum('pi,pi->p', ket_a[:,:nao].real, bra_a.real)
    rhoaa+= np.einsum('pi,pi->p', ket_a[:,:nao].imag, bra_a.imag)
    rhoab = np.einsum('pi,pi->p', ket_a[:,nao:], bra_b.conj())
    rhoba = np.einsum('pi,pi->p', ket_b[:,:nao], bra_a.conj())
    rhobb = np.einsum('pi,pi->p', ket_b[:,nao:].real, bra_b.real)
    rhobb+= np.einsum('pi,pi->p', ket_b[:,nao:].imag, bra_b.imag)
    return rhoaa, rhoab, rhoba, rhobb

def _rho2x2_to_rho_m(rho2x2):
    # rho = einsum('xgi,ij,xgj->g', ket, dm, bra.conj())
    # mx = einsum('xy,ygi,ij,xgj->g', sx, ket, dm, bra.conj())
    # my = einsum('xy,ygi,ij,xgj->g', sy, ket, dm, bra.conj())
    # mz = einsum('xy,ygi,ij,xgj->g', sz, ket, dm, bra.conj())
    raa, rab, rba, rbb = rho2x2
    ngrids = raa.size
    rho_m = np.empty((4, ngrids))
    rho, mx, my, mz = rho_m
    rho[:] = raa.real + rbb.real
    mx[:] = rab.real + rba.real
    my[:] = rba.imag - rab.imag
    mz[:] = raa.real - rbb.real
    return rho_m

class NumInt2C(numint._NumIntMixin):
    '''Numerical integration methods for 2-component basis (used by GKS)'''

    # collinear schemes:
    #   'col' (collinear, by default)
    #   'ncol' (non-collinear)
    #   'mcol' (multi-collinear)
    collinear = getattr(__config__, 'dft_numint_RnumInt_collinear', 'col')
    spin_samples = getattr(__config__, 'dft_numint_RnumInt_spin_samples', 770)
    collinear_thrd = getattr(__config__, 'dft_numint_RnumInt_collinear_thrd', 0.99)
    collinear_samples = getattr(__config__, 'dft_numint_RnumInt_collinear_samples', 200)

    def __init__(self):
        self.omega = None  # RSH paramter

    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
                 with_lapl=True, verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        nao = ao.shape[-1]
        assert dm.ndim == 2 and nao * 2 == dm.shape[0]

        ngrids, nao = ao.shape[-2:]
        xctype = xctype.upper()
        if non0tab is None:
            non0tab = np.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                              dtype=np.uint8)
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc

        if xctype == 'LDA':
            c0a = _dot_ao_dm(mol, ao, dm[:nao], non0tab, shls_slice, ao_loc)
            c0b = _dot_ao_dm(mol, ao, dm[nao:], non0tab, shls_slice, ao_loc)
            tmp = _contract_rho2x2((ao, ao), (c0a, c0b))
            rho_m = _rho2x2_to_rho_m(tmp)
        elif xctype == 'GGA':
            # first 4 ~ (rho, m), second 4 ~ (rho0, dx, dy, dz)
            rho_m = np.empty((4, 4, ngrids))
            c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
            c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
            c0 = (c0a, c0b)
            rho_m[:,0] = _rho2x2_to_rho_m(_contract_rho2x2((ao[0], ao[0]), c0))
            for i in range(1, 4):
                rho_m[:,i] = _rho2x2_to_rho_m(_contract_rho2x2((ao[i], ao[i]), c0))
                rho_m[:,i] *= 2  # *2 for +c.c. corresponding to |dx ao> dm < ao|
        else: # meta-GGA
            if with_lapl:
                rho_m = np.zeros((4, 6, ngrids))
                tau_idx = 5
            else:
                rho_m = np.zeros((4, 5, ngrids))
                tau_idx = 4
            c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
            c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
            c0 = (c0a, c0b)
            rho_m[:,0] = _rho2x2_to_rho_m(_contract_rho2x2((ao[0], ao[0]), c0))
            for i in range(1, 4):
                rho_m[:,i] = _rho2x2_to_rho_m(_contract_rho2x2((ao[i], ao[i]), c0))
                rho_m[:,i] *= 2  # *2 for +c.c. corresponding to |dx ao> dm < ao|
                c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
                c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
                tmp = _contract_rho2x2((ao[i], ao[i]), (c1a, c1b))
                rho_m[:,tau_idx] += _rho2x2_to_rho_m(tmp)
            if with_lapl:
                # TODO: rho_m[:,4] = \nabla^2 rho
                pass
            # tau = 1/2 (\nabla f)^2
            rho_m[:,tau_idx] *= .5
        return rho_m

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  with_lapl=True, verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        if self.collinear[0] in ('n', 'm'):
            # TODO:
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            hermi = 1
            return self.eval_rho(mol, ao, dm, non0tab, xctype, 1, with_lapl, verbose)

        if mo_coeff.dtype == np.double:
            nao = ao.shape[-1]
            assert nao * 2 == mo_coeff.shape[0]
            mo_aR = mo_coeff[:nao]
            mo_bR = mo_coeff[nao:]
            hermi = 1
            rho  = numint.eval_rho2(mol, ao, mo_aR, mo_occ, non0tab, xctype, hermi, verbose)
            rho += numint.eval_rho2(mol, ao, mo_bR, mo_occ, non0tab, xctype, hermi, verbose)
        else:
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            hermi = 1
            rho = self.eval_rho(mol, dm, ao, dm, non0tab, xctype, hermi,
                                with_lapl, verbose)
        mx, my, mz = None
        return rho, mx, my, mz

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        xctype = self._xc_type(xc_code)
        if xctype in ('GGA', 'MGGA'):
            ao_deriv = 1
        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        else:
            ao_deriv = 0
        n2c = mo_coeff.shape[0]
        nao = n2c // 2
        dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)

        if self.collinear[0] == 'm':  # mcol
            rho = []
            for ao, mask, weight, coords \
                    in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
                # rhoa and rhob have to be real
                rho.append(self.eval_rho(mol, ao, dm, mask, xctype,
                                         with_lapl=False))
            rho = np.hstack(rho)
            fn_eval_xc = self.mcfun_eval_xc_wrapper(xc_code)
            nproc = lib.num_threads()
            vxc, fxc = mcfun.eval_xc_eff(
                fn_eval_xc, rho, deriv=2, spin_samples=self.spin_samples,
                collinear_threshold=self.collinear_thrd,
                collinear_samples=self.collinear_samples, workers=nproc)[1:3]
        else:
            dm_a = dm[:nao,:nao].real.copy()
            dm_b = dm[nao:,nao:].real.copy()
            rhoa = []
            rhob = []
            for ao, mask, weight, coords \
                    in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
                # rhoa and rhob have to be real
                rhoa.append(numint.eval_rho(mol, ao, dm_a, mask, xctype))
                rhob.append(numint.eval_rho(mol, ao, dm_b, mask, xctype))
            rho = (np.hstack(rhoa), np.hstack(rhob))
            vxc, fxc = self.eval_xc(xc_code, rho, spin=1, relativity=0, deriv=2,
                                    verbose=0)[1:3]
        return rho, vxc, fxc

    def get_rho(self, mol, dm, grids, max_memory=2000):
        '''Density in real space
        '''
        nao = dm.shape[-1] // 2
        dm_a = dm[:nao,:nao].real
        dm_b = dm[nao:,nao:].real
        ni = self.view(numint.NumInt)
        res = numint.get_rho(ni, mol, dm_a+dm_b, grids, max_memory)
        return res

    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            n, exc, vmat = gks_mcol_vxc(self, mol, grids, xc_code, dms,
                                        relativity, hermi, max_memory, verbose)
        else:
            dms = np.asarray(dms)
            nao = dms.shape[-1] // 2
            # ground state density is always real
            dm_a = dms[...,:nao,:nao].real.copy()
            dm_b = dms[...,nao:,nao:].real.copy()
            ni = self.view(numint.NumInt)
            n, exc, vxc = numint.nr_uks(ni, mol, grids, xc_code, (dm_a, dm_b),
                                        relativity, hermi, max_memory, verbose)
            vmat = np.zeros_like(dms)
            vmat[...,:nao,:nao] = vxc[0]
            vmat[...,nao:,nao:] = vxc[1]
        return n, exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if self.collinear[0] not in ('c', 'm'):  # col or mcol
            raise NotImplementedError('non-collinear fxc')

        if self.collinear[0] == 'm':  # mcol
            fxcmat = gks_mcol_fxc(
                self, mol, grids, xc_code, dm0, dms,
                relativity, hermi, rho0, vxc, fxc, max_memory, verbose)
        else:
            dms = np.asarray(dms)
            nao = dms.shape[-1] // 2
            if dm0 is not None:
                dm0a = dm0[:nao,:nao].real.copy()
                dm0b = dm0[nao:,nao:].real.copy()
                dm0 = (dm0a, dm0b)
            # dms_a and dms_b may be complex if they are TDDFT amplitudes
            dms_a = dms[...,:nao,:nao].copy()
            dms_b = dms[...,nao:,nao:].copy()
            ni = self.view(numint.NumInt)
            vmat = numint.nr_uks_fxc(
                ni, mol, grids, xc_code, dm0, (dms_a, dms_b),
                relativity, hermi, rho0, vxc, fxc, max_memory, verbose)
            fxcmat = np.zeros_like(dms)
            fxcmat[...,:nao,:nao] = vmat[0]
            fxcmat[...,nao:,nao:] = vmat[1]
        return fxcmat
    get_fxc = nr_gks_fxc = nr_fxc

    def eval_xc(ni, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None):
        '''eval_xc for density in rho/m representation. Support LDA only'''
        if omega is None: omega = ni.omega
        if ni.collinear[0] == 'c':  # collinear
            xc = ni.libxc.eval_xc(xc_code, rho, 1, relativity, deriv, omega, verbose)
        elif ni.collinear[0] == 'n':  # ncol
            # only support LDA
            # JTCC, 2, 257
            r = rho[0]
            m = rho[1:4]
            s = lib.norm(m, axis=0)
            rhou = (r + s) * .5
            rhod = (r - s) * .5
            rho = (rhou, rhod)
            xc = ni.libxc.eval_xc(xc_code, rho, 1, relativity, deriv,
                                    omega, verbose)
            exc, vxc = xc[:2]
            # update vxc[0] inplace
            vrho = vxc[0]
            vr, vm = (vrho[:,0]+vrho[:,1])*.5, (vrho[:,0]-vrho[:,1])*.5
            vrho[:,0] = vr
            vrho[:,1] = vm
        elif ni.collinear[0] == 'm':  # mcol
            raise RuntimeError('should not be called for mcol')
        else:
            raise RuntimeError(f'Unknown collinear scheme {ni.collinear}')
        return xc

    mcfun_eval_xc_wrapper = mcfun_eval_xc_wrapper
