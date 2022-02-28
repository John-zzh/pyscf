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
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, _scale_ao, _tau_dot, BLKSIZE
from pyscf.dft import xc_deriv
from pyscf.dft import mcfun
from pyscf import __config__


def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
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
        rho_m = _contract_rho_m((ao, ao), (c0a, c0b), hermi)
    elif xctype == 'GGA':
        # first 4 ~ (rho, m), second 4 ~ (0th order, dx, dy, dz)
        if hermi:
            rho_m = np.empty((4, 4, ngrids))
        else:
            rho_m = np.empty((4, 4, ngrids), dtype=np.complex128)
        c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
        c0 = (c0a, c0b)
        rho_m[:,0] = _contract_rho_m((ao[0], ao[0]), c0, hermi)
        for i in range(1, 4):
            rho_m[:,i] = _contract_rho_m((ao[i], ao[i]), c0, hermi)
        if hermi:
            rho_m[:,1:4] *= 2  # *2 for |ao> dm < dx ao| + |dx ao> dm < ao|
        else:
            for i in range(1, 4):
                c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
                c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
                rho_m[:,i] += _contract_rho_m((ao[0], ao[0]), (c1a, c1b), hermi)
    else: # meta-GGA
        if hermi:
            dtype = np.double
        else:
            dtype = np.complex128
        if with_lapl:
            rho_m = np.empty((4, 6, ngrids), dtype=dtype)
            tau_idx = 5
        else:
            rho_m = np.empty((4, 5, ngrids), dtype=dtype)
            tau_idx = 4
        c0a = _dot_ao_dm(mol, ao[0], dm[:nao], non0tab, shls_slice, ao_loc)
        c0b = _dot_ao_dm(mol, ao[0], dm[nao:], non0tab, shls_slice, ao_loc)
        c0 = (c0a, c0b)
        rho_m[:,0] = _contract_rho_m((ao[0], ao[0]), c0, hermi)
        rho_m[:,tau_idx] = 0
        for i in range(1, 4):
            c1a = _dot_ao_dm(mol, ao[i], dm[:nao], non0tab, shls_slice, ao_loc)
            c1b = _dot_ao_dm(mol, ao[i], dm[nao:], non0tab, shls_slice, ao_loc)
            rho_m[:,tau_idx] += _contract_rho_m((ao[i], ao[i]), (c1a, c1b), hermi)

            rho_m[:,i] = _contract_rho_m((ao[i], ao[i]), c0, hermi)
            if hermi:
                rho_m[:,i] *= 2
            else:
                rho_m[:,i] += _contract_rho_m((ao[0], ao[0]), (c1a, c1b), hermi)
        if with_lapl:
            # TODO: rho_m[:,4] = \nabla^2 rho
            raise NotImplementedError
        # tau = 1/2 (\nabla f)^2
        rho_m[:,tau_idx] *= .5
    return rho_m

def _gks_mcol_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                  max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    make_rho, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi)
    nao = n2c // 2

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'n'): (_ncol_lda_vxc_mat , 0),
            ('LDA' , 'm'): (_mcol_lda_vxc_mat , 0),
            ('GGA' , 'm'): (_mcol_gga_vxc_mat , 1),
            ('MGGA', 'm'): (_mcol_mgga_vxc_mat, 1),
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
                    exc, vxc = ni.eval_xc_deriv(xc_code, rho, deriv=1,
                                                xctype=xctype)[:2]
                if xctype == 'LDA':
                    den = rho[0] * weight
                else:
                    den = rho[0,0] * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                vmat[i] += fmat(mol, ao, weight, rho, vxc, mask, shls_slice,
                                ao_loc, hermi)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_vxc for functional {xc_code}')

    if hermi:
        vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat

def _gks_mcol_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                  rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    assert ni.collinear[0] == 'm'  # mcol
    xctype = ni._xc_type(xc_code)
    if xctype == 'MGGA':
        fmat, ao_deriv = (_mcol_mgga_fxc_mat , 1)
    elif xctype == 'GGA':
        fmat, ao_deriv = (_mcol_gga_fxc_mat  , 1)
    else:
        fmat, ao_deriv = (_mcol_lda_fxc_mat  , 0)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    if rho0 is None and (xctype != 'LDA' or fxc is None):
        make_rho0 = ni._gen_rho_evaluator(mol, dm0, 1)[0]
    else:
        make_rho0 = None

    make_rho1, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi)
    nao = n2c // 2
    vmat = np.zeros((nset,n2c,n2c), dtype=np.complex128)

    fn_eval_xc = ni.mcfun_eval_xc_wrapper(xc_code)
    nproc = lib.num_threads()

    if xctype in ('LDA', 'GGA', 'MGGA'):
        _rho0 = None
        _vxc = None
        deriv = 2
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            if fxc is None:
                if rho0 is not None:
                    if xctype == 'LDA':
                        _rho0 = np.asarray(rho0[:,p0:p1], order='C')
                    else:
                        _rho0 = np.asarray(rho0[:,:,p0:p1], order='C')
                elif make_rho0 is not None:
                    _rho0 = make_rho0(0, ao, mask, xctype)

                _fxc = mcfun.eval_xc_eff(
                    fn_eval_xc, _rho0, deriv, spin_samples=ni.spin_samples,
                    collinear_threshold=ni.collinear_thrd,
                    collinear_samples=ni.collinear_samples, workers=nproc)[2]
            else:
                _fxc = fxc[:,:,:,:,p0:p1]

            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                vmat[i] += fmat(mol, ao, weight, _rho0, rho1, _vxc, _fxc,
                                mask, shls_slice, ao_loc, hermi)
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint2c.get_fxc for functional {xc_code}')

    if hermi:
        vmat = vmat + vmat.conj().transpose(0,2,1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

def _ncol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of non-collinear LDA'''
    # NOTE vxc in u/d representation
    r, mx, my, mz = rho
    vr, vs = vxc
    s = lib.norm(rho[1:4], axis=0)

    idx = s < 1e-20
    with np.errstate(divide='ignore',invalid='ignore'):
        ws = vs * weight / s
    ws[idx] = 0

    # * .5 because of v+v.conj().T in r_vxc
    if hermi:
        wv = .5 * weight * vr
        ws *= .5
    aow = None
    aow = _scale_ao(ao, ws*mx, out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, ws*my, out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    if hermi:
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = tmpx + tmpy * 1j
        matab = matba.conj().T
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
    tmpx = tmpy = None
    aow = _scale_ao(ao, wv+ws*mz, out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wv-ws*mz, out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    mat = np.asarray(np.bmat([[mataa, matab], [matba, matbb]]))
    return mat

def _eval_xc_deriv(ni, xc_code, rho, deriv=1, spin=1, omega=None,
                   xctype=None, verbose=None):
    r'''Returns the derivative tensor against the total density and spin density
    parameters
    [[density_t, (nabla_x)_t, (nabla_y)_t, (nabla_z)_t, tau_t],
     [density_s, (nabla_x)_s, (nabla_y)_s, (nabla_z)_s, tau_s]].

    rather than the XC functional derivatives to sigma (|\nabla \rho|^2)
    returned by eval_xc function
    '''
    if omega is None: omega = ni.omega
    if xctype is None: xctype = ni._xc_type(xc_code)
    if ni.collinear[0] == 'c':  # collinear
        t = rho[0]
        s = rho[3]
    elif ni.collinear[0] == 'n':  # ncol
        t = rho[0]
        m = rho[1:4]
        s = lib.norm(m, axis=0)
    elif ni.collinear[0] == 'm':  # mcol
        # called by mcfun.eval_xc_eff which passes (rho, s) only
        t, s = rho
    else:
        raise RuntimeError(f'Unknown collinear scheme {ni.collinear}')

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
        rhop = np.stack([(t + s) * .5, (t - s) * .5])

    spin = 1
    exc, vxc, fxc, kxc = ni.libxc.eval_xc(xc_code, rhop, spin, deriv=deriv,
                                          omega=omega)

    if deriv > 2:
        kxc = xc_deriv.transform_kxc(rhop, fxc, kxc, xctype, spin)
        if ni.collinear[0] != 'c':
            kxc = xc_deriv.ud2ts(kxc)
    if deriv > 1:
        fxc = xc_deriv.transform_fxc(rhop, vxc, fxc, xctype, spin)
        if ni.collinear[0] != 'c':
            fxc = xc_deriv.ud2ts(fxc)
    if deriv > 0:
        vxc = xc_deriv.transform_vxc(rhop, vxc, xctype, spin)
        if ni.collinear[0] != 'c':
            vxc = xc_deriv.ud2ts(vxc)
    return exc, vxc, fxc, kxc

def mcfun_eval_xc_wrapper(ni, xc_code):
    '''Wrapper to generate the eval_xc function required by mcfun'''
    omega = ni.omega
    xctype = ni._xc_type(xc_code)
    return functools.partial(_eval_xc_deriv, ni, xc_code,
                             omega=omega, xctype=xctype)

def _mcol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    if hermi:
        wv *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    # einsum('g,g,xgi,xgj->ij', vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vxc, weight, ao, ao)
    aow = None
    aow = _scale_ao(ao, wmx[0], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wmy[0], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    if hermi:
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = tmpx + tmpy * 1j
        matab = matba.conj().T
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
    tmpx = tmpy = None
    aow = _scale_ao(ao, wr[0]+wmz[0], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao, wr[0]-wmz[0], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    mat = np.asarray(np.bmat([[mataa, matab], [matba, matbb]]))
    return mat

def _mcol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of multi-collinear LDA'''
    wv = weight * vxc
    if hermi:
        wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao[:4], wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmx[:4], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmy[:4], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == np.double
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = tmpx + tmpy * 1j
        matab = matba.conj().T
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        aow = _scale_ao(ao[1:4], wmx[1:4].conj(), out=aow)  # Mx
        tmpx += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], wmy[1:4].conj(), out=aow)  # My
        tmpy += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
        aow = _scale_ao(ao[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mataa += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        matbb += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

    mat = np.asarray(np.bmat([[mataa, matab], [matba, matbb]]))
    return mat

def _mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, hermi):
    '''Vxc matrix of multi-collinear MGGA'''
    wv = weight * vxc
    tau_idx = 4
    wv[:,tau_idx] *= .5  # *.5 for 1/2 in tau
    if hermi:
        wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
        wv[:,tau_idx] *= .5
    wr, wmx, wmy, wmz = wv

    aow = None
    aow = _scale_ao(ao[:4], wr[:4]+wmz[:4], out=aow)  # Mz
    mataa = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    mataa += _tau_dot(mol, ao, ao, wr[tau_idx]+wmz[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wr[:4]-wmz[:4], out=aow)  # Mz
    matbb = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    matbb += _tau_dot(mol, ao, ao, wr[tau_idx]-wmz[tau_idx], mask, shls_slice, ao_loc)

    aow = _scale_ao(ao[:4], wmx[:4], out=aow)  # Mx
    tmpx = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmpx += _tau_dot(mol, ao, ao, wmx[tau_idx], mask, shls_slice, ao_loc)
    aow = _scale_ao(ao[:4], wmy[:4], out=aow)  # My
    tmpy = _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    tmpy += _tau_dot(mol, ao, ao, wmy[tau_idx], mask, shls_slice, ao_loc)
    if hermi:
        assert vxc.dtype == np.double
        # conj(mx+my*1j) == mx-my*1j, tmpx and tmpy should be real
        matba = tmpx + tmpy * 1j
        matab = matba.conj().T
    else:
        # conj(mx+my*1j) != mx-my*1j, tmpx and tmpy should be complex
        aow = _scale_ao(ao[1:4], wmx[1:4].conj(), out=aow)  # Mx
        tmpx += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], wmy[1:4].conj(), out=aow)  # My
        tmpy += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        matba = tmpx + tmpy * 1j
        matab = tmpx - tmpy * 1j
        aow = _scale_ao(ao[1:4], (wr[1:4]+wmz[1:4]).conj(), out=aow)  # Mz
        mataa += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
        aow = _scale_ao(ao[1:4], (wr[1:4]-wmz[1:4]).conj(), out=aow)  # Mz
        matbb += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

    mat = np.asarray(np.bmat([[mataa, matab], [matba, matbb]]))
    return mat

def _mcol_lda_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc, hermi):
    '''Kernel matrix of multi-collinear LDA'''
    vxc1 = np.einsum('ag,abyg->byg', rho1, fxc[:,0,:])
    return _mcol_lda_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi)

def _mcol_gga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc, hermi):
    '''Kernel matrix of multi-collinear GGA'''
    vxc1 = np.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_gga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                             ao_loc, hermi)

def _mcol_mgga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                       mask, shls_slice, ao_loc, hermi):
    '''Kernel matrix of multi-collinear MGGA'''
    vxc1 = np.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_mgga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice,
                              ao_loc, hermi)

def _contract_rho_m(bra, ket, hermi=0):
    # rho = einsum('xgi,ij,xgj->g', ket, dm, bra.conj())
    # mx = einsum('xy,ygi,ij,xgj->g', sx, ket, dm, bra.conj())
    # my = einsum('xy,ygi,ij,xgj->g', sy, ket, dm, bra.conj())
    # mz = einsum('xy,ygi,ij,xgj->g', sz, ket, dm, bra.conj())
    ket_a, ket_b = ket
    bra_a, bra_b = bra
    nao = min(ket_a.shape[1], bra_a.shape[1])
    ngrids = ket_a.shape[0]
    if hermi:
        rho_m = np.empty((4, ngrids))
        raa = np.einsum('pi,pi->p', ket_a[:,:nao].real, bra_a.real)
        raa+= np.einsum('pi,pi->p', ket_a[:,:nao].imag, bra_a.imag)
        rba = np.einsum('pi,pi->p', ket_b[:,:nao], bra_a.conj())
        rbb = np.einsum('pi,pi->p', ket_b[:,nao:].real, bra_b.real)
        rbb+= np.einsum('pi,pi->p', ket_b[:,nao:].imag, bra_b.imag)
        rho_m[0,:] = raa + rbb     # rho
        rho_m[1,:] = rba.real * 2  # mx
        rho_m[2,:] = rba.imag * 2  # my
        rho_m[3,:] = raa - rbb     # mz
    else:
        raa = np.einsum('pi,pi->p', ket_a[:,:nao], bra_a.conj())
        rba = np.einsum('pi,pi->p', ket_a[:,nao:], bra_b.conj())
        rab = np.einsum('pi,pi->p', ket_b[:,:nao], bra_a.conj())
        rbb = np.einsum('pi,pi->p', ket_b[:,nao:], bra_b.conj())
        rho_m = np.empty((4, ngrids), dtype=np.complex128)
        rho_m[0,:] = raa + rbb         # rho
        rho_m[1,:] = rab + rba         # mx
        rho_m[2,:] = (rba - rab) * 1j  # my
        rho_m[3,:] = raa - rbb         # mz
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

    @lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
                 with_lapl=True, verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  with_lapl=True, verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        if self.collinear[0] in ('n', 'm'):
            # TODO:
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            hermi = 1
            rho = self.eval_rho(mol, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
            return rho

        #if mo_coeff.dtype == np.double:
        #    nao = ao.shape[-1]
        #    assert nao * 2 == mo_coeff.shape[0]
        #    mo_aR = mo_coeff[:nao]
        #    mo_bR = mo_coeff[nao:]
        #    rho  = numint.eval_rho2(mol, ao, mo_aR, mo_occ, non0tab, xctype, with_lapl, verbose)
        #    rho += numint.eval_rho2(mol, ao, mo_bR, mo_occ, non0tab, xctype, with_lapl, verbose)
        #else:
        #    dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
        #    hermi = 1
        #    rho = self.eval_rho(mol, dm, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
        #mx = my = mz = None
        #return rho, mx, my, mz
        raise NotImplementedError(self.collinear)

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

        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            with_lapl = False
            rho = []
            for ao, mask, weight, coords \
                    in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho.append(self.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype,
                                          with_lapl))
            rho = np.hstack(rho)
            if self.collinear[0] == 'm':
                fn_eval_xc = self.mcfun_eval_xc_wrapper(xc_code)
                nproc = lib.num_threads()
                vxc, fxc = mcfun.eval_xc_eff(
                    fn_eval_xc, rho, deriv=2, spin_samples=self.spin_samples,
                    collinear_threshold=self.collinear_thrd,
                    collinear_samples=self.collinear_samples, workers=nproc)[1:3]
            else:
                vxc, fxc = self.eval_xc_deriv(xc_code, rho, deriv=2,
                                              xctype=xctype)[1:3]
        else:
            # rhoa and rhob must be real
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            dm_a = dm[:nao,:nao].real.copy()
            dm_b = dm[nao:,nao:].real.copy()
            ni = self._to_numint1c()
            with_lapl = True
            hermi = 1
            rhoa = []
            rhob = []
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                # rhoa and rhob must be real
                rhoa.append(ni.eval_rho(mol, ao, dm_a, mask, xctype, hermi, with_lapl))
                rhob.append(ni.eval_rho(mol, ao, dm_b, mask, xctype, hermi, with_lapl))
            rho = np.stack([np.hstack(rhoa), np.hstack(rhob)])
            assert rho.dtype == np.double
            vxc, fxc = ni.eval_xc(xc_code, rho, spin=1, relativity=0, deriv=2,
                                  verbose=0)[1:3]
        return rho, vxc, fxc

    def get_rho(self, mol, dm, grids, max_memory=2000):
        '''Density in real space
        '''
        nao = dm.shape[-1] // 2
        dm_a = dm[:nao,:nao].real
        dm_b = dm[nao:,nao:].real
        ni = self._to_numint1c()
        return ni.get_rho(mol, dm_a+dm_b, grids, max_memory)

    _gks_mcol_vxc = _gks_mcol_vxc
    _gks_mcol_fxc = _gks_mcol_fxc

    @lib.with_doc(numint.nr_rks.__doc__)
    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if self.collinear[0] in ('m', 'n'):  # mcol or ncol
            n, exc, vmat = self._gks_mcol_vxc(mol, grids, xc_code, dms, relativity,
                                              hermi, max_memory, verbose)
        else:
            nao = dms.shape[-1] // 2
            # ground state density is always real
            dm_a = dms[...,:nao,:nao].real.copy()
            dm_b = dms[...,nao:,nao:].real.copy()
            dm1 = (dm_a, dm_b)
            ni = self._to_numint1c()
            n, exc, v = ni.nr_uks(mol, grids, xc_code, dm1, relativity,
                                  hermi, max_memory, verbose)
            vmat = np.zeros_like(dms)
            vmat[...,:nao,:nao] = v[0]
            vmat[...,nao:,nao:] = v[1]
        return n, exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    @lib.with_doc(numint.nr_rks_fxc.__doc__)
    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if self.collinear[0] not in ('c', 'm'):  # col or mcol
            raise NotImplementedError('non-collinear fxc')

        if self.collinear[0] == 'm':  # mcol
            fxcmat = self._gks_mcol_fxc(mol, grids, xc_code, dm0, dms,
                                        relativity, hermi, rho0, vxc, fxc,
                                        max_memory, verbose)
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
            dm1 = (dms_a, dms_b)
            ni = self._to_numint1c()
            vmat = ni.nr_uks_fxc(mol, grids, xc_code, dm0, dm1, relativity,
                                 hermi, rho0, vxc, fxc, max_memory, verbose)
            fxcmat = np.zeros_like(dms)
            fxcmat[...,:nao,:nao] = vmat[0]
            fxcmat[...,nao:,nao:] = vmat[1]
        return fxcmat
    get_fxc = nr_gks_fxc = nr_fxc

    def eval_xc(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None):
        '''eval_xc for density in rho/m representation. Support LDA only'''
        if omega is None: omega = self.omega
        if self.collinear[0] == 'c':  # collinear
            xc = self.libxc.eval_xc(xc_code, rho, 1, relativity, deriv, omega, verbose)
        elif self.collinear[0] == 'n':  # ncol
            # only support LDA
            # JTCC, 2, 257
            r = rho[0]
            m = rho[1:4]
            s = lib.norm(m, axis=0)
            rhou = (r + s) * .5
            rhod = (r - s) * .5
            rho = (rhou, rhod)
            xc = self.libxc.eval_xc(xc_code, rho, 1, relativity, deriv,
                                    omega, verbose)
            exc, vxc = xc[:2]
            # update vxc[0] inplace
            vrho = vxc[0]
            vr, vm = (vrho[:,0]+vrho[:,1])*.5, (vrho[:,0]-vrho[:,1])*.5
            vrho[:,0] = vr
            vrho[:,1] = vm
        elif self.collinear[0] == 'm':  # mcol
            raise RuntimeError('should not be called for mcol')
        else:
            raise RuntimeError(f'Unknown collinear scheme {self.collinear}')
        return xc

    eval_xc_deriv = _eval_xc_deriv
    mcfun_eval_xc_wrapper = mcfun_eval_xc_wrapper

    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=False):
        if getattr(dms, 'mo_coeff', None) is not None:
            #TODO: test whether dm.mo_coeff matching dm
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms, np.ndarray) and dms.ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = mo_coeff[0].shape[0]
            ndms = len(mo_occ)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(mol, ao, mo_coeff[idm], mo_occ[idm],
                                      non0tab, xctype, with_lapl)
        else:
            if isinstance(dms, np.ndarray) and dms.ndim == 2:
                dms = [dms]
            if not hermi and dms[0].dtype == np.double:
                # (D + D.T)/2 because eval_rho computes 2*(|\nabla i> D_ij <j|) instead of
                # |\nabla i> D_ij <j| + |i> D_ij <\nabla j| for efficiency
                dms = [(dm+dm.conj().T)*.5 for dm in dms]
                hermi = 1
            nao = dms[0].shape[0]
            ndms = len(dms)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho(mol, ao, dms[idm], non0tab, xctype,
                                     hermi, with_lapl)
        return make_rho, ndms, nao

    def _to_numint1c(self):
        '''Converts to the associated class to handle collinear systems'''
        return self.view(numint.NumInt)
