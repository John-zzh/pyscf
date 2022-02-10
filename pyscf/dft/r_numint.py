#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Numerical integration functions for (4-component) DKS with j-adapted AO basis
'''

import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import _dot_ao_dm, _dot_ao_ao, _scale_ao, _tau_dot, BLKSIZE
from pyscf.dft import numint2c
from pyscf.dft import mcfun
from pyscf import __config__


def eval_ao(mol, coords, deriv=0, with_s=True, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    '''Evaluates the value of 2-component or 4-component j-adapted basis on grids.
    '''
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    if not with_s:
        # aoLa, aoLb = aoL
        ao = aoL = mol.eval_gto(feval, coords, comp, shls_slice, non0tab, out=out)
    else:
        assert(deriv <= 1)  # only GTOval_ipsp_spinor
        ngrids = coords.shape[0]
        nao = mol.nao_2c()
        ao = numpy.ndarray((4,comp,nao,ngrids), dtype=numpy.complex128, buffer=out)
        aoL = mol.eval_gto(feval, coords, comp, shls_slice, non0tab, out=ao[:2])  # noqa
        ao = ao.transpose(0,1,3,2)
        aoS = ao[2:]
        aoSa, aoSb = aoS
        feval_gto = ['GTOval_sp_spinor', 'GTOval_ipsp_spinor']
        p1 = 0
        for n in range(deriv+1):
            comp = (n+1)*(n+2)//2
            p0, p1 = p1, p1 + comp
            aoSa[p0:p1], aoSb[p0:p1] = mol.eval_gto(
                feval_gto[n], coords, comp, shls_slice, non0tab)

        if deriv == 0:
            ao = ao[:,0]
    return ao

def _dot_spinor_dm(mol, ket, dm, non0tab, shls_slice, ao_loc):
    ket_a, ket_b = ket
    outa = _dot_ao_dm(mol, ket_a, dm, non0tab, shls_slice, ao_loc)
    outb = _dot_ao_dm(mol, ket_b, dm, non0tab, shls_slice, ao_loc)
    return outa, outb

def _contract_rho2x2(bra, ket):
    '''rho2x2 = sum_i |ket_i> <bra_i|
    '''
    ket_a, ket_b = ket
    bra_a, bra_b = bra
    rhoaa = numpy.einsum('pi,pi->p', ket_a.real, bra_a.real)
    rhoaa+= numpy.einsum('pi,pi->p', ket_a.imag, bra_a.imag)
    rhoab = numpy.einsum('pi,pi->p', ket_a, bra_b.conj())
    rhoba = numpy.einsum('pi,pi->p', ket_b, bra_a.conj())
    rhobb = numpy.einsum('pi,pi->p', ket_b.real, bra_b.real)
    rhobb+= numpy.einsum('pi,pi->p', ket_b.imag, bra_b.imag)
    return rhoaa, rhoab, rhoba, rhobb

def _rho2x2_to_rho_m(rho2x2):
    # rho = einsum('xgi,ij,xgj->g', ket, dm, bra.conj())
    # mx = einsum('xy,ygi,ij,xgj->g', sx, ket, dm, bra.conj())
    # my = einsum('xy,ygi,ij,xgj->g', sy, ket, dm, bra.conj())
    # mz = einsum('xy,ygi,ij,xgj->g', sz, ket, dm, bra.conj())
    raa, rab, rba, rbb = rho2x2
    ngrids = raa.size
    rho_m = numpy.empty((4, ngrids))
    rho, mx, my, mz = rho_m
    rho[:] = raa.real + rbb.real
    mx[:] = rab.real + rba.real
    my[:] = rba.imag - rab.imag
    mz[:] = raa.real - rbb.real
    return rho_m

def _get_rho_m(mol, ket, dm, bra, non0tab, shls_slice, ao_loc):
    out = _dot_spinor_dm(mol, ket, dm, non0tab, shls_slice, ao_loc)
    tmp = _contract_rho2x2(bra, out)
    return _rho2x2_to_rho_m(tmp)

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    aoa, aob = ao
    ngrids, nao = aoa.shape[-2:]
    xctype = xctype.upper()

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()

    if xctype == 'LDA':
        rho_m = _get_rho_m(mol, ao, dm, ao, non0tab, shls_slice, ao_loc)
    elif xctype == 'GGA':
        # first 4 ~ (rho, m), second 4 ~ (rho0, dx, dy, dz)
        rho_m = numpy.empty((4, 4, ngrids))
        c0 = _dot_spinor_dm(mol, ao[:,0], dm, non0tab, shls_slice, ao_loc)
        rho_m[:,0] = _rho2x2_to_rho_m(_contract_rho2x2(ao[:,0], c0))
        for i in range(1, 4):
            rho_m[:,i] = _rho2x2_to_rho_m(_contract_rho2x2(ao[:,i], c0))
            rho_m[:,i] *= 2  # *2 for +c.c. corresponding to |dx ao> dm < ao|
    else: # meta-GGA
        rho_m = numpy.zeros((4, 6, ngrids))
        c0 = _dot_spinor_dm(mol, ao[:,0], dm, non0tab, shls_slice, ao_loc)
        rho_m[:,0] = _rho2x2_to_rho_m(_contract_rho2x2(ao[:,0], c0))
        for i in range(1, 4):
            rho_m[:,i] = _rho2x2_to_rho_m(_contract_rho2x2(ao[:,i], c0))
            rho_m[:,i] *= 2  # *2 for +c.c. corresponding to |dx ao> dm < ao|
            c1 = _dot_spinor_dm(mol, ao[:,1], dm, non0tab, shls_slice, ao_loc)
            rho_m[:,5] += _rho2x2_to_rho_m(_contract_rho2x2(ao[i], c1))
        # TODO: rho_m[:,4] = \nabla^2 rho
        # tau = 1/2 (\nabla f)^2
        rho_m[:,5] *= .5
    return rho_m

def _ncol_lda_vxc_mat(mol, ao, weight, rho_tm, vxc, mask, shls_slice, ao_loc,
                      on_LL=True):
    '''Vxc matrix of non-collinear LDA'''
    aoa, aob = ao
    r, mx, my, mz = rho_tm
    vrho = vxc[0]
    vr, vs = vrho.T
    s = lib.norm(rho_tm[1:4], axis=0)

    idx = s < 1e-20
    with numpy.errstate(divide='ignore',invalid='ignore'):
        ws = vs * weight / s
    ws[idx] = 0

    # * .5 because of v+v.conj().T in r_vxc
    wv = .5 * weight * vr
    if on_LL:
        ws *= .5
    else:  # for SS block
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        ws *= -.5

    # einsum('g,g,xgi,xgj->ij', vr, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vs*m[0]/s, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vs*m[1]/s, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vs*m[2]/s, weight, ao, ao)
    aow = None
    aow = _scale_ao(aoa, ws*mx, out=aow)  # Mx
    tmp = _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    mat = tmp + tmp.conj().T
    aow = _scale_ao(aoa, ws*my, out=aow)  # My
    tmp = _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    mat+= (tmp - tmp.conj().T) * 1j
    aow = _scale_ao(aoa, wv+ws*mz, out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wv-ws*mz, out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    return mat

def _ncol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                      on_LL=True):
    '''Vxc matrix of non-collinear GGA'''
    raise NotImplementedError

def _ncol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                       on_LL=True):
    '''Vxc matrix of non-collinear MGGA'''
    raise NotImplementedError

def _dks_gga_wv0(rho, vxc, weight):
    r, mx, my, mz = rho
    rhoa = (r + mz) * .5
    rhob = (r - mz) * .5
    return numint._uks_gga_wv0((rhoa, rhob), vxc, weight)

def _col_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                     on_LL=True):
    '''Vxc matrix of collinear LDA'''
    vrho = vxc[0]
    aoa, aob = ao
    if on_LL:
        wva, wvb = .5 * weight * vrho.T
    else:  # for SS block
        # v_rho = (vxc_a + vxc_b) * .5
        # v_mz  = (vxc_a - vxc_b) * .5
        # For small components, M = \beta\Sigma leads to
        # (v_rho - sigma_z*v_mz) = [vxc_b, 0    ]
        #                          [0    , vxc_a]
        wvb, wva = .5 * weight * vrho.T
    mat  = _dot_ao_ao(mol, aoa, _scale_ao(aoa, wva), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob, _scale_ao(aob, wvb), mask, shls_slice, ao_loc)
    return mat

def _col_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                     on_LL=True):
    '''Vxc matrix of collinear GGA'''
    if on_LL:
        wva, wvb = _dks_gga_wv0(rho, vxc, weight)
    else:  # for SS block
        wvb, wva = _dks_gga_wv0(rho, vxc, weight)
    aoa, aob = ao
    mat  = _dot_ao_ao(mol, aoa[0], _scale_ao(aoa[:4], wva[:4]), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob[0], _scale_ao(aob[:4], wvb[:4]), mask, shls_slice, ao_loc)
    return mat

def _col_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc,
                      on_LL=True):
    '''Vxc matrix of collinear MGGA'''
    if on_LL:
        wva, wvb = _dks_gga_wv0(rho, vxc, weight)
    else:
        wvb, wva = _dks_gga_wv0(rho, vxc, weight)
    aoa, aob = ao
    mat  = _dot_ao_ao(mol, aoa[0], _scale_ao(aoa[:4], wva[:4]), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob[0], _scale_ao(aob[:4], wvb[:4]), mask, shls_slice, ao_loc)

    vtau = vxc[3]
    if on_LL:
        wva, wvb = .25 * weight * vtau.T
    else:
        wvb, wva = .25 * weight * vtau.T
    mat += _dot_ao_ao(mol, aoa[1], _scale_ao(aoa[1], wva), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aoa[2], _scale_ao(aoa[2], wva), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aoa[3], _scale_ao(aoa[3], wva), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob[1], _scale_ao(aob[1], wvb), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob[2], _scale_ao(aob[2], wvb), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, aob[3], _scale_ao(aob[3], wvb), mask, shls_slice, ao_loc)
    return mat

def _mcol_lda_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, on_LL=True):
    '''Vxc matrix of multi-collinear LDA'''
    if on_LL:
        # * .5 because of v+v.conj().T in r_vxc
        wv = .5 * weight * vxc
    else:  # for SS block
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        wv = -.5 * weight * vxc
    wr, wmx, wmy, wmz = wv

    aoa, aob = ao
    # einsum('g,g,xgi,xgj->ij', vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sx, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sy, vxc, weight, ao, ao)
    # + einsum('xy,g,g,xgi,ygj->ij', sz, vxc, weight, ao, ao)
    aow = None
    aow = _scale_ao(aoa, wmx[0], out=aow)  # Mx
    tmp = _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    mat = tmp + tmp.conj().T
    aow = _scale_ao(aoa, wmy[0], out=aow)  # My
    tmp = _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    mat+= (tmp - tmp.conj().T) * 1j
    aow = _scale_ao(aoa, wr[0]+wmz[0], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa, aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wr[0]-wmz[0], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob, aow, mask, shls_slice, ao_loc)
    return mat

def _mcol_gga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, on_LL=True):
    '''Vxc matrix of multi-collinear LDA'''
    if on_LL:
        wv = weight * vxc
    else:  # for SS block
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        wv = -weight * vxc
    wv[:,0] *= .5  # * .5 because of v+v.conj().T in r_vxc
    wr, wmx, wmy, wmz = wv

    aoa, aob = ao
    aow = None
    aow = _scale_ao(aoa, wmx[:4], out=aow)  # Mx
    tmp = _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    mat = tmp + tmp.conj().T
    aow = _scale_ao(aoa, wmy[:4], out=aow)  # My
    tmp = _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    mat+= (tmp - tmp.conj().T) * 1j
    aow = _scale_ao(aoa, wr[:4]+wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wr[:4]-wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    return mat

def _mcol_mgga_vxc_mat(mol, ao, weight, rho, vxc, mask, shls_slice, ao_loc, on_LL=True):
    '''Vxc matrix of multi-collinear MGGA'''
    if on_LL:
        wv = weight * vxc
    else:  # for SS block
        # flip the sign for small components because
        # M = \beta\Sigma = [sigma,  0    ]
        #                   [0    , -sigma]
        wv = -weight * vxc
    # * .5 because of v+v.conj().T in r_vxc
    wv[:,0] *= .5
    wv[:,5] *= .5*.5  # *.5 for 1/2 in tau
    wr, wmx, wmy, wmz = wv

    aoa, aob = ao
    aow = None
    aow = _scale_ao(aoa, wmx[:4], out=aow)  # Mx
    tmp = _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    tmp += _tau_dot(mol, aob, aoa, wmx[5], mask, shls_slice, ao_loc)
    mat = tmp + tmp.conj().T

    aow = _scale_ao(aoa, wmy[:4], out=aow)  # My
    tmp = _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    tmp += _tau_dot(mol, aob, aoa, wmx[5], mask, shls_slice, ao_loc)
    mat+= (tmp - tmp.conj().T) * 1j

    aow = _scale_ao(aoa, wr[:4]+wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aoa[0], aow, mask, shls_slice, ao_loc)
    mat+= _tau_dot(mol, aoa, aoa, wr[5]+wmz[5], mask, shls_slice, ao_loc)
    aow = _scale_ao(aob, wr[:4]-wmz[:4], out=aow)  # Mz
    mat+= _dot_ao_ao(mol, aob[0], aow, mask, shls_slice, ao_loc)
    mat+= _tau_dot(mol, aob, aob, wr[5]-wmz[5], mask, shls_slice, ao_loc)
    return mat

def r_vxc(ni, mol, grids, xc_code, dms, spin=0, relativity=1, hermi=1,
          max_memory=2000, verbose=None):
    '''Calculate 2-component or 4-component Vxc matrix in j-adapted basis
    '''
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    matLL = numpy.zeros((nset,n2c,n2c), dtype=numpy.complex128)
    matSS = numpy.zeros_like(matLL)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'c'): (_col_lda_vxc_mat  , 0),
            ('GGA' , 'c'): (_col_gga_vxc_mat  , 1),
            ('MGGA', 'c'): (_col_mgga_vxc_mat , 2),
            ('LDA' , 'n'): (_ncol_lda_vxc_mat , 0),
            ('GGA' , 'n'): (_ncol_gga_vxc_mat , 1),
            ('MGGA', 'n'): (_ncol_mgga_vxc_mat, 2),
            ('LDA' , 'm'): (_mcol_lda_vxc_mat , 0),
            ('GGA' , 'm'): (_mcol_gga_vxc_mat , 1),
            ('MGGA', 'm'): (_mcol_mgga_vxc_mat, 2),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]

        if ni.collinear[0] == 'm':  # mcol
            fn_eval_xc = ni.mcfun_eval_xc_wrapper(xc_code)

        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, with_s, max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                if ni.collinear[0] == 'm':
                    exc, vxc = mcfun.eval_xc_eff(fn_eval_xc, rho, deriv=1, xctype=xctype,
                                                 ang_samples=ni.ang_samples)[:2]
                else:
                    exc, vxc = ni.eval_xc(xc_code, rho, spin=1,
                                          relativity=relativity, deriv=1,
                                          verbose=verbose)[:2]
                den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)

                matLL[i] += fmat(mol, ao[:2], weight, rho, vxc,
                                 mask, shls_slice, ao_loc, True)
                if with_s:
                    matSS[i] += fmat(mol, ao[2:], weight, rho, vxc,
                                     mask, shls_slice, ao_loc, False)

        matLL = matLL + matLL.conj().transpose(0,2,1)
        if with_s:
            matSS = matSS + matSS.conj().transpose(0,2,1)
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'r_vxc for functional {xc_code}')

    if with_s:
        matSS *= (.5 / lib.param.LIGHT_SPEED)**2
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        vmat[:,:n2c,:n2c] = matLL
        vmat[:,n2c:,n2c:] = matSS
    else:
        vmat = matLL

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat.reshape(dms.shape)

def _dks_lda_wv1(rho0, rho1, vxc, fxc, weight):
    u_u, u_d, d_d = fxc[0].T
    rho1, m1x, m1y, m1z = rho1
    rho1a = (rho1 + m1z) * .5
    rho1b = (rho1 - m1z) * .5
    # * .5 because v+v.T is applied in the caller
    wva = (u_u * rho1a + u_d * rho1b) * weight * .5
    wvb = (u_d * rho1a + d_d * rho1b) * weight * .5
    return wva, wvb

def _dks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    r0, m0x, m0y, m0z = rho0
    r1, m1x, m1y, m1z = rho1
    rho0 = ((r0 + m0z) * .5, (r0 - m0z) * .5)
    rho1 = ((r1 + m1z) * .5, (r1 - m1z) * .5)
    return numint._uks_gga_wv1(rho0, rho1, vxc, fxc, weight)

def _dks_mgga_wv1(rho0, rho1, vxc, fxc, weight):
    r0, m0x, m0y, m0z = rho0
    r1, m1x, m1y, m1z = rho1
    rho0 = ((r0 + m0z) * .5, (r0 - m0z) * .5)
    rho1 = ((r1 + m1z) * .5, (r1 - m1z) * .5)
    return numint._uks_mgga_wv1(rho0, rho1, vxc, fxc, weight)

def _col_lda_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                     mask, shls_slice, ao_loc, on_LL=True):
    '''Kernel matrix of collinear LDA'''
    if on_LL:
        wva, wvb = _dks_lda_wv1(rho0, rho1, vxc, fxc, weight)
    else:
        wvb, wva = _dks_lda_wv1(rho0, rho1, vxc, fxc, weight)
    aoa, aob = ao
    mat  = _dot_ao_ao(mol, _scale_ao(aoa, wva), aoa, mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, _scale_ao(aob, wvb), aob, mask, shls_slice, ao_loc)
    return mat

def _col_gga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                     mask, shls_slice, ao_loc, on_LL=True):
    '''Kernel matrix of collinear GGA'''
    if on_LL:
        wva, wvb = _dks_gga_wv1(rho0, rho1, vxc, fxc, weight)
    else:
        wvb, wva = _dks_gga_wv1(rho0, rho1, vxc, fxc, weight)
    aoa, aob = ao
    aow = None
    aow = _scale_ao(aoa[:4], wva[:4], aow)
    mat = _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
    aow = _scale_ao(aob[:4], wvb[:4], aow)
    mat += _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)
    return mat

def _col_mgga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc, on_LL=True):
    '''Kernel matrix of collinear MGGA'''
    if on_LL:
        wva, wvb = _dks_mgga_wv1(rho0, rho1, vxc, fxc, weight)
    else:
        wvb, wva = _dks_mgga_wv1(rho0, rho1, vxc, fxc, weight)
    aoa, aob = ao
    aow = None
    aow = _scale_ao(aoa[:4], wva[:4], aow)
    mat = _dot_ao_ao(mol, aow, aoa[0], mask, shls_slice, ao_loc)
    aow = _scale_ao(aob[:4], wvb[:4], aow)
    mat += _dot_ao_ao(mol, aow, aob[0], mask, shls_slice, ao_loc)

    mat += _dot_ao_ao(mol, ao[1], _scale_ao(ao[1], wva[5], aow), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, ao[2], _scale_ao(ao[2], wva[5], aow), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, ao[3], _scale_ao(ao[3], wva[5], aow), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, ao[1], _scale_ao(ao[1], wvb[5], aow), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, ao[2], _scale_ao(ao[2], wvb[5], aow), mask, shls_slice, ao_loc)
    mat += _dot_ao_ao(mol, ao[3], _scale_ao(ao[3], wvb[5], aow), mask, shls_slice, ao_loc)
    return mat

def _mcol_lda_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc, on_LL=True):
    '''Kernel matrix of multi-collinear LDA'''
    vxc1 = numpy.einsum('ag,abyg->byg', rho1, fxc[:,0,:])
    return _mcol_lda_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice, ao_loc, on_LL)

def _mcol_gga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                      mask, shls_slice, ao_loc, on_LL=True):
    '''Kernel matrix of multi-collinear GGA'''
    vxc1 = numpy.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_gga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice, ao_loc, on_LL)

def _mcol_mgga_fxc_mat(mol, ao, weight, rho0, rho1, vxc, fxc,
                       mask, shls_slice, ao_loc, on_LL=True):
    '''Kernel matrix of multi-collinear MGGA'''
    vxc1 = numpy.einsum('axg,axbyg->byg', rho1, fxc)
    return _mcol_mgga_vxc_mat(mol, ao, weight, rho0, vxc1, mask, shls_slice, ao_loc, on_LL)

def r_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=1, hermi=0,
          rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Calculate 2-component or 4-component Vxc matrix in j-adapted basis
    '''
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]
    if ni.collinear[0] not in ('c', 'm'):  # col or mcol
        raise NotImplementedError('non-collinear fxc')

    make_rho1, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    with_s = (nao == n2c*2)  # 4C DM

    if rho0 is None and (xctype != 'LDA' or fxc is None):
        make_rho0 = ni._gen_rho_evaluator(mol, dm0, 1)[0]
    else:
        make_rho0 = None

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_2c()

    matLL = numpy.zeros((nset,n2c,n2c), dtype=dms.dtype)
    matSS = numpy.zeros_like(matLL)
    if xctype in ('LDA', 'GGA', 'MGGA'):
        f_eval_mat = {
            ('LDA' , 'c'): (_col_lda_fxc_mat   , 0),
            ('GGA' , 'c'): (_col_gga_fxc_mat   , 1),
            ('MGGA', 'c'): (_col_mgga_fxc_mat  , 2),
            ('LDA' , 'm'): (_mcol_lda_fxc_mat  , 0),
            ('GGA' , 'm'): (_mcol_gga_fxc_mat  , 1),
            ('MGGA', 'm'): (_mcol_mgga_fxc_mat , 2),
        }
        fmat, ao_deriv = f_eval_mat[(xctype, ni.collinear[0])]
        if ni.collinear[0] == 'm':  # mcol
            fn_eval_xc = ni.mcfun_eval_xc_wrapper(xc_code)

        _rho0 = None
        _vxc = None
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            if rho0 is not None:
                if xctype == 'LDA':
                    _rho0 = numpy.asarray(rho0[:,p0:p1], order='C')
                else:
                    _rho0 = numpy.asarray(rho0[:,:,p0:p1], order='C')
            elif make_rho0 is not None:
                _rho0 = make_rho0(0, ao, mask, xctype)

            if fxc is None:
                if ni.collinear[0] == 'm':  # mcol
                    _vxc, _fxc = mcfun.eval_xc_eff(fn_eval_xc, _rho0, 2, xctype,
                                                   ang_samples=ni.ang_samples)[1:3]
                else:
                    _vxc, _fxc = ni.eval_xc(xc_code, _rho0, spin=1,
                                            relativity=relativity, deriv=2,
                                            verbose=verbose)[1:3]
            else:
                if ni.collinear[0] == 'm':
                    _fxc = [None if x is None else x[:,:,:,:,p0:p1] for x in fxc]
                else:
                    _vxc = [None if x is None else x[p0:p1] for x in vxc]
                    _fxc = [None if x is None else x[p0:p1] for x in fxc]

            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                matLL[i] += fmat(mol, ao[:2], weight, _rho0, rho1, _vxc, _fxc,
                                 mask, shls_slice, ao_loc)
                if with_s:
                    matSS[i] += fmat(mol, ao[2:], weight, _rho0, rho1, _vxc, _fxc,
                                     mask, shls_slice, ao_loc, False)

        # for (\nabla\mu) \nu + \mu (\nabla\nu)
        matLL = matLL + matLL.conj().transpose(0,2,1)
        if with_s:
            matSS = matSS + matSS.conj().transpose(0,2,1)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'r_fxc for functional {xc_code}')

    if with_s:
        matSS *= (.5 / lib.param.LIGHT_SPEED)**2
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        vmat[:,:n2c,:n2c] = matLL
        vmat[:,n2c:,n2c:] = matSS
    else:
        vmat = matLL

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

def cache_xc_kernel(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=1,
                    max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'MGGA':
        ao_deriv = 2
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        ao_deriv = 0

    dm = numpy.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi)
    rho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        rho = make_rho(0, ao, mask, xctype)
    rho = numpy.hstack(rho)

    if ni.collinear[0] == 'm':  # mcol
        fn_eval_xc = ni.mcfun_eval_xc_wrapper(xc_code)
        vxc, fxc = mcfun.eval_xc_eff(fn_eval_xc, rho, deriv=2, xctype=xctype,
                                     ang_samples=ni.ang_samples)[1:3]
    else:
        vxc, fxc = ni.eval_xc(xc_code, rho, spin=spin, relativity=1, deriv=2,
                              verbose=0)[1:3]
    return rho, vxc, fxc

def get_rho(ni, mol, dm, grids, max_memory=2000):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi=1)
    n2c = mol.nao_2c()
    with_s = (nao == n2c*2)  # 4C DM
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, with_s, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao, mask, 'LDA')[0]
    return rho


class RNumInt(numint._NumIntMixin):
    '''NumInt for j-adapted (spinor) basis'''

    # collinear schemes:
    #   'col' (collinear, by default)
    #   'ncol' (non-collinear)
    #   'mcol' (multi-collinear)
    collinear = getattr(__config__, 'dft_numint_RnumInt_collinear', 'col')
    ang_samples = getattr(__config__, 'dft_numint_RnumInt_ang_samples', 5810)

    def __init__(self):
        self.omega = None  # RSH paramter

    mcfun_eval_xc_wrapper = numint2c.mcfun_eval_xc_wrapper
    get_rho = get_rho
    cache_xc_kernel = cache_xc_kernel
    get_vxc = r_vxc = r_vxc
    get_fxc = r_fxc = r_fxc

    def eval_ao(self, mol, coords, deriv=0, with_s=True, shls_slice=None,
                non0tab=None, out=None, verbose=None):
        return eval_ao(mol, coords, deriv, with_s, shls_slice, non0tab, out, verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        raise NotImplementedError

    @lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, verbose)

    def block_loop(self, mol, grids, nao, deriv=0, with_s=False, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids = grids.weights.size
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index ni.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = min(int(max_memory*1e6/((comp*4+4)*nao*16*BLKSIZE))*BLKSIZE, ngrids)
            blksize = max(blksize, BLKSIZE)
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                 dtype=numpy.uint8)

        if buf is None:
            buf = numpy.empty((4,comp,blksize,nao), dtype=numpy.complex128)
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = self.eval_ao(mol, coords, deriv=deriv, with_s=with_s,
                              non0tab=non0, out=buf)
            yield ao, non0, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=1):
        dms = numpy.asarray(dms)
        nao = dms.shape[-1]
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            dms = dms.reshape(1,nao,nao)
        ndms = len(dms)
        n2c = mol.nao_2c()
        with_s = (nao == n2c*2)  # 4C DM
        if with_s:
            c1 = .5 / lib.param.LIGHT_SPEED
            dmLL = dms[:,:n2c,:n2c].copy('C')
            dmSS = dms[:,n2c:,n2c:] * c1**2

            def make_rho(idm, ao, non0tab, xctype):
                rho  = self.eval_rho(mol, ao[:2], dmLL[idm], non0tab, xctype)
                rhoS = self.eval_rho(mol, ao[2:], dmSS[idm], non0tab, xctype)
                rho[0] += rhoS[0]
                # M = |\beta\Sigma|
                rho[1:4] -= rhoS[1:4]
                return rho
        else:
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho(mol, ao, dms[idm], non0tab, xctype)
        return make_rho, ndms, nao

    def eval_xc(self, xc_code, rho, spin=1, relativity=0, deriv=1, omega=None,
                verbose=None):
        if omega is None: omega = self.omega
        if self.collinear[0] == 'c':  # collinear
            r, mx, my, mz = rho
            rhou = (r + mz) * .5
            rhod = (r - mz) * .5
            rho = (rhou, rhod)
            xc = self.libxc.eval_xc(xc_code, rho, 1, relativity, deriv,
                                    omega, verbose)
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

_RNumInt = RNumInt
