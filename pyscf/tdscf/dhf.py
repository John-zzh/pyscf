#!/usr/bin/env python
# Copyright 2021-2022 The PySCF Developers. All Rights Reserved.
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
TDA and TDHF for no-pair DKS Hamiltonian
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf.data import nist
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_uhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_uhf_TDDFT_pick_eig_threshold', 1e-4)

# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_uhf_TDDFT_positive_eig_threshold', 1e-3)

def gen_tda_operation(mf, fock_ao=None):
    '''A x
    '''
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    # Remove all negative states
    n2c = nmo // 2
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if fock_ao is None:
        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]
    hdiag = (fvv.diagonal() - foo.diagonal()[:,None]).ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(hermi=0)

    def vind(zs):
        zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
        dmov = lib.einsum('xov,qv,po->xpq', zs, orbv.conj(), orbo)
        v1ao = vresp(dmov)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1ov += lib.einsum('xqs,sp->xqp', zs, fvv)
        v1ov -= lib.einsum('xpr,sp->xsr', zs, foo)
        return v1ov.reshape(v1ov.shape[0], -1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    assert(mo_coeff.dtype == numpy.double)

    mol = mf.mol
    nao, nmo = mo_coeff.shape
    n2c = nmo // 2
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    nmo = nocc + nvir
    mo = numpy.hstack((orbo, orbv))
    moL = numpy.asarray(mo[:n2c], order='F')
    moS = numpy.asarray(mo[n2c:], order='F')
    orboL = moL[:,:nocc]
    orboS = moS[:,:nocc]

    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        eri_mo = ao2mo.kernel(mol, [orboL, moL, moL, moL], intor='int2e_spinor')
        eri_mo+= ao2mo.kernel(mol, [orboS, moS, moS, moS], intor='int2e_spsp1_spsp2_spinor')
        eri_mo+= ao2mo.kernel(mol, [orboS, moS, moL, moL], intor='int2e_spsp1_spinor')
        eri_mo+= ao2mo.kernel(mol, [moS, moS, orboL, moL], intor='int2e_spsp1_spinor').T
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)

        a = a + numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc])
        a = a - numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb
        b = b + numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:])
        b = b - numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb
        return a, b

    if isinstance(mf, dft.KohnShamDFT):
        from pyscf.dft import xc_deriv, mcfun
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            raise NotImplementedError

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        a, b = add_hf_(a, b, hyb)

        if ni.collinear[0] == 'm':  # mcol
            fn_eval_xc = ni.mcfun_eval_xc_wrapper(mf.xc)
            nproc = lib.num_threads()

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        def get_mo_value(ao):
            aoLa, aoLb, aoSa, aoSb = ao
            c1 = .5 / lib.param.LIGHT_SPEED
            if aoLa.ndim == 2:
                moLa = lib.einsum('rp,pi->ri', aoLa, moL)
                moLb = lib.einsum('rp,pi->ri', aoLb, moL)
                moSa = lib.einsum('rp,pi->ri', aoSa, moS * c1)
                moSb = lib.einsum('rp,pi->ri', aoSb, moS * c1)
                return (moLa[:,:nocc], moLa[:,nocc:], moLb[:,:nocc], moLb[:,nocc:],
                        moSa[:,:nocc], moSa[:,nocc:], moSb[:,:nocc], moSb[:,nocc:])
            else:
                moLa = lib.einsum('xrp,pi->xri', aoLa, moL)
                moLb = lib.einsum('xrp,pi->xri', aoLb, moL)
                moSa = lib.einsum('xrp,pi->xri', aoSa, moS * c1)
                moSb = lib.einsum('xrp,pi->xri', aoSb, moS * c1)
                return (moLa[:,:,:nocc], moLa[:,:,nocc:], moLb[:,:,:nocc], moLb[:,:,nocc:],
                        moSa[:,:,:nocc], moSa[:,:,nocc:], moSb[:,:,:nocc], moSb[:,:,nocc:])

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = mcfun.eval_xc_eff(fn_eval_xc, rho, deriv=2,
                                            spin_samples=ni.spin_samples,
                                            workers=nproc)[2]
                    wr, wmx, wmy, wmz = weight * fxc.reshape(4,4,-1)
                    moLoa, moLva, moLob, moLvb, moSoa, moSva, moSob, moSvb = get_mo_value(ao)
                    rho_ov_aa = numpy.einsum('ri,ra->ria', moLoa.conj(), moLva)
                    rho_ov_ab = numpy.einsum('ri,ra->ria', moLoa.conj(), moLvb)
                    rho_ov_ba = numpy.einsum('ri,ra->ria', moLob.conj(), moLva)
                    rho_ov_bb = numpy.einsum('ri,ra->ria', moLob.conj(), moLvb)
                    rho_ov_aa += numpy.einsum('ri,ra->ria', moSoa.conj(), moSva)
                    rho_ov_ab += numpy.einsum('ri,ra->ria', moSoa.conj(), moSvb)
                    rho_ov_ba += numpy.einsum('ri,ra->ria', moSob.conj(), moSva)
                    rho_ov_bb += numpy.einsum('ri,ra->ria', moSob.conj(), moSvb)
                    rho_vo_aa = rho_ov_aa.conj()
                    rho_vo_ab = rho_ov_ba.conj()
                    rho_vo_ba = rho_ov_ab.conj()
                    rho_vo_bb = rho_ov_bb.conj()
                    w_ov  = (wmx+wmy*1j)[:,:,None,None] * rho_ov_ba
                    w_ov += (wmx-wmy*1j)[:,:,None,None] * rho_ov_ab
                    w_ov += (wr+wmz)[:,:,None,None] * rho_ov_aa
                    w_ov += (wr-wmz)[:,:,None,None] * rho_ov_bb
                    wr_ov, wmx_ov, wmy_ov, wmz_ov = w_ov
                    a += lib.einsum('ria,rjb->iajb', wmx_ov+wmy_ov*1j, rho_vo_ba)
                    a += lib.einsum('ria,rjb->iajb', wmx_ov-wmy_ov*1j, rho_vo_ab)
                    a += lib.einsum('ria,rjb->iajb', wr_ov+wmz_ov    , rho_vo_aa)
                    a += lib.einsum('ria,rjb->iajb', wr_ov-wmz_ov    , rho_vo_bb)
                    b += lib.einsum('ria,rjb->iajb', wmx_ov+wmy_ov*1j, rho_ov_ba)
                    b += lib.einsum('ria,rjb->iajb', wmx_ov-wmy_ov*1j, rho_ov_ab)
                    b += lib.einsum('ria,rjb->iajb', wr_ov+wmz_ov    , rho_ov_aa)
                    b += lib.einsum('ria,rjb->iajb', wr_ov-wmz_ov    , rho_ov_bb)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1)
                    vxc, fxc = ni.eval_xc(mf.xc, rho, 1, deriv=2)[1:3]
                    fxc = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
                    wv_a, wv_b = weight * fxc.reshape(2,2,-1)
                    moLoa, moLva, moLob, moLvb, moSoa, moSva, moSob, moSvb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('ri,ra->ria', moLoa.conj(), moLva)
                    rho_ov_b = numpy.einsum('ri,ra->ria', moLob.conj(), moLvb)
                    rho_ov_a += numpy.einsum('ri,ra->ria', moSoa.conj(), moSva)
                    rho_ov_b += numpy.einsum('ri,ra->ria', moSob.conj(), moSvb)
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_ov  = wv_a[:,:,None,None] * rho_ov_a
                    w_ov += wv_b[:,:,None,None] * rho_ov_b
                    wa_ov, wb_ov = w_ov
                    a += lib.einsum('ria,rjb->iajb', wa_ov, rho_vo_a)
                    a += lib.einsum('ria,rjb->iajb', wb_ov, rho_vo_b)
                    b += lib.einsum('ria,rjb->iajb', wa_ov, rho_ov_a)
                    b += lib.einsum('ria,rjb->iajb', wb_ov, rho_ov_b)
                else:
                    raise NotImplementedError(ni.collinear)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = mcfun.eval_xc_eff(fn_eval_xc, rho, deriv=2,
                                            spin_samples=ni.spin_samples,
                                            workers=nproc)[2]
                    wr, wmx, wmy, wmz = weight * fxc
                    moLoa, moLva, moLob, moLvb, moSoa, moSva, moSob, moSvb = get_mo_value(ao)
                    rho_ov_aa = numpy.einsum('ri,xra->xria', moLoa[0].conj(), moLva)
                    rho_ov_ab = numpy.einsum('ri,xra->xria', moLoa[0].conj(), moLvb)
                    rho_ov_ba = numpy.einsum('ri,xra->xria', moLob[0].conj(), moLva)
                    rho_ov_bb = numpy.einsum('ri,xra->xria', moLob[0].conj(), moLvb)
                    rho_ov_aa[1:4] += numpy.einsum('xri,ra->xria', moLoa[1:4].conj(), moLva[0])
                    rho_ov_ab[1:4] += numpy.einsum('xri,ra->xria', moLoa[1:4].conj(), moLvb[0])
                    rho_ov_ba[1:4] += numpy.einsum('xri,ra->xria', moLob[1:4].conj(), moLva[0])
                    rho_ov_bb[1:4] += numpy.einsum('xri,ra->xria', moLob[1:4].conj(), moLvb[0])
                    rho_ov_aa += numpy.einsum('ri,xra->xria', moSoa[0].conj(), moSva)
                    rho_ov_ab += numpy.einsum('ri,xra->xria', moSoa[0].conj(), moSvb)
                    rho_ov_ba += numpy.einsum('ri,xra->xria', moSob[0].conj(), moSva)
                    rho_ov_bb += numpy.einsum('ri,xra->xria', moSob[0].conj(), moSvb)
                    rho_ov_aa[1:4] += numpy.einsum('xri,ra->xria', moSoa[1:4].conj(), moSva[0])
                    rho_ov_ab[1:4] += numpy.einsum('xri,ra->xria', moSoa[1:4].conj(), moSvb[0])
                    rho_ov_ba[1:4] += numpy.einsum('xri,ra->xria', moSob[1:4].conj(), moSva[0])
                    rho_ov_bb[1:4] += numpy.einsum('xri,ra->xria', moSob[1:4].conj(), moSvb[0])
                    rho_vo_aa = rho_ov_aa.conj()
                    rho_vo_ab = rho_ov_ba.conj()
                    rho_vo_ba = rho_ov_ab.conj()
                    rho_vo_bb = rho_ov_bb.conj()
                    w_ov  = numpy.einsum('xsyr,xria->syria', wmx+wmy*1j, rho_ov_ba)
                    w_ov += numpy.einsum('xsyr,xria->syria', wmx-wmy*1j, rho_ov_ab)
                    w_ov += numpy.einsum('xsyr,xria->syria', wr+wmz, rho_ov_aa)
                    w_ov += numpy.einsum('xsyr,xria->syria', wr-wmz, rho_ov_bb)
                    wr_ov, wmx_ov, wmy_ov, wmz_ov = w_ov
                    a += lib.einsum('xria,xrjb->iajb', wmx_ov+wmy_ov*1j, rho_vo_ba)
                    a += lib.einsum('xria,xrjb->iajb', wmx_ov-wmy_ov*1j, rho_vo_ab)
                    a += lib.einsum('xria,xrjb->iajb', wr_ov+wmz_ov    , rho_vo_aa)
                    a += lib.einsum('xria,xrjb->iajb', wr_ov-wmz_ov    , rho_vo_bb)
                    b += lib.einsum('xria,xrjb->iajb', wmx_ov+wmy_ov*1j, rho_ov_ba)
                    b += lib.einsum('xria,xrjb->iajb', wmx_ov-wmy_ov*1j, rho_ov_ab)
                    b += lib.einsum('xria,xrjb->iajb', wr_ov+wmz_ov    , rho_ov_aa)
                    b += lib.einsum('xria,xrjb->iajb', wr_ov-wmz_ov    , rho_ov_bb)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1)
                    vxc, fxc = ni.eval_xc(mf.xc, rho, 1, deriv=2)[1:3]
                    fxc = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
                    wv_a, wv_b = weight * fxc
                    moLoa, moLva, moLob, moLvb, moSoa, moSva, moSob, moSvb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('xri,ra->xria', moLoa.conj(), moLva[0])
                    rho_ov_b = numpy.einsum('xri,ra->xria', moLob.conj(), moLvb[0])
                    rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', moLoa[0].conj(), moLva[1:4])
                    rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', moLob[0].conj(), moLvb[1:4])
                    rho_ov_a += numpy.einsum('xri,ra->xria', moSoa.conj(), moSva[0])
                    rho_ov_b += numpy.einsum('xri,ra->xria', moSob.conj(), moSvb[0])
                    rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', moSoa[0].conj(), moSva[1:4])
                    rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', moSob[0].conj(), moSvb[1:4])
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_ov  = numpy.einsum('sxyr,xria->syria', wv_a, rho_ov_a)
                    w_ov += numpy.einsum('sxyr,xria->syria', wv_b, rho_ov_b)
                    wa_ov, wb_ov = w_ov
                    a += lib.einsum('xria,xrjb->iajb', wa_ov, rho_vo_a)
                    a += lib.einsum('xria,xrjb->iajb', wb_ov, rho_vo_b)
                    b += lib.einsum('xria,xrjb->iajb', wa_ov, rho_ov_a)
                    b += lib.einsum('xria,xrjb->iajb', wb_ov, rho_ov_b)
                else:
                    raise NotImplementedError(ni.collinear)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                if ni.collinear[0] == 'm':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1, with_lapl=False)
                    fxc = mcfun.eval_xc_eff(fn_eval_xc, rho, deriv=2,
                                            spin_samples=ni.spin_samples,
                                            workers=nproc)[2]
                    wr, wmx, wmy, wmz = weight * fxc
                    moLoa, moLva, moLob, moLvb, moSoa, moSva, moSob, moSvb = get_mo_value(ao)
                    rho_ov_aa = numpy.einsum('ri,xra->xria', moLoa[0].conj(), moLva)
                    rho_ov_ab = numpy.einsum('ri,xra->xria', moLoa[0].conj(), moLvb)
                    rho_ov_ba = numpy.einsum('ri,xra->xria', moLob[0].conj(), moLva)
                    rho_ov_bb = numpy.einsum('ri,xra->xria', moLob[0].conj(), moLvb)
                    rho_ov_aa += numpy.einsum('ri,xra->xria', moSoa[0].conj(), moSva)
                    rho_ov_ab += numpy.einsum('ri,xra->xria', moSoa[0].conj(), moSvb)
                    rho_ov_ba += numpy.einsum('ri,xra->xria', moSob[0].conj(), moSva)
                    rho_ov_bb += numpy.einsum('ri,xra->xria', moSob[0].conj(), moSvb)
                    rho_ov_aa[1:4] += numpy.einsum('xri,ra->xria', moLoa[1:4].conj(), moLva[0])
                    rho_ov_ab[1:4] += numpy.einsum('xri,ra->xria', moLoa[1:4].conj(), moLvb[0])
                    rho_ov_ba[1:4] += numpy.einsum('xri,ra->xria', moLob[1:4].conj(), moLva[0])
                    rho_ov_bb[1:4] += numpy.einsum('xri,ra->xria', moLob[1:4].conj(), moLvb[0])
                    rho_ov_aa[1:4] += numpy.einsum('xri,ra->xria', moSoa[1:4].conj(), moSva[0])
                    rho_ov_ab[1:4] += numpy.einsum('xri,ra->xria', moSoa[1:4].conj(), moSvb[0])
                    rho_ov_ba[1:4] += numpy.einsum('xri,ra->xria', moSob[1:4].conj(), moSva[0])
                    rho_ov_bb[1:4] += numpy.einsum('xri,ra->xria', moSob[1:4].conj(), moSvb[0])
                    tau_ov_aa = numpy.einsum('xri,xra->ria', moLoa[1:4].conj(), moLva[1:4]) * .5
                    tau_ov_ab = numpy.einsum('xri,xra->ria', moLoa[1:4].conj(), moLvb[1:4]) * .5
                    tau_ov_ba = numpy.einsum('xri,xra->ria', moLob[1:4].conj(), moLva[1:4]) * .5
                    tau_ov_bb = numpy.einsum('xri,xra->ria', moLob[1:4].conj(), moLvb[1:4]) * .5
                    tau_ov_aa += numpy.einsum('xri,xra->ria', moSoa[1:4].conj(), moSva[1:4]) * .5
                    tau_ov_ab += numpy.einsum('xri,xra->ria', moSoa[1:4].conj(), moSvb[1:4]) * .5
                    tau_ov_ba += numpy.einsum('xri,xra->ria', moSob[1:4].conj(), moSva[1:4]) * .5
                    tau_ov_bb += numpy.einsum('xri,xra->ria', moSob[1:4].conj(), moSvb[1:4]) * .5
                    rho_ov_aa = numpy.vstack([rho_ov_aa, tau_ov_aa[numpy.newaxis]])
                    rho_ov_ab = numpy.vstack([rho_ov_ab, tau_ov_ab[numpy.newaxis]])
                    rho_ov_ba = numpy.vstack([rho_ov_ba, tau_ov_ba[numpy.newaxis]])
                    rho_ov_bb = numpy.vstack([rho_ov_bb, tau_ov_bb[numpy.newaxis]])
                    rho_vo_aa = rho_ov_aa.conj()
                    rho_vo_ab = rho_ov_ba.conj()
                    rho_vo_ba = rho_ov_ab.conj()
                    rho_vo_bb = rho_ov_bb.conj()
                    w_ov  = numpy.einsum('xsyr,xria->syria', wmx+wmy*1j, rho_ov_ba)
                    w_ov += numpy.einsum('xsyr,xria->syria', wmx-wmy*1j, rho_ov_ab)
                    w_ov += numpy.einsum('xsyr,xria->syria', wr+wmz, rho_ov_aa)
                    w_ov += numpy.einsum('xsyr,xria->syria', wr-wmz, rho_ov_bb)
                    wr_ov, wmx_ov, wmy_ov, wmz_ov = w_ov
                    a += lib.einsum('xria,xrjb->iajb', wmx_ov+wmy_ov*1j, rho_vo_ba)
                    a += lib.einsum('xria,xrjb->iajb', wmx_ov-wmy_ov*1j, rho_vo_ab)
                    a += lib.einsum('xria,xrjb->iajb', wr_ov+wmz_ov    , rho_vo_aa)
                    a += lib.einsum('xria,xrjb->iajb', wr_ov-wmz_ov    , rho_vo_bb)
                    b += lib.einsum('xria,xrjb->iajb', wmx_ov+wmy_ov*1j, rho_ov_ba)
                    b += lib.einsum('xria,xrjb->iajb', wmx_ov-wmy_ov*1j, rho_ov_ab)
                    b += lib.einsum('xria,xrjb->iajb', wr_ov+wmz_ov    , rho_ov_aa)
                    b += lib.einsum('xria,xrjb->iajb', wr_ov-wmz_ov    , rho_ov_bb)
                elif ni.collinear[0] == 'c':
                    rho = ni.eval_rho(mol, ao, dm0, mask, xctype, hermi=1)
                    vxc, fxc = ni.eval_xc(mf.xc, rho, 1, deriv=2)[1:3]
                    fxc = xc_deriv.transform_fxc(rho, vxc, fxc, xctype, spin=1)
                    wv_a, wv_b = weight * fxc
                    moLoa, moLva, moLob, moLvb, moSoa, moSva, moSob, moSvb = get_mo_value(ao)
                    rho_ov_a = numpy.einsum('xri,ra->xria', moLoa.conj(), moLva[0])
                    rho_ov_b = numpy.einsum('xri,ra->xria', moLob.conj(), moLvb[0])
                    rho_ov_a += numpy.einsum('xri,ra->xria', moSoa.conj(), moSva[0])
                    rho_ov_b += numpy.einsum('xri,ra->xria', moSob.conj(), moSvb[0])
                    rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', moLoa[0].conj(), moLva[1:4])
                    rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', moLob[0].conj(), moLvb[1:4])
                    rho_ov_a[1:4] += numpy.einsum('ri,xra->xria', moSoa[0].conj(), moSva[1:4])
                    rho_ov_b[1:4] += numpy.einsum('ri,xra->xria', moSob[0].conj(), moSvb[1:4])
                    tau_ov_a = numpy.einsum('xri,xra->ria', moLoa[1:4].conj(), moLva[1:4]) * .5
                    tau_ov_b = numpy.einsum('xri,xra->ria', moLob[1:4].conj(), moLvb[1:4]) * .5
                    tau_ov_a += numpy.einsum('xri,xra->ria', moSoa[1:4].conj(), moSva[1:4]) * .5
                    tau_ov_b += numpy.einsum('xri,xra->ria', moSob[1:4].conj(), moSvb[1:4]) * .5
                    rho_ov_a = numpy.vstack([rho_ov_a, tau_ov_a[numpy.newaxis]])
                    rho_ov_b = numpy.vstack([rho_ov_b, tau_ov_b[numpy.newaxis]])
                    rho_vo_a = rho_ov_a.conj()
                    rho_vo_b = rho_ov_b.conj()
                    w_ov  = numpy.einsum('xsyr,xria->syria', wv_a, rho_ov_a)
                    w_ov += numpy.einsum('xsyr,xria->syria', wv_b, rho_ov_b)
                    wa_ov, wb_ov = w_ov
                    a += lib.einsum('xria,xrjb->iajb', wa_ov, rho_vo_a)
                    a += lib.einsum('xria,xrjb->iajb', wb_ov, rho_vo_b)
                    b += lib.einsum('xria,xrjb->iajb', wa_ov, rho_ov_a)
                    b += lib.einsum('xria,xrjb->iajb', wb_ov, rho_ov_b)
                else:
                    raise NotImplementedError(ni.collinear)

    else:
        a, b = add_hf_(a, b)

    return a, b

def get_nto(tdobj, state=1, threshold=OUTPUT_THRESHOLD, verbose=None):
    raise NotImplementedError('get_nto')

def analyze(tdobj, verbose=None):
    raise NotImplementedError('analyze')

def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    raise NotImplementedError


class TDMixin(rhf.TDMixin):

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    analyze = analyze
    get_nto = get_nto
    _contract_multipole = _contract_multipole  # needed by transition dipoles

    def nuc_grad_method(self):
        raise NotImplementedError


@lib.with_doc(rhf.TDA.__doc__)
class TDA(TDMixin):
    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_tda_hop(mf)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        # Remove all negative states
        n2c = mf.mo_occ.size // 2
        occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
        viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia_max = e_ia.max()

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_ia = e_ia.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > POSTIVE_EIG_THRESHOLD**2)[0]
            return w[idx], v[:,idx], idx

        # FIXME: Is it correct to call davidson1 for complex integrals
        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        n2c = nmo // 2
        nvir = n2c - nocc
        self.xy = [(xi.reshape(nocc,nvir), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        self._finalize()
        return self.e, self.xy


def gen_tdhf_operation(mf, fock_ao=None):
    '''Generate function to compute

    [ A   B ][X]
    [-B* -A*][Y]
    '''
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    n2c = nmo // 2
    occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
    viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    foo = numpy.diag(mo_energy[occidx])
    fvv = numpy.diag(mo_energy[viridx])
    hdiag = fvv.diagonal() - foo.diagonal()[:,None]
    hdiag = numpy.hstack((hdiag.ravel(), -hdiag.ravel())).real

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = mf.gen_response(hermi=0)

    def vind(xys):
        xys = numpy.asarray(xys).reshape(-1,2,nocc,nvir)
        xs, ys = xys.transpose(1,0,2,3)
        # dms = AX + BY
        dms  = lib.einsum('xov,qv,po->xpq', xs, orbv.conj(), orbo)
        dms += lib.einsum('xov,pv,qo->xpq', ys, orbv, orbo.conj())

        v1ao = vresp(dms)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv.conj())
        v1ov += lib.einsum('xqs,sp->xqp', xs, fvv)  # AX
        v1ov -= lib.einsum('xpr,sp->xsr', xs, foo)  # AX
        v1vo += lib.einsum('xqs,sp->xqp', ys, fvv)  # AY
        v1vo -= lib.einsum('xpr,sp->xsr', ys, foo)  # AY

        # (AX, (-A*)Y)
        nz = xys.shape[0]
        hx = numpy.hstack((v1ov.reshape(nz,-1), -v1vo.reshape(nz,-1)))
        return hx

    return vind, hdiag


class TDHF(TDMixin):
    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        return gen_tdhf_operation(mf)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        x0 = TDA.init_guess(self, mf, nstates, wfnsym)
        y0 = numpy.zeros_like(x0)
        return numpy.hstack((x0, y0))

    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > POSTIVE_EIG_THRESHOLD))[0]
            # FIXME: Should the amplitudes be real?
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                      real_eigenvectors=False)

        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_cycle=self.max_cycle,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        n2c = nmo // 2
        nvir = n2c - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(1./norm)
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

from pyscf import scf
scf.dhf.DHF.TDA = lib.class_as_method(TDA)
scf.dhf.DHF.TDHF = lib.class_as_method(TDHF)

del(OUTPUT_THRESHOLD)
