#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
Non-relativistic RKS analytical Hessian
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint


# import pyscf.grad.rks to activate nuc_grad_method method
from pyscf.grad import rks  # noqa


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, beta = mf._numint.rsh_coeff(mf.xc)
    if abs(omega) > 1e-10:
        hyb = alpha + beta
    else:
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=mol.spin)
    de2, ej, ek = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                             atmlst, max_memory, verbose,
                                             abs(hyb) > 1e-10)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    if abs(omega) > 1e-10:
        with mol.with_range_coulomb(omega):
            vk1 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                                   ['jk->s1il', dm0])[0]
        veff_diag -= (alpha-hyb)*.5 * vk1.reshape(3,3,nao,nao)
    vk1 = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    vxc = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        veff = vxc[ia]
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk1, vk2 = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                            ['li->s1kj', dm0[:,p0:p1],  # vk1
                                             'lj->s1ki', dm0         ], # vk2
                                            shls_slice=shls_slice)
            veff -= (alpha-hyb)*.5 * vk1.reshape(3,3,nao,nao)
            veff[:,:,:,p0:p1] -= (alpha-hyb)*.5 * vk2.reshape(3,3,nao,p1-p0)
            t1 = log.timer_debug1('range-separated int2e_ip1ip2 for atom %d'%ia, *t1)
            with mol.with_range_coulomb(omega):
                vk1 = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                                       ['li->s1kj', dm0[:,p0:p1]], # vk1
                                       shls_slice=shls_slice)[0]
            veff -= (alpha-hyb)*.5 * vk1.transpose(0,2,1).reshape(3,3,nao,nao)
            t1 = log.timer_debug1('range-separated int2e_ipvip1 for atom %d'%ia, *t1)
            vk1 = vk2 = None

        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', veff[:,:,q0:q1], dm0[q0:q1])*2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if abs(hyb) > 1e-10:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if abs(omega) > 1e-10:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice)
                veff -= (alpha-hyb) * .5 * vk1
                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice)
            veff = vj1
            veff[:,p0:p1] += vj2

        h1ao[ia] += veff + veff.transpose(0,2,1)
        h1ao[ia] += hcore_deriv(ia)

    if chkfile is None:
        return h1ao
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
        return chkfile

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            aow = numint._scale_ao(ao[0], weight*vrho)
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
            aow = None

    elif xctype == 'GGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            # *2 because v.T is not applied. Only v is computed in the next _dot_ao_ao
            wv[0] *= 2
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            contract_(vmat[0], ao, [XXX,XXY,XXZ], wv, mask)
            contract_(vmat[1], ao, [XXY,XYY,XYZ], wv, mask)
            contract_(vmat[2], ao, [XXZ,XYZ,XZZ], wv, mask)
            contract_(vmat[3], ao, [XYY,YYY,YYZ], wv, mask)
            contract_(vmat[4], ao, [XYZ,YYZ,YZZ], wv, mask)
            contract_(vmat[5], ao, [XZZ,YZZ,ZZZ], wv, mask)
            rho = vxc = wv = aow = None

    elif xctype == 'MGGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            # *2 because v.T is not applied. Only v is computed in the next _dot_ao_ao
            wv[0] *= 2
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            contract_(vmat[0], ao, [XXX,XXY,XXZ], wv, mask)
            contract_(vmat[1], ao, [XXY,XYY,XYZ], wv, mask)
            contract_(vmat[2], ao, [XXZ,XYZ,XZZ], wv, mask)
            contract_(vmat[3], ao, [XYY,YYY,YYZ], wv, mask)
            contract_(vmat[4], ao, [XYZ,YYZ,YZZ], wv, mask)
            contract_(vmat[5], ao, [XZZ,YZZ,ZZZ], wv, mask)

            vtau = vxc[3]
            wv = .5 * weight * vtau
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[0], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[1], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[2], mask, shls_slice, ao_loc)

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices, xctype='GGA'):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[0]
    if xctype == 'GGA':
        rho1 = numpy.zeros((3,4,ngrids))
    elif xctype == 'MGGA':
        rho1 = numpy.zeros((3,6,ngrids))
        ao_dm0_x = ao_dm0[1][:,p0:p1]
        ao_dm0_y = ao_dm0[2][:,p0:p1]
        ao_dm0_z = ao_dm0[3][:,p0:p1]
        # (d_X \nabla mu) dot \nalba nu DM_{mu,nu}
        rho1[0,5] += numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_x)
        rho1[0,5] += numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_y)
        rho1[0,5] += numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_z)
        rho1[1,5] += numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_x)
        rho1[1,5] += numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_y)
        rho1[1,5] += numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_z)
        rho1[2,5] += numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_x)
        rho1[2,5] += numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_y)
        rho1[2,5] += numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_z)
        rho1[:,5] *= .5
    else:
        raise RuntimeError

    ao_dm0_0 = ao_dm0[0][:,p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[:,0] = numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0_0)
    rho1[0,1]+= numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_0)
    rho1[0,2]+= numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_0)
    rho1[0,3]+= numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_0)
    rho1[1,1]+= numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_0)
    rho1[1,2]+= numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_0)
    rho1[1,3]+= numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_0)
    rho1[2,1]+= numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_0)
    rho1[2,2]+= numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_0)
    rho1[2,3]+= numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_0)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    rho1[:,1] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[1][:,p0:p1])
    rho1[:,2] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[2][:,p0:p1])
    rho1[:,3] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[3][:,p0:p1])

    # *2 for |mu> DM <d_X nu|
    return rho1 * 2

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d1], ao2[d2], mask,
                                                 shls_slice, ao_loc)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d2], ao2[d1], mask,
                                                 shls_slice, ao_loc)

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    vmat = numpy.zeros((mol.natm,3,3,nao,nao))
    ipip = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho = vxc[0]
            frr = fxc[0]
            wv = weight * vrho
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                # aow ~ rho1 ~ d/dR1
                wv = weight * frr * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices)
                for i in range(3):
                    wv[i] = numint._rks_gga_wv1(rho, dR_rho1[i], vxc, fxc, weight)
                    aow = rks_grad._make_dR_dao_w(ao, wv[i])
                    rks_grad._d1_dot_(vmat[ia,i], mol, aow, ao[0], mask, ao_loc, True)

                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    elif xctype == 'MGGA':
        XX, XY, XZ = 4, 5, 6
        YX, YY, YZ = 5, 7, 8
        ZX, ZY, ZZ = 6, 8, 9
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_mgga_wv0(rho, vxc, weight)
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            aow = [numint._scale_ao(ao[i], wv[5]) for i in range(4, 10)]
            _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                for i in range(3):
                    wv[i] = numint._rks_mgga_wv1(rho, dR_rho1[i], vxc, fxc, weight)
                    aow = rks_grad._make_dR_dao_w(ao, wv[i])
                    rks_grad._d1_dot_(vmat[ia,i], mol, aow, ao[0], mask, ao_loc, True)

                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)

                # *2 because wv[5] is scaled by 0.5 in _rks_mgga_wv1
                wv = wv[:,5] * 2
                aow = [numint._scale_ao(ao[1], wv[i]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[XX], ao[XY], ao[XZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[2], wv[i]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[YX], ao[YY], ao[YZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[3], wv[i]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[ZX], ao[ZY], ao[ZZ]], aow, mask, ao_loc, False)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    return vmat

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    v_ip = numpy.zeros((3,nao,nao))
    vmat = numpy.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-vmat.size*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho = vxc[0]
            frr = fxc[0]
            aow = numint._scale_ao(ao[0], weight*vrho)
            rks_grad._d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                wv = weight * frr * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices)
                wv[0] = numint._rks_gga_wv1(rho, dR_rho1[0], vxc, fxc, weight)
                wv[1] = numint._rks_gga_wv1(rho, dR_rho1[1], vxc, fxc, weight)
                wv[2] = numint._rks_gga_wv1(rho, dR_rho1[2], vxc, fxc, weight)
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = None

    elif xctype == 'MGGA':
        if grids.level < 5:
            logger.warn(mol, 'MGGA Hessian is sensitive to dft grids.')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]

            wv = numint._rks_gga_wv0(rho, vxc, weight)
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

            wv = .5 * weight * vxc[3]
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            rks_grad._d1_dot_(v_ip, mol, [ao[XX], ao[XY], ao[XZ]], aow[0], mask, ao_loc, True)
            rks_grad._d1_dot_(v_ip, mol, [ao[YX], ao[YY], ao[YZ]], aow[1], mask, ao_loc, True)
            rks_grad._d1_dot_(v_ip, mol, [ao[ZX], ao[ZY], ao[ZZ]], aow[2], mask, ao_loc, True)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv[0] = numint._rks_mgga_wv1(rho, dR_rho1[0], vxc, fxc, weight)
                wv[1] = numint._rks_mgga_wv1(rho, dR_rho1[1], vxc, fxc, weight)
                wv[2] = numint._rks_mgga_wv1(rho, dR_rho1[2], vxc, fxc, weight)
                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                rks_grad._d1_dot_(vmat[ia], mol, aow, ao[0], mask, ao_loc, True)

                for j in range(1, 4):
                    aow = [numint._scale_ao(ao[j], wv[i,5]) for i in range(3)]
                    rks_grad._d1_dot_(vmat[ia], mol, aow, ao[j], mask, ao_loc, True)
            ao_dm0 = aow = None

    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        vmat[ia,:,p0:p1] += v_ip[:,p0:p1]
        vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    return vmat


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic RKS hessian'''
    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grids'])

    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1

from pyscf import dft
dft.rks.RKS.Hessian = dft.rks_symm.RKS.Hessian = lib.class_as_method(Hessian)
