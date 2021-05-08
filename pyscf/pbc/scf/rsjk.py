#!/usr/bin/env python
# Copyright 2020-2021 The PySCF Developers. All Rights Reserved.
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Range separation JK builder

Ref:
    Q. Sun, arXiv:2012.07929
'''

import copy
import ctypes
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df import aft, aft_jk, rsdf
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.df_jk import (zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0,
                                _format_dms, _format_kpts_band, _format_jks)
from pyscf.pbc.df.rsdf import _get_cache_size, _ExtendedMoleFT
from pyscf.pbc.lib.kpts_helper import (is_zero, unique_with_wrap_around,
                                       group_by_conj_pairs)
from pyscf import __config__

# Threshold of steep bases and local bases
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 3.2)
# kecut=10 can rougly converge GTO with alpha=0.5
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)
# cutoff penalty regarding to lattice summation
LATTICE_SUM_PENALTY = 1e-2
STEEP_BASIS = ft_ao.STEEP_BASIS
LOCAL_BASIS = ft_ao.LOCAL_BASIS
SMOOTH_BASIS = ft_ao.SMOOTH_BASIS

libpbc = lib.load_library('libpbc')

class RangeSeparationJKBuilder(object):
    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = None
        self.kpts = np.reshape(kpts, (-1, 3))
        self.purify = True

        self.omega = None
        self.rs_cell = None
        # Born-von Karman supercell
        self.bvk_kmesh = None
        self.supmol_sr = None
        self.supmol_ft = None
        # For shells in bvkcell, use overlap mask to remove d-d block
        self.ovlp_mask = None
        self.ke_cutoff = None
        self.vhfopt = None
        # Use fully uncontracted basis for jk_sr part
        self.uncontract_sr = True

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'omega = %s', self.omega)
        #logger.info(self, 'len(kpts) = %d', len(self.kpts))
        #logger.debug1(self, '    kpts = %s', self.kpts)
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.rs_cell = None
        self.supmol_sr = None
        self.supmol_ft = None
        return self

    def build(self, omega=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts

        if omega is not None:
            self.omega = omega

        if self.omega is None:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            self.ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), self.ke_cutoff)

        log.info('omega = %.15g  ke_cutoff = %s  mesh = %s',
                 self.omega, self.ke_cutoff, self.mesh)

        direct_scf_tol = cell.precision**1.5 * LATTICE_SUM_PENALTY
        log.debug('Set direct_scf_tol %g', direct_scf_tol)

        rs_cell = ft_ao._RangeSeperationCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)
        self.rs_cell = rs_cell

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        # Estimate rcut to generate Ls. rcut (and the translation vectors Ls)
        # here needs to cover all possible shells to converge int3c2e.
        # cell.rcut only converge the GTOval on grids thus it cannot be used here.
        smooth_bas_mask = rs_cell.bas_type == SMOOTH_BASIS
        exps_d = [rs_cell.bas_exp(ib) for ib in range(rs_cell.nbas) if smooth_bas_mask[ib]]
        exps_c = [rs_cell.bas_exp(ib) for ib in range(rs_cell.nbas) if not smooth_bas_mask[ib]]
        if not exps_c: # Only smooth functions
            rcut_sr = cell.rcut
        else:
            if not exps_d:  # Only compact functions
                exp_d_min = exp_c_min = np.hstack(exps_c).min()
                aij = exp_d_min
                eij = exp_d_min / 2
            else:  # both smooth and compact functions exist
                exp_d_min = np.hstack(exps_d).min()
                exp_c_min = np.hstack(exps_c).min()
                aij = exp_d_min + exp_c_min
                eij = exp_d_min * exp_c_min / aij
            theta = 1/(self.omega**-2 + 2./aij)
            fac = 8*(exp_d_min*exp_c_min/aij**2)**1.5 / (theta * np.pi)**.5
            x_ratio = 1 / (exp_d_min/theta + 2*exp_c_min/aij)
            exp_fac = 2 * eij * x_ratio**2 + theta * (1 - 2*exp_c_min/aij*x_ratio)**2

            rcut_sr = cell.rcut  # initial guess
            rcut_sr = ((-np.log(direct_scf_tol
                                * (1 - 2*exp_c_min/aij*x_ratio) * rcut_sr
                                / (2*np.pi*fac)) / exp_fac)**.5
                       + pbcgto.cell._rcut_penalty(cell))
            log.debug1('exp_d_min = %g, exp_c_min = %g, rcut_sr = %g',
                       exp_d_min, exp_c_min, rcut_sr)

        self.supmol_ft = _ExtendedMoleFT.from_cell(rs_cell, kmesh, rcut_sr, log)

        if self.uncontract_sr:
            log.debug('make supmol from fully uncontracted cell basis')
            pcell, contr_coeff = rs_cell.decontract_basis(to_cart=True)
            self.contr_coeff = scipy.linalg.block_diag(*contr_coeff)
            self.supmol_sr = supmol = _ExtendedMoleSR.from_cell(
                pcell, kmesh, self.omega, rcut_sr, log)
        else:
            log.debug('make supmol from partially uncontracted cell basis')
            self.supmol_sr = supmol = _ExtendedMoleSR.from_cell(
                rs_cell, kmesh, self.omega, rcut_sr, log)

        log.timer_debug1('initializing supmol', *cpu0)
        log.info('sup-mol nbas = %d cGTO = %d pGTO = %d',
                 supmol.nbas, supmol.nao, supmol.npgto_nr())

        # Intialize vhfopt
        with supmol.with_integral_screen(direct_scf_tol**2):
            vhfopt = _vhf.VHFOpt(supmol, 'int2e_sph',
                                 qcondname=libpbc.PBCVHFsetnr_direct_scf)
        self.vhfopt = vhfopt
        vhfopt.direct_scf_tol = direct_scf_tol
        log.timer('initializing vhfopt', *cpu0)

        # Remove the smooth-smooth basis block.
        # Modify the contents of vhfopt.q_cond inplace
        q_cond = self.get_qcond()
        diffused_mask = supmol.bas_type_to_indices(SMOOTH_BASIS)
        q_cond[diffused_mask[:,None], diffused_mask] = 1e-200

        sh_loc = supmol.sh_loc
        bvk_q_cond = lib.condense('NP_absmax', q_cond, sh_loc, sh_loc)

        # In lattice sum for eri, looping over <ish| only needs the shells of
        # first primitive cell in bvk-cell. cell0_basis_mask indicates which
        # shells are located in first primitive cell.
        cell0_basis_mask = np.zeros_like(supmol.bas_mask)
        # Remove all basis except those in the first primitive cell in bvk-cell
        cell0_basis_mask[0,:,0] = True
        cell0_basis_mask = cell0_basis_mask[supmol.bas_mask]
        ovlp_mask = bvk_q_cond > direct_scf_tol
        self.ovlp_mask = np.append(ovlp_mask.ravel(), cell0_basis_mask).astype(np.int8)
        return self

    def get_qcond(self):
        supmol = self.supmol_sr
        q_cond = self.vhfopt.get_q_cond((supmol.nbas, supmol.nbas))
        return q_cond

    def _get_jk_sr(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
                   with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            # TODO: call AFTDF.get_jk function
            raise NotImplementedError

        cpu0 = logger.process_clock(), logger.perf_counter()
        if self.supmol_sr is None:
            self.build()

        comp = 1
        nkpts = kpts.shape[0]
        nbands = nkpts
        vhfopt = self.vhfopt
        supmol = self.supmol_sr
        cell = self.cell
        nao = cell.nao
        bvk_ncells = len(self.supmol_sr.bvkmesh_Ls)

        if dm_kpts.ndim != 4:
            dm = dm_kpts.reshape(-1, nkpts, nao, nao)
        else:
            dm = dm_kpts
        n_dm = dm.shape[0]

        if self.uncontract_sr:
            # parameters for decontracted basis are different
            c = self.contr_coeff
            dm = lib.einsum('nkij,pi,qj->nkpq', dm, c, c)
            decontracted_cell = supmol.rs_cell
            nbasp = decontracted_cell.nbas
            cell0_ao_loc = decontracted_cell.ao_loc
        else:
            nbasp = cell.nbas  # The number of shells in the primitive cell
            cell0_ao_loc = cell.ao_loc
        nao = dm.shape[-1]

        phase = np.exp(1j*np.dot(self.supmol_sr.bvkmesh_Ls, kpts.T)) / bvk_ncells**.5
        # Utilized symmetry sc_dm[R,S] = sc_dm[S-R] = sc_dm[(S-R)%N]
        #:sc_dm = lib.einsum('Rk,nkuv,Sk->nRuSv', phase, sc_dm, phase.conj())
        sc_dm = lib.einsum('k,Sk,nkuv->nSuv', phase[0], phase.conj(), dm)
        dm_translation = k2gamma.double_translation_indices(self.bvk_kmesh).astype(np.int32)
        dm_imag_max = abs(sc_dm.imag).max()
        is_complex_dm = dm_imag_max > 1e-6
        if is_complex_dm:
            if dm_imag_max < 1e-2:
                logger.warn(self, 'DM in (BvK) cell has small imaginary part.  '
                            'It may be a signal of symmetry broken in k-point symmetry')
            sc_dm = np.vstack([sc_dm.real, sc_dm.imag])
        else:
            sc_dm = sc_dm.real
        sc_dm = np.asarray(sc_dm.reshape(-1, bvk_ncells, nao, nao), order='C')
        n_sc_dm = sc_dm.shape[0]

        # * sparse_ao_loc has dimension (Nk,nbas), corresponding to the
        # bvkcell with all basis
        sparse_ao_loc = nao * np.arange(bvk_ncells)[:,None] + cell0_ao_loc[:-1]
        sparse_ao_loc = np.append(sparse_ao_loc.ravel(), nao * bvk_ncells)
        dm_cond = [lib.condense('NP_absmax', d, sparse_ao_loc, sparse_ao_loc[:nbasp+1])
                   for d in sc_dm]
        dm_cond = np.asarray(np.max(dm_cond, axis=0), order='C')
        libpbc.CVHFset_dm_cond(vhfopt._this,
                               dm_cond.ctypes.data_as(ctypes.c_void_p), dm_cond.size)
        dm_cond = None

        bvk_nbas = nbasp * bvk_ncells
        shls_slice = (0, nbasp, 0, bvk_nbas, 0, bvk_nbas, 0, bvk_nbas)

        cache_size = _get_cache_size(cell, 'int2e_sph')
        cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
        cache_size += cell0_dims.max()**4 * comp * 2

        if hermi:
            fdot_suffix = 's2kl'
        else:
            fdot_suffix = 's1'
        if with_j and with_k:
            fdot = 'PBCVHF_contract_jk_' + fdot_suffix
            vs = np.zeros((2, n_sc_dm, nao, bvk_ncells, nao))
        elif with_j:
            fdot = 'PBCVHF_contract_j_' + fdot_suffix
            vs = np.zeros((1, n_sc_dm, nao, bvk_ncells, nao))
        else:  # with_k
            fdot = 'PBCVHF_contract_k_' + fdot_suffix
            vs = np.zeros((1, n_sc_dm, nao, bvk_ncells, nao))

        if supmol.cart:
            intor = 'PBCint2e_cart'
        else:
            intor = 'PBCint2e_sph'

        drv = libpbc.PBCVHF_direct_drv
        drv(getattr(libpbc, fdot), getattr(libpbc, intor),
            vs.ctypes.data_as(ctypes.c_void_p),
            sc_dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(vs.size), ctypes.c_int(n_dm),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nkpts),
            ctypes.c_int(nbands), ctypes.c_int(nbasp), ctypes.c_int(comp),
            supmol.sh_loc.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*8)(*shls_slice),
            dm_translation.ctypes.data_as(ctypes.c_void_p),
            self.ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            vhfopt._cintopt, vhfopt._this, ctypes.c_int(cache_size),
            supmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            supmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            supmol._env.ctypes.data_as(ctypes.c_void_p))

        if is_complex_dm:
            vs = vs[:,:n_dm] + vs[:,n_dm:] * 1j

        if self.uncontract_sr:
            c = self.contr_coeff
            vs = lib.einsum('snpkq,pi,qj->snikj', vs, c, c)

        logger.timer_debug1(self, 'short range part vj and vk', *cpu0)
        return vs

    def get_jk(self, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

        # Does not support to specify arbitrary kpts
        if kpts is not None and abs(kpts-self.kpts).max() > 1e-7:
            raise RuntimeError('kpts error. kpts cannot be modified in RSJK')
        kpts = self.kpts

        vs = self._get_jk_sr(dm_kpts, hermi, kpts, kpts_band,
                             with_j, with_k, omega, exxdiv)
        if with_j and with_k:
            vj, vk = vs
        elif with_j:
            vj, vk = vs[0], None
        else:
            vj, vk = None, vs[0]

        if kpts_band is None:
            expRk = np.exp(1j*np.dot(self.supmol_sr.bvkmesh_Ls, kpts.T))
        else:
            logger.warn(self, 'Approximate J/K matrices at kpts_band '
                        'with the bvk-cell dervied from kpts')
            kpts_band = np.reshape(kpts_band, (-1, 3))
            expRk = np.exp(1j*np.dot(self.supmol_sr.bvkmesh_Ls, kpts_band.T))
        bvk_ncells = expRk.shape[0]
        phase = expRk / np.sqrt(bvk_ncells)

        if with_j:
            vj = lib.einsum('npRq,Rk->nkpq', vj, expRk)
            vj += self._get_lr_j_kpts(dm_kpts, hermi, kpts, kpts_band)
            if hermi:
                vj = (vj + vj.conj().transpose(0,1,3,2)) * .5
            if self.purify and kpts_band is None:
                vj = _purify(vj, phase)
            vj = _format_jks(vj, dm_kpts, kpts_band, kpts)
            if is_zero(kpts) and dm_kpts.dtype == np.double:
                vj = vj.real.copy()

        if with_k:
            vk = lib.einsum('npRq,Rk->nkpq', vk, expRk)
            vk += self._get_lr_k_kpts(dm_kpts, hermi, kpts, kpts_band, exxdiv)
            if hermi:
                vk = (vk + vk.conj().transpose(0,1,3,2)) * .5
            if self.purify and kpts_band is None:
                vk = _purify(vk, phase)
            vk = _format_jks(vk, dm_kpts, kpts_band, kpts)
            if is_zero(kpts) and dm_kpts.dtype == np.double:
                vk = vk.real.copy()

        return vj, vk

    weighted_coulG = aft.weighted_coulG
    weighted_coulG_LR = aft.weighted_coulG_LR
    weighted_coulG_SR = aft.weighted_coulG_SR

    def _get_lr_j_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        '''
        Long-range part of J matrix

        C ~ compact basis, D ~ diffused basis

        Compute J matrix with coulG_LR:
        (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)

        Compute J matrix with full coulG:
        (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
        '''
        if kpts_band is not None:
            return self._get_lr_j_for_bands(dm_kpts, hermi, kpts, kpts_band)

        if len(kpts) == 1 and not is_zero(kpts):
            raise NotImplementedError('Single k-point get-j')

        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        cell_d = rs_cell.smooth_basis_cell()
        mesh = self.mesh
        ngrids = np.prod(mesh)
        kpts = np.asarray(kpts.reshape(-1, 3), order='C')
        dms = _format_dms(dm_kpts, kpts)
        n_dm, nkpts, nao = dms.shape[:3]
        naod = cell_d.nao

        vj_kpts = np.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)

        # TODO: aosym == 's2'
        aosym = 's1'
        ft_kern = self.supmol_ft.gen_ft_kernel(
            aosym, return_complex=True, verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        weight = 1./nkpts

        kpt_allow = np.zeros(3)
        coulG_LR = self.weighted_coulG_LR(kpt_allow, False, mesh)
        if cell.dimension >= 2:
            G0_idx = 0  # due to np.fft.fftfreq convension
            G0_weight = kws[0] if isinstance(kws, np.ndarray) else kws
            coulG_LR[G0_idx] -= np.pi/self.omega**2 * G0_weight

        if naod > 0:
            coulG = self.weighted_coulG(kpt_allow, False, mesh)
            coulG_SR = coulG - coulG_LR

            aoR_ks, aoI_ks = rsdf._eval_gto(cell_d, mesh, kpts)
            smooth_bas_mask = rs_cell.bas_type == SMOOTH_BASIS
            smooth_bas_idx = rs_cell.bas_map[smooth_bas_mask]
            smooth_ao_idx = rs_cell.get_ao_indices(smooth_bas_idx, cell.ao_loc)

            # rho = einsum('nkji,kig,kjg->ng', dm, ao.conj(), ao)
            rho = np.zeros((n_dm, ngrids))
            tmpR = np.empty((naod, ngrids))
            tmpI = np.empty((naod, ngrids))
            dmR_dd = np.asarray(dms.real[:,:,smooth_ao_idx[:,None],smooth_ao_idx], order='C')
            dmI_dd = np.asarray(dms.imag[:,:,smooth_ao_idx[:,None],smooth_ao_idx], order='C')
            # vG = einsum('ij,gji->g', dm_dd[k], aoao[k]) * coulG
            for i in range(n_dm):
                for k in range(nkpts):
                    zdotNN(dmR_dd[i,k].T, dmI_dd[i,k].T, aoR_ks[k], aoI_ks[k], 1, tmpR, tmpI)
                    rho[i] += np.einsum('ig,ig->g', aoR_ks[k], tmpR)
                    rho[i] += np.einsum('ig,ig->g', aoI_ks[k], tmpI)
            vG_dd = pbctools.ifft(rho, mesh) * cell.vol * coulG
            vG_dd *= weight
            tmpR = tmpI = dmR_dd = dmI_dd = None
            cpu1 = log.timer_debug1('get_lr_j_kpts dd block', *cpu0)

            max_memory = (self.max_memory - lib.current_memory()[0]) * .9
            Gblksize = max(16, int(max_memory*1e6/16/nao**2/(nkpts+1)))
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts)
                #: aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG[p0:p1], weight)
                #: aft_jk._update_vj_(vj_kpts, aoaoks, dms, coulG_SR[p0:p1], -weight)
                for i in range(n_dm):
                    rho = np.einsum('kij,kgij->g', dms[i].conj(), Gpq).conj()
                    # NOTE: vG_dd are updated inplace. It stores the full vG then
                    vG = vG_dd[i,p0:p1]
                    vG += coulG[p0:p1] * weight * rho
                    vG_SR = coulG_SR[p0:p1] * weight * rho
                    # vG_LR contains full vG of dd-block and vG_LR of rest blocks
                    vG_LR = vG - vG_SR
                    vj_kpts[i] += np.einsum('g,kgij->kij', vG_LR, Gpq)
                Gpq = None
            log.timer_debug1('get_lr_j_kpts ft_aopair', *cpu1)

            vR = pbctools.fft(vG_dd, mesh).real * (cell.vol/ngrids)
            vjR_dd = np.empty((naod, naod))
            vjI_dd = np.empty((naod, naod))
            for i in range(n_dm):
                for k in range(nkpts):
                    tmpR = aoR_ks[k] * vR[i]
                    tmpI = aoI_ks[k] * vR[i]
                    zdotCN(aoR_ks[k], aoI_ks[k], tmpR.T, tmpI.T, 1, vjR_dd, vjI_dd)
                    lib.takebak_2d(vj_kpts[i,k], vjR_dd + vjI_dd * 1j,
                                   smooth_ao_idx, smooth_ao_idx)

        else:
            max_memory = (self.max_memory - lib.current_memory()[0]) * .9
            Gblksize = max(16, int(max_memory*1e6/16/nao**2/(nkpts+1)))
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts)
                for i in range(n_dm):
                    rho = np.einsum('kij,kgij->g', dms[i].conj(), Gpq).conj()
                    vG_LR = coulG_LR[p0:p1] * weight * rho
                    vj_kpts[i] += np.einsum('g,kgij->kij', vG_LR, Gpq)
                Gpq = None

        log.timer_debug1('get_lr_j_kpts', *cpu0)
        return vj_kpts

    def _get_lr_j_for_bands(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
        raise NotImplementedError

    def _get_lr_k_kpts(self, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
                       exxdiv=None):
        '''
        Long-range part of K matrix

        C ~ compact basis, D ~ diffused basis

        Compute K matrix with coulG_LR:
        (CC|CC) (CC|CD) (CC|DC) (CD|CC) (CD|CD) (CD|DC) (DC|CC) (DC|CD) (DC|DC)

        Compute K matrix with full coulG:
        (CC|DD) (CD|DD) (DC|DD) (DD|CC) (DD|CD) (DD|DC) (DD|DD)
        '''
        assert kpts_band is None
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        rs_cell = self.rs_cell
        cell_d = rs_cell.smooth_basis_cell()

        mesh = self.mesh
        ngrids = np.prod(mesh)
        dm_kpts = lib.asarray(dm_kpts, order='C')
        dms = _format_dms(dm_kpts, kpts)
        nset, nkpts, nao = dms.shape[:3]
        naod = cell_d.nao

        kpts_band = _format_kpts_band(kpts_band, kpts)
        nband = len(kpts_band)
        vkR = np.zeros((nset,nband,nao,nao))
        vkI = np.zeros((nset,nband,nao,nao))
        dmsR = np.asarray(dms.real, order='C')
        dmsI = np.asarray(dms.imag, order='C')
        vk = [vkR, vkI]
        dm = [dmsR, dmsI]
        weight = 1. / nkpts

        if naod > 0:
            aoR_ks, aoI_ks = rsdf._eval_gto(cell_d, mesh, kpts)
            coords = cell_d.get_uniform_grids(mesh)

            def ft_aopair_dd(ki, kj, expmikr):
                # einsum('g,ig,jg->ijg', expmikr, ao_ki.conj(), ao_kj)
                pqG_ddR = np.empty((naod**2, ngrids))
                pqG_ddI = np.empty((naod**2, ngrids))
                expmikrR, expmikrI = expmikr
                libpbc.PBC_zjoin_fCN_s1(
                    pqG_ddR.ctypes.data_as(ctypes.c_void_p),
                    pqG_ddI.ctypes.data_as(ctypes.c_void_p),
                    expmikrR.ctypes.data_as(ctypes.c_void_p),
                    expmikrI.ctypes.data_as(ctypes.c_void_p),
                    aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
                    aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
                    aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
                    aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naod), ctypes.c_int(naod), ctypes.c_int(ngrids))
                pqG_dd = pqG_ddR + pqG_ddI * 1j
                pqG_dd = pbctools.fft(pqG_dd, mesh)
                pqG_dd *= cell.vol / ngrids
                return pqG_dd.reshape(naod, naod, ngrids)

            ao_loc = cell.ao_loc
            smooth_bas_mask = rs_cell.bas_type == SMOOTH_BASIS
            smooth_bas_idx = rs_cell.bas_map[smooth_bas_mask]
            smooth_ao_idx = rs_cell.get_ao_indices(smooth_bas_idx, ao_loc)

            def merge_dd(Gpq, ki_lst, kj_lst, p0, p1, cache):
                '''Merge diffused basis block into ao-pair tensor inplace'''
                expmikr = np.exp(-1j * np.dot(coords, kpts[kj_lst[0]]-kpts[ki_lst[0]]))
                expmikrR = np.asarray(expmikr.real, order='C')
                expmikrI = np.asarray(expmikr.imag, order='C')
                GpqR, GpqI = Gpq
                # Gpq should be an array of (nkpts,ni,nj,ngrids) in C order
                if not GpqR[0].flags.c_contiguous:
                    assert GpqR[0].strides[0] == 8  # stride for grids
                for k, (ki, kj) in enumerate(zip(ki_lst, kj_lst)):
                    if cache:
                        pqG_dd = cache[(ki, kj)]
                    else:
                        pqG_dd = ft_aopair_dd(ki, kj, (expmikrR, expmikrI))
                    libpbc.PBC_ft_fuse_dd_s1(
                        GpqR[k].ctypes.data_as(ctypes.c_void_p),
                        GpqI[k].ctypes.data_as(ctypes.c_void_p),
                        pqG_dd.ctypes.data_as(ctypes.c_void_p),
                        smooth_ao_idx.ctypes.data_as(ctypes.c_void_p),
                        (ctypes.c_int*2)(p0, p1),
                        ctypes.c_int(nao), ctypes.c_int(naod), ctypes.c_int(ngrids))
                return (GpqR, GpqI)
        else:
            merge_dd = None
        cpu1 = log.timer_debug1('get_lr_k_kpts dd block', *cpu0)

        aosym = 's1'
        ft_kern = self.supmol_ft.gen_ft_kernel(
            aosym, return_complex=False, verbose=log)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

        uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
            cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
        scaled_uniq_kpts = cell_d.get_scaled_kpts(uniq_kpts).round(5)
        log.debug('Num uniq kpts %d', len(uniq_kpts))
        log.debug2('Scaled unique kpts %s', scaled_uniq_kpts)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now)) * .8
        log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)

        for group_id, (k, k_conj) in enumerate(group_by_conj_pairs(cell, uniq_kpts)[0]):
            kpt_ij_idx = np.where(uniq_inverse == k)[0]
            kpti_idx = kpt_ij_idx // nkpts
            kptj_idx = kpt_ij_idx % nkpts
            nkptj = len(kptj_idx)
            kptjs = kpts[kptj_idx]
            kpt = uniq_kpts[k]
            log.debug1('ft_ao_pair for kpt = %s', kpt)
            log.debug2('ft_ao_pair for kpti_idx = %s', kpti_idx)
            log.debug2('ft_ao_pair for kptj_idx = %s', kptj_idx)
            swap_2e = k_conj is not None

            cache_size = (naod**2*ngrids*2 * len(kpt_ij_idx))*8e-6
            if naod > 0 and max_memory*.7 > cache_size:
                expmikr = np.exp(-1j * np.dot(coords, kpt))
                expmikrR = np.asarray(expmikr.real, order='C')
                expmikrI = np.asarray(expmikr.imag, order='C')
                cache = {}
                for ki, kj in zip(kpti_idx, kptj_idx):
                    cache[(ki, kj)] = ft_aopair_dd(ki, kj, (expmikrR, expmikrI))
                Gblksize = max(16, int((max_memory-cache_size)*1e6/16/nao**2/(nkptj+5)))
            else:
                Gblksize = max(16, int(max_memory*1e6/16/nao**2/(nkptj+5)))
            Gblksize = min(Gblksize, ngrids, 200000)

            coulG_LR = self.weighted_coulG_LR(kpt, exxdiv, mesh)
            # G=0 associated to 2e integrals in real-space
            if cell.dimension >= 2 and is_zero(uniq_kpts[k]):
                G0_idx = 0
                G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
                coulG_LR[G0_idx] -= np.pi/self.omega**2 * G0_weight

            if naod > 0:
                vkcoulG = self.weighted_coulG(kpt, exxdiv, mesh)
                coulG_SR = vkcoulG - coulG_LR
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, kptjs)
                    _update_vk_(vk, Gpq, dm, coulG_SR[p0:p1],
                                -weight, kpti_idx, kptj_idx, swap_2e)
                    Gpq = merge_dd(Gpq, kpti_idx, kptj_idx, p0, p1, cache)
                    _update_vk_(vk, Gpq, dm, vkcoulG[p0:p1],
                                weight, kpti_idx, kptj_idx, swap_2e)
                    Gpq = None
            else:
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, kptjs)
                    _update_vk_(vk, Gpq, dm, coulG_LR[p0:p1],
                                weight, kpti_idx, kptj_idx, swap_2e)
                    Gpq = None
            cpu1 = log.timer_debug1('ft_aopair group %d'%group_id, *cpu1)

        if (is_zero(kpts) and is_zero(kpts_band) and
            not np.iscomplexobj(dm_kpts)):
            vk_kpts = vkR
        else:
            vk_kpts = vkR + vkI * 1j

        # Add ewald_exxdiv contribution because G=0 was not included in the
        # non-uniform grids
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            _ewald_exxdiv_for_G0(cell, kpts_band, dms, vk_kpts, kpts_band)

        log.timer_debug1('get_lr_k_kpts', *cpu0)
        return vk_kpts

def _purify(mat_kpts, phase):
    #:mat_bvk = np.einsum('Rk,nkij,Sk->nRSij', phase, mat_kpts, phase.conj())
    #:return np.einsum('Rk,nRSij,Sk->nkij', phase.conj(), mat_bvk.real, phase)
    nkpts = phase.shape[1]
    mat_bvk = lib.einsum('k,Sk,nkuv->nSuv', phase[0], phase.conj(), mat_kpts)
    return lib.einsum('S,Sk,nSuv->nkuv', nkpts*phase[:,0].conj(), phase, mat_bvk.real)

class _PrimitiveCell(ft_ao._RangeSeperationCell):
    '''Cell with partially de-contracted basis'''
    def __init__(self):
        self.ref_cell = None
        self.bas_map = None
        self.bas_type = None
        self.sh_loc = None
        self.contr_coeff = None

    @classmethod
    def from_cell(cls, cell, ke_cut_threshold=KECUT_THRESHOLD,
                  rcut_threshold=None, precision=None, verbose=None):
        log = logger.new_logger(cell, verbose)
        if precision is None:
            precision = cell.precision

        pcell, contr_coeff = cell.decontract_basis(to_cart=True)
        rs_cell = pcell.view(cls)
        rs_cell.ref_cell = pcell

        rs_cell.contr_coeff = scipy.linalg.block_diag(*contr_coeff)
        exps = rs_cell._env[rs_cell._bas[:,gto.PTR_EXP]]
        l = rs_cell._bas[:,gto.ANG_OF]
        ke_cut = pbcgto.cell._estimate_ke_cutoff(exps, l, 1, precision)

        rs_cell.bas_type = np.empty(rs_cell.nbas, dtype=np.int32)
        rs_cell.bas_type[ke_cut < ke_cut_threshold] = SMOOTH_BASIS
        rs_cell.bas_type[ke_cut >= ke_cut_threshold] = LOCAL_BASIS
        # For each basis of rs_cell, bas_map gives the basis in cell
        rs_cell.bas_map = np.arange(rs_cell.nbas, dtype=np.int32)
        rs_cell.sh_loc = np.append(np.arange(rs_cell.nbas), rs_cell.nbas).astype(np.int32)
        rs_cell.ke_cutoff = ke_cut_threshold
        rs_cell.precision = precision
        return rs_cell

class _ExtendedMoleSR(rsdf._ExtendedMoleSR):
    '''Extended Mole for short-range ERIs without dd-blocks'''
    @classmethod
    def from_cell(cls, cell, kmesh, omega, rcut=None, verbose=None):
        assert isinstance(cell, ft_ao._RangeSeperationCell)
        if rcut is None: rcut = cell.rcut

        bvkcell = pbctools.super_cell(cell, kmesh)
        Ls = bvkcell.get_lattice_Ls(rcut=rcut)
        Ls = Ls[np.linalg.norm(Ls, axis=1) < rcut]
        Ls = Ls[np.linalg.norm(Ls, axis=1).argsort()]
        bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
        LKs = Ls[:,None,:] + bvkmesh_Ls
        nimgs, bvk_ncells = LKs.shape[:2]

        supmol = cls()
        supmol.__dict__.update(cell.to_mol().__dict__)
        supmol = pbctools._build_supcell_(supmol, cell, LKs.reshape(nimgs*bvk_ncells, 3))
        supmol.rs_cell = cell
        supmol.bvk_kmesh = kmesh
        supmol.bvkmesh_Ls = bvkmesh_Ls
        supmol.omega = -omega
        supmol.precision = cell.precision
        rs_cell = cell

        bas_mask = np.ones((bvk_ncells, rs_cell.nbas, nimgs), dtype=bool)
        compact_bas_mask = rs_cell.bas_type != SMOOTH_BASIS
        exps = np.array([rs_cell.bas_exp(ib).min() for ib in range(rs_cell.nbas)])
        exps_c = exps[compact_bas_mask]
        if exps_c.size > 0:
            exp_min = exps.min()
            exp_c_min = exps_c.min()
            aij = exp_min + exps_c
            eij = exp_min * exps_c / aij
            akl = exp_c_min + exp_min
            ekl = exp_c_min * exp_min / akl
            theta = 1./(omega**-2 + 1./aij + 1./akl)
            rLK = np.linalg.norm(LKs, axis=2) - pbcgto.cell._rcut_penalty(rs_cell)
            rLK[rLK < 1e-2] = 1e-2  # avoid singularity in upper_bounds

            x_ratio = 1. / (exp_min**2/(exps_c*aij) + exp_c_min/akl + exp_min/theta)
            y_ratio = 1. / (exp_c_min*exps_c/(exp_min*akl) + exp_min/aij + exps_c/theta)
            exp_fac = (ekl * x_ratio**2 + eij * y_ratio**2 +
                       theta * (1 - exp_c_min/akl*x_ratio - exp_min/aij*y_ratio)**2)

            fac = 8*(exp_min**2*exp_c_min*exps_c/(aij*akl)**2)**.75 / (theta*np.pi)**.5
            upper_bounds = np.einsum('i,lk,ilk->ilk', fac, 2*np.pi/rLK,
                                     np.exp(-exp_fac[:,None,None]*rLK**2))
            cutoff = rs_cell.precision * LATTICE_SUM_PENALTY
            bas_mask[:,compact_bas_mask] = upper_bounds.transpose(2,0,1) > cutoff

            # determine rcut boundary for diffused functions
            exps_d = exps[~compact_bas_mask]
            if exps_d.size > 0:
                aij = exp_c_min + exps_d
                eij = exp_c_min * exps_d / aij
                akl = exp_c_min + exp_min
                ekl = exp_c_min * exp_min / akl
                theta = 1./(omega**-2 + 1./aij + 1./akl)
                x_ratio = 1. / (exp_min*exp_c_min/(exps_d *aij) + exp_c_min/akl + exp_min/theta)
                y_ratio = 1. / (exps_d *exp_c_min/(exp_min*akl) + exp_c_min/aij + exps_d /theta)
                exp_fac = (ekl * x_ratio**2 + eij * y_ratio**2 +
                           theta * (1 - exp_c_min/akl*x_ratio - exp_c_min/aij*y_ratio)**2)
                fac = 8*(exp_min*exp_c_min**2*exps_d/(aij*akl)**2)**.75 / (theta*np.pi)**.5
                upper_bounds = np.einsum('i,lk,ilk->ilk', fac, 2*np.pi/rLK,
                                         np.exp(-exp_fac[:,None,None]*rLK**2))
                bas_mask[:,~compact_bas_mask] = upper_bounds.transpose(2,0,1) > cutoff

        bas_mask[0,:,0] = True
        _bas_reordered = supmol._bas.reshape(
            nimgs, bvk_ncells, rs_cell.nbas, gto.BAS_SLOTS).transpose(1,2,0,3)
        supmol._bas = np.asarray(_bas_reordered[bas_mask], dtype=np.int32, order='C')
        supmol.sh_loc = supmol.bas_mask_to_sh_loc(rs_cell, bas_mask, verbose)
        supmol.bas_mask = bas_mask
        return supmol


def _guess_omega(cell, kpts, mesh=None):
    nao = cell.npgto_nr()
    nkpts = len(kpts)
    nkk = nkpts**(1./3) * 2 - 1
    if mesh is None:
        #mesh = [max(5, int(cell.rcut * nao ** (1./3) / nkk + 1))] * 3
        mesh = [max(5, int((cell.rcut**3/cell.vol * nao**2) ** (1./3) / nkk + .5))] * 3
        mesh = np.min([cell.mesh, mesh], axis=0)
    ke_cutoff = min(pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh[:cell.dimension]))
    omega = aft.estimate_omega_for_ke_cutoff(cell, ke_cutoff)
    return omega, mesh, ke_cutoff

def _update_vk_(vk, Gpq, dms, coulG, weight, kpti_idx, kptj_idx, swap_2e):
    vkR, vkI = vk
    GpqR, GpqI = Gpq
    dmsR, dmsI = dms
    nG = len(coulG)
    n_dm = vkR.shape[0]
    nao = vkR.shape[-1]
    bufR = np.empty((nG*nao**2))
    bufI = np.empty((nG*nao**2))
    buf1R = np.empty((nG*nao**2))
    buf1I = np.empty((nG*nao**2))

    for k, (ki, kj) in enumerate(zip(kpti_idx, kptj_idx)):
        # case 1: k_pq = (pi|iq)
        #:v4 = np.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,jk->il', v4, dm)
        pLqR = np.ndarray((nao,nG,nao), buffer=bufR)
        pLqI = np.ndarray((nao,nG,nao), buffer=bufI)
        pLqR[:] = GpqR[k].transpose(1,0,2)
        pLqI[:] = GpqI[k].transpose(1,0,2)
        iLkR = np.ndarray((nao,nG,nao), buffer=buf1R)
        iLkI = np.ndarray((nao,nG,nao), buffer=buf1I)
        for i in range(n_dm):
            zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                   dmsR[i,kj], dmsI[i,kj], 1,
                   iLkR.reshape(-1,nao), iLkI.reshape(-1,nao))
            iLkR *= coulG.reshape(1,nG,1)
            iLkI *= coulG.reshape(1,nG,1)
            zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                   pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                   weight, vkR[i,ki], vkI[i,ki], 1)

        # case 2: k_pq = (iq|pi)
        #:v4 = np.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
        #:vk += np.einsum('ijkl,li->kj', v4, dm)
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        if swap_2e:
            for i in range(n_dm):
                zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1,
                       iLkR.reshape(nao,-1), iLkI.reshape(nao,-1))
                iLkR *= coulG.reshape(1,nG,1)
                iLkI *= coulG.reshape(1,nG,1)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                       weight, vkR[i,kj], vkI[i,kj], 1)
