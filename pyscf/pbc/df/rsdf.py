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
Build GDF tensor with range-separation technique
'''

__all__ = ['make_j3c']

import os
import ctypes
import tempfile
import numpy as np
import scipy.linalg
import h5py
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.gto import NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, ATOM_OF, ANG_OF
from pyscf.pbc.df import aft, aft_jk
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.incore import make_auxcell
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC, _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import (is_zero, unique_with_wrap_around,
                                       group_by_conj_pairs)
from pyscf import __config__

# Threshold of steep bases and local bases
RCUT_THRESHOLD = 3.2
# cutoff penalty regarding to lattice summation
LATTICE_SUM_PENALTY = 1e-2
STEEP_BASIS = ft_ao.STEEP_BASIS
LOCAL_BASIS = ft_ao.LOCAL_BASIS
SMOOTH_BASIS = ft_ao.SMOOTH_BASIS

libpbc = lib.load_library('libpbc')

# TODO: test no diffused functions

def _gen_int3c_kernel(rsdf, intor='int3c2e', aosym='s2', comp=None,
                      kpts=np.zeros((1,3)), j_only=False):
    '''Generate function to compute int3c2e with double lattice-sum'''
    supmol = rsdf.supmol_sr
    cell = rsdf.cell
    auxcell = rsdf.auxcell
    rs_auxcell = rsdf.rs_auxcell
    nkpts = len(kpts)
    bvk_ncells = len(rsdf.supmol_sr.bvkmesh_Ls)
    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
    nbasp = cell.nbas  # The number of shells in the primitive cell

    # integral mask for supmol
    cutoff = rsdf.precision * LATTICE_SUM_PENALTY
    q_cond = rsdf.get_qcond()
    ovlp_mask = q_cond[:supmol.nbas] > cutoff
    bvk_ovlp_mask = lib.condense('np.any', ovlp_mask, supmol.sh_loc)
    cell0_ovlp_mask = bvk_ovlp_mask.reshape(
        bvk_ncells, nbasp, bvk_ncells, nbasp).any(axis=2).any(axis=0)
    q_cond_aux = q_cond[supmol.nbas:].copy()

######################## TODO
    # TODO: create auxcell_c = rs_auxcell.compact_basis_cell() and only
    # computing integrals of (cell,cell|auxcell_c). In _make_j3c extracting
    # the compact-j3c to the entire tensor
    if rsdf.compact_sr_j3c:
        # Use aux_mask to skip smooth auxiliary basis and handle them in AFT part.
        aux_mask = rs_auxcell.bas_type != SMOOTH_BASIS
    else:
        aux_mask = np.ones(rs_auxcell.nbas, dtype=bool)
    ovlp_mask = np.append(ovlp_mask.ravel(), aux_mask).astype(np.int8)

    atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                 rs_auxcell._atm, rs_auxcell._bas, rs_auxcell._env)
    cell0_ao_loc = _conc_locs(cell.ao_loc, auxcell.ao_loc)
    sh_loc = _conc_locs(supmol.sh_loc, rs_auxcell.sh_loc)

    # Estimate the buffer size required by PBCfill_nr3c functions
    cache_size = max(_get_cache_size(cell, intor),
                     _get_cache_size(auxcell, intor))
    cell0_dims = cell0_ao_loc[1:] - cell0_ao_loc[:-1]
    dijk = cell0_dims[:nbasp].max()**2 * cell0_dims[nbasp:].max() * comp

    expLk = np.exp(1j*np.dot(rsdf.supmol_sr.bvkmesh_Ls, kpts.T))
    gamma_point_only = is_zero(kpts)
    if gamma_point_only:
        fill = f'PBCfill_nr3c_g{aosym}'
        expLkR = expLk.real
        expLkI = expLk.imag
        nkpts_ij = 1
        cache_size += dijk
    elif kpts.shape[0] == 1:
        fill = f'PBCfill_nr3c_nk1{aosym}'
        expLkR = np.asarray(expLk.real, order='C')
        expLkI = np.asarray(expLk.imag, order='C')
        nkpts_ij = 1
        cache_size += dijk * 3
    elif j_only:
        fill = f'PBCfill_nr3c_k{aosym}'
        eexpLk = np.einsum('Lk,Mk->LMk', expLk.conj(), expLk)
        expLkR = np.asarray(eexpLk.real, order='C')
        expLkI = np.asarray(eexpLk.imag, order='C')
        nkpts_ij = nkpts
        cache_size = dijk * bvk_ncells**2 + max(dijk * nkpts * 2, cache_size)
    else:
        fill = f'PBCfill_nr3c_kk{aosym}'
        expLkR = np.asarray(expLk.real, order='C')
        expLkI = np.asarray(expLk.imag, order='C')
        nkpts_ij = nkpts * nkpts
        cache_size = (dijk * max(bvk_ncells**2, nkpts**2 * 2) +
                      max(dijk * bvk_ncells * nkpts * 2, cache_size))
    expLk = None

    drv = libpbc.PBCfill_nr3c_drv
    cintopt = rsdf.cintopt

    def int3c(shls_slice=None, outR=None, outI=None):
        if shls_slice is None:
            shls_slice = [0, nbasp, 0, nbasp, nbasp, nbasp + auxcell.nbas]
        else:
            ksh0 = nbasp + shls_slice[4]
            ksh1 = nbasp + shls_slice[5]
            shls_slice = list(shls_slice[:4]) + [ksh0, ksh1]
        i0, i1, j0, j1, k0, k1 = cell0_ao_loc[shls_slice]
        if aosym == 's1':
            shape = (nkpts_ij, comp, (i1-i0)*(j1-j0), k1-k0)
        else:
            nrow = i1*(i1+1)//2 - i0*(i0+1)//2
            shape = (nkpts_ij, comp, nrow, k1-k0)
        outR = np.ndarray(shape, buffer=outR)
        outR[:] = 0
        if gamma_point_only:
            outI = np.zeros(0)
        else:
            outI = np.ndarray(shape, buffer=outI)
            outI[:] = 0
        drv(getattr(libpbc, intor), getattr(libpbc, fill),
            outR.ctypes.data_as(ctypes.c_void_p),
            outI.ctypes.data_as(ctypes.c_void_p),
            expLkR.ctypes.data_as(ctypes.c_void_p),
            expLkI.ctypes.data_as(ctypes.c_void_p),
            sh_loc.ctypes.data_as(ctypes.c_void_p),
            cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice),
            ctypes.c_int(bvk_ncells), ctypes.c_int(nkpts),
            ctypes.c_int(nbasp), ctypes.c_int(comp),
            ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            cell0_ovlp_mask.ctypes.data_as(ctypes.c_void_p),
            q_cond_aux.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(cutoff), cintopt, ctypes.c_int(cache_size),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
            env.ctypes.data_as(ctypes.c_void_p))
        return outR, outI
    return int3c

def _outcore_smooth_block(rsdf, h5group, intor='int3c2e', aosym='s2', comp=None,
                          kpts=np.zeros((1,3)), j_only=False,
                          dataname='j3c', shls_slice=None):
    '''
    The block of smooth AO basis in i and j of (ij|L) with full Coulomb kernel
    '''
    if intor not in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart'):
        raise NotImplementedError

    if shls_slice is not None:
        raise NotImplementedError

    log = logger.new_logger(rsdf)
    cell = rsdf.cell
    cell_d = rsdf.rs_cell.smooth_basis_cell()
    auxcell = rsdf.auxcell
    nao = cell_d.nao
    naux = auxcell.nao
    nkpts = kpts.shape[0]
    if nao == 0 or naux == 0:
        log.debug2('Not found diffused basis. Skip outcore_smooth_block')
        return

    aoR_ks, aoI_ks = _eval_gto(cell_d, cell_d.mesh, kpts)
    coords = cell_d.get_uniform_grids(cell_d.mesh)

    # TODO check if max_memory is enough
    Gv, Gvbase, kws = auxcell.get_Gv_weights(cell_d.mesh)
    b = cell_d.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = Gv.shape[0]

    def get_Vaux(kpt):
        # int3c2e = fft(ao.conj()*ao*exp(-1j*coords.dot(kpt))) * coulG *
        #           (cell.vol/ngrids) * fft(aux*exp(-1j*coords.dot(-kpt)))
        #         = fft(ao.conj()*ao*exp(-1j*coords.dot(kpt))) * coulG *
        #           ft_ao(aux, -kpt)
        #         = ao.conj()*ao*exp(-1j*coords.dot(kpt)) *
        #           ifft(coulG * ft_ao(aux, -kpt))
        #         = ao.conj()*ao*Vaux
        # where
        # Vaux = ao*exp(-1j*coords.dot(kpt)) * ifft(coulG * ft_ao(aux, -kpt))
        auxG = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, -kpt).T
        auxG *= pbctools.get_coulG(cell, -kpt, False, None, cell_d.mesh, Gv)
        Vaux = pbctools.ifft(auxG, cell_d.mesh)
        Vaux *= np.exp(-1j * coords.dot(kpt))
        return Vaux

    def join_R(ki, kj):
        #:aopair = np.einsum('ig,jg->ijg', aoR_ks[ki], aoR_ks[kj])
        #:aopair+= np.einsum('ig,jg->ijg', aoI_ks[ki], aoI_ks[kj])
        aopair = np.empty((nao**2, ngrids))
        libpbc.PBC_zjoinR_CN_s1(
            aopair.ctypes.data_as(ctypes.c_void_p),
            aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
            aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
            aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
            aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
        return aopair

    def join_I(ki, kj):
        #:aopair = np.einsum('ig,jg->ijg', aoR_ks[ki], aoI_ks[kj])
        #:aopair-= np.einsum('ig,jg->ijg', aoI_ks[ki], aoR_ks[kj])
        aopair = np.empty((nao**2, ngrids))
        libpbc.PBC_zjoinI_CN_s1(
            aopair.ctypes.data_as(ctypes.c_void_p),
            aoR_ks[ki].ctypes.data_as(ctypes.c_void_p),
            aoI_ks[ki].ctypes.data_as(ctypes.c_void_p),
            aoR_ks[kj].ctypes.data_as(ctypes.c_void_p),
            aoI_ks[kj].ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
        return aopair

    gamma_point_only = is_zero(kpts)
    if j_only or gamma_point_only:
        Vaux = np.asarray(get_Vaux(np.zeros(3)).real, order='C')
        if gamma_point_only:
            #:aopair = np.einsum('ig,jg->ijg', aoR_ks[0], aoR_ks[0])
            aopair = np.empty((nao**2, ngrids))
            libpbc.PBC_djoin_NN_s1(
                aopair.ctypes.data_as(ctypes.c_void_p),
                aoR_ks[0].ctypes.data_as(ctypes.c_void_p),
                aoR_ks[0].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao), ctypes.c_int(nao), ctypes.c_int(ngrids))
            j3c = lib.ddot(aopair.reshape(nao**2, ngrids), Vaux.T)
            h5group[f'{dataname}R-dd/0'] = j3c.reshape(nao, nao, naux)
            aopair = j3c = None

        else:
            #:for k in range(nkpts):
            #:    h5group[f'{dataname}R-dd/{k*nkpts+k}'] = lib.ddot(join_R(k, k), Vaux.T)
            #:    h5group[f'{dataname}I-dd/{k*nkpts+k}'] = lib.ddot(join_I(k, k), Vaux.T)
            k_idx = np.arange(nkpts, dtype=np.int32)
            kpt_ij_idx = k_idx * nkpts + k_idx
            j3cR = np.empty((nkpts, nao, nao, naux))
            j3cI = np.empty((nkpts, nao, nao, naux))
            libpbc.PBC_kzdot_CNN_s1(j3cR.ctypes.data_as(ctypes.c_void_p),
                                    j3cI.ctypes.data_as(ctypes.c_void_p),
                                    aoR_ks.ctypes.data_as(ctypes.c_void_p),
                                    aoI_ks.ctypes.data_as(ctypes.c_void_p),
                                    Vaux.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
                                    kpt_ij_idx.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(nao), ctypes.c_int(nao),
                                    ctypes.c_int(naux), ctypes.c_int(ngrids),
                                    ctypes.c_int(nkpts), ctypes.c_int(nkpts))
            for k, kk_idx in enumerate(kpt_ij_idx):
                h5group[f'{dataname}R-dd/{kk_idx}'] = j3cR[k]
                h5group[f'{dataname}I-dd/{kk_idx}'] = j3cI[k]

    else:
        uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
            cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
        scaled_uniq_kpts = cell_d.get_scaled_kpts(uniq_kpts).round(5)
        log.debug('Num uniq kpts %d', len(uniq_kpts))
        log.debug2('Scaled unique kpts %s', scaled_uniq_kpts)
        for k, k_conj in group_by_conj_pairs(cell, uniq_kpts)[0]:
            # Find ki's and kj's that satisfy k_aux = kj - ki
            kpt_ij_idx = np.asarray(np.where(uniq_inverse == k)[0], dtype=np.int32)
            nkptij = len(kpt_ij_idx)

            Vaux = get_Vaux(uniq_kpts[k])
            VauxR = np.asarray(Vaux.real, order='C')
            VauxI = np.asarray(Vaux.imag, order='C')
            Vaux = None
            #:for kk_idx in kpt_ij_idx:
            #:    ki = kk_idx // nkpts
            #:    kj = kk_idx % nkpts
            #:    aopair = join_R(ki, kj, exp(-i*k dot r))
            #:    j3cR = lib.ddot(aopair.reshape(nao**2, ngrids), VauxR.T)
            #:    j3cI = lib.ddot(aopair.reshape(nao**2, ngrids), VauxI.T)
            #:    aopair = join_I(ki, kj, exp(-i*k dot r))
            #:    j3cR = lib.ddot(aopair.reshape(nao**2, ngrids), VauxI.T,-1, j3cR, 1)
            #:    j3cI = lib.ddot(aopair.reshape(nao**2, ngrids), VauxR.T, 1, j3cI, 1)
            j3cR = np.empty((nkptij, nao, nao, naux))
            j3cI = np.empty((nkptij, nao, nao, naux))
            libpbc.PBC_kzdot_CNN_s1(j3cR.ctypes.data_as(ctypes.c_void_p),
                                    j3cI.ctypes.data_as(ctypes.c_void_p),
                                    aoR_ks.ctypes.data_as(ctypes.c_void_p),
                                    aoI_ks.ctypes.data_as(ctypes.c_void_p),
                                    VauxR.ctypes.data_as(ctypes.c_void_p),
                                    VauxI.ctypes.data_as(ctypes.c_void_p),
                                    kpt_ij_idx.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(nao), ctypes.c_int(nao),
                                    ctypes.c_int(naux), ctypes.c_int(ngrids),
                                    ctypes.c_int(nkptij), ctypes.c_int(nkpts))
            for k, kk_idx in enumerate(kpt_ij_idx):
                h5group[f'{dataname}R-dd/{kk_idx}'] = j3cR[k]
                h5group[f'{dataname}I-dd/{kk_idx}'] = j3cI[k]
            j3cR = j3cI = VauxR = VauxI = None

def _outcore_auxe2(rsdf, h5group, intor='int3c2e', aosym='s2', comp=None,
                   kpts=np.zeros((1,3)), j_only=False,
                   dataname='j3c', shls_slice=None):
    r'''The SR part of 3-center integrals (ij|L) with double lattice sum.

    Kwargs:
        shls_slice :
            Indicate the shell slices in the primitive cell
    '''
    log = logger.new_logger(rsdf)
    cell = rsdf.cell
    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
    rsdf.outcore_smooth_block(h5group, intor, aosym, comp,
                              kpts, j_only, dataname, shls_slice)
    int3c = rsdf.gen_int3c_kernel(intor, aosym, comp, kpts, j_only)

    rs_cell = rsdf.rs_cell
    auxcell = rsdf.auxcell
    naux = auxcell.nao
    kpts = np.asarray(kpts.reshape(-1, 3), order='C')
    nkpts = kpts.shape[0]

    gamma_point_only = is_zero(kpts)
    if gamma_point_only:
        j_only = True

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    i0, i1, j0, j1 = [ao_loc[i] for i in shls_slice[:4]]
    k0, k1 = aux_loc[shls_slice[4]],  aux_loc[shls_slice[5]]
    if aosym == 's1':
        nao_pair = (i1 - i0) * (j1 - j0)
    else:
        nao_pair = i1*(i1+1)//2 - i0*(i0+1)//2
    naux = k1 - k0

    feri = h5group
    if dataname in feri:
        del(feri[dataname])

    if np.any(rs_cell.bas_type == SMOOTH_BASIS):
        merge_dd = rs_cell.merge_diffused_block(aosym)
    else:
        merge_dd = None

    if j_only:
        for k in range(nkpts):
            shape = (comp, nao_pair, naux)
            feri.create_dataset(f'{dataname}R/{k*nkpts+k}', shape, 'f8')
            # exclude imaginary part for gamma point
            if not is_zero(kpts[k]):
                feri.create_dataset(f'{dataname}I/{k*nkpts+k}', shape, 'f8')
        nkpts_ij = nkpts
        kikj_idx = [k*nkpts+k for k in range(nkpts)]
    else:
        for ki in range(nkpts):
            for kj in range(nkpts):
                shape = (comp, nao_pair, naux)
                feri.create_dataset(f'{dataname}R/{ki*nkpts+kj}', shape, 'f8')
                feri.create_dataset(f'{dataname}I/{ki*nkpts+kj}', shape, 'f8')
            # exclude imaginary part for gamma point
            if is_zero(kpts[ki]):
                del feri[f'{dataname}I/{ki*nkpts+ki}']
        nkpts_ij = nkpts * nkpts
        kikj_idx = range(nkpts_ij)
        if merge_dd:
            uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
                cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
            kpt_ij_pairs = group_by_conj_pairs(cell, uniq_kpts)[0]

    if naux == 0:
        return

    mem_now = lib.current_memory()[0]
    log.debug2('memory = %s', mem_now)
    max_memory = max(2000, rsdf.max_memory-mem_now)

    # split the 3-center tensor (nkpts_ij, i, j, aux) along shell i.
    # plus 1 to ensure the intermediates in libpbc do not overflow
    buflen = min(max(int(max_memory*.9e6/16/naux/(nkpts_ij+1)), 1), nao_pair)
    # lower triangle part
    sh_ranges = _guess_shell_ranges(cell, buflen, aosym,
                                    start=shls_slice[0], stop=shls_slice[1])
    max_buflen = max([x[2] for x in sh_ranges])
    if max_buflen > buflen:
        log.warn('memory usage of rsdf.outcore may be '
                 f'{(max_buflen/buflen - 1):.2%} over max_memory')

    bufR = np.empty((nkpts_ij, comp, max_buflen, naux))
    bufI = np.empty_like(bufR)
    cpu0 = logger.process_clock(), logger.perf_counter()
    nsteps = len(sh_ranges)
    row1 = 0
    for istep, (sh_start, sh_end, nrow) in enumerate(sh_ranges):
        shape = (nkpts_ij, comp, nrow, naux)
        outR, outI = int3c(shls_slice, bufR, bufI)
        log.debug2('      step [%d/%d], shell range [%d:%d], len(buf) = %d',
                   istep+1, nsteps, sh_start, sh_end, nrow)
        cpu0 = log.timer_debug1(f'outcore_auxe2 [{istep+1}/{nsteps}]', *cpu0)

        shls_slice = (sh_start, sh_end, 0, cell.nbas)
        row0, row1 = row1, row1 + nrow
        if merge_dd is not None:
            if gamma_point_only:
                merge_dd(outR[0], feri[f'{dataname}R-dd/0'], shls_slice)
            elif j_only:
                for k in range(nkpts):
                    merge_dd(outR[k], feri[f'{dataname}R-dd/{k*nkpts+k}'], shls_slice)
                    merge_dd(outI[k], feri[f'{dataname}I-dd/{k*nkpts+k}'], shls_slice)
            else:
                for k, k_conj in kpt_ij_pairs:
                    kpt_ij_idx = np.where(uniq_inverse == k)[0]
                    if k_conj is None:
                        for ij_idx in kpt_ij_idx:
                            merge_dd(outR[ij_idx], feri[f'{dataname}R-dd/{ij_idx}'], shls_slice)
                            merge_dd(outI[ij_idx], feri[f'{dataname}I-dd/{ij_idx}'], shls_slice)
                    else:
                        ki_lst = kpt_ij_idx // nkpts
                        kj_lst = kpt_ij_idx % nkpts
                        kpt_ji_idx = kj_lst * nkpts + ki_lst
                        for ij_idx, ji_idx in zip(kpt_ij_idx, kpt_ji_idx):
                            j3cR_dd = np.asarray(feri[f'{dataname}R-dd/{ij_idx}'])
                            merge_dd(outR[ij_idx], j3cR_dd, shls_slice)
                            merge_dd(outR[ji_idx], j3cR_dd.transpose(1,0,2), shls_slice)
                            j3cI_dd = np.asarray(feri[f'{dataname}I-dd/{ij_idx}'])
                            merge_dd(outI[ij_idx], j3cI_dd, shls_slice)
                            merge_dd(outI[ji_idx],-j3cI_dd.transpose(1,0,2), shls_slice)

        for k, kk_idx in enumerate(kikj_idx):
            feri[f'{dataname}R/{kk_idx}'][:,row0:row1] = outR[k]
            if f'{dataname}I/{kk_idx}' in feri:
                feri[f'{dataname}I/{kk_idx}'][:,row0:row1] = outI[k]
        outR = outI = None
    bufR = bufI = None

def make_j3c(cell, auxcell_or_auxbasis, cderi_file,
             intor='int3c2e', aosym='s2', comp=None,
             kpts=np.zeros((1,3)), j_only=False, shls_slice=None):
    assert comp is None or comp == 1

    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)
    if isinstance(auxcell_or_auxbasis, gto.Mole):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)
    rsdf = _RangeSeparationDFBuilder(cell, auxcell, kpts).build()
    return rsdf.make_j3c(cderi_file, intor, aosym, comp, j_only, shls_slice)

def _make_j3c(rsdf, cderi_file, intor='int3c2e', aosym='s2', comp=None,
              j_only=False, shls_slice=None):
    log = logger.new_logger(rsdf)
    cpu0 = logger.process_clock(), logger.perf_counter()

    cell = rsdf.cell
    rs_cell = rsdf.rs_cell
    auxcell = rsdf.auxcell
    rs_auxcell = rsdf.rs_auxcell
    kpts = rsdf.kpts
    nkpts = len(kpts)
    nao = cell.nao
    naux = auxcell.nao
    if aosym == 's2':
        nao_pair = nao*(nao+1)//2
    else:
        nao_pair = nao**2

    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    dataname = 'j3c'
    rsdf.outcore_auxe2(fswap, intor, aosym=aosym, comp=comp, kpts=kpts,
                       j_only=j_only, dataname=dataname, shls_slice=shls_slice)
    cpu1 = logger.timer(rsdf, 'pass1: short range part of int3c2e', *cpu0)

    recontract_1d = rs_auxcell.recontract()
    ft_kern = rsdf.supmol_ft.gen_ft_kernel(
        aosym, return_complex=False, verbose=log)

    Gv, Gvbase, kws = rs_cell.get_Gv_weights(rsdf.mesh)
    b = rs_cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = Gv.shape[0]

    # exp(-i*(G + k) dot r) * Coulomb_kernel
    def get_weighted_Gaux(kpt):
        coulG_LR = rsdf.weighted_coulG_LR(kpt, False, rsdf.mesh)
        # Remove the G=0 contribution, from 3c2e SR-integrals in real-space
        if cell.dimension >= 2 and is_zero(kpt):
            G0_idx = 0  # due to np.fft.fftfreq convension
            G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
            coulG_LR[G0_idx] -= np.pi/rsdf.omega**2 * G0_weight

        if rsdf.compact_sr_j3c:
            # The smooth basis in auxcell was excluded in outcore_auxe2.
            # Full Coulomb kernel needs to be applied for the smooth basis
            smooth_aux_mask = rs_auxcell.get_ao_type() == SMOOTH_BASIS
            auxG = ft_ao.ft_ao(rs_auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
            auxG[smooth_aux_mask] *= rsdf.weighted_coulG(kpt, False, rsdf.mesh)
            auxG[~smooth_aux_mask] *= coulG_LR
            auxG = recontract_1d(auxG)
        else:
            auxG = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt).T
            auxG *= coulG_LR
        Gaux = lib.transpose(auxG)
        GauxR = np.asarray(Gaux.real, order='C')
        GauxI = np.asarray(Gaux.imag, order='C')
        return GauxR, GauxI

    feri = h5py.File(cderi_file, 'w')
    feri[f'{dataname}-kpts'] = kpts
    def make_kpt(kpt, kpt_ij_idx, cholesky_j2c):
        log.debug1('make_kpt for %s', kpt)
        log.debug1('kpt_ij_idx = %s', kpt_ij_idx)
        kj_lst = kpt_ij_idx % nkpts
        nkptj = len(kj_lst)
        kptjs = kpts[kj_lst]

        j2c, j2c_negative, j2ctag = cholesky_j2c
        GauxR, GauxI = get_weighted_Gaux(kpt)

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(1000, rsdf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.3e6/16/naux/(nkptj+1)), 1), nao_pair)
        sh_ranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in sh_ranges])
        # * 2 for the buffer used in preload
        max_memory -= buflen * naux * (nkptj+1) * 16e-6 * 2

        # +1 for a pqkbuf
        Gblksize = max(16, int(max_memory*1e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngrids, 200000)

        def load(col0, col1):
            j3cR = []
            j3cI = []
            for kk in kpt_ij_idx:
                j3cR.append(fswap[f'{dataname}R/{kk}'][col0:col1].reshape(-1, naux))
                if f'{dataname}I/{kk}' in fswap:
                    j3cI.append(fswap[f'{dataname}I/{kk}'][col0:col1].reshape(-1, naux))
                else:
                    j3cI.append(None)
            return j3cR, j3cI

        #buf = np.empty(nkptj*buflen*Gblksize, dtype=np.complex128)
        cols = [sh_range[2] for sh_range in sh_ranges]
        locs = np.append(0, np.cumsum(cols))
        # ij in (ij|k) is composite index in h5 swap file
        if aosym == 's2':
            locs = locs * (locs + 1) // 2
        else:
            locs = locs * nao
        for istep, dat in enumerate(lib.map_with_prefetch(load, locs[:-1], locs[1:])):
            j3cR, j3cI = dat
            bstart, bend, ncol = sh_ranges[istep]
            log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                       istep+1, len(sh_ranges), bstart, bend, ncol)
            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)

            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                dat = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, kptjs, shls_slice)
                # shape of dat (nkpts, nGv, ni, nj)
                datR, datI = dat
                nG = p1 - p0
                for k in range(nkptj):
                    pqkR = datR[k].reshape(nG, -1)
                    pqkI = datI[k].reshape(nG, -1)
                    # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                    # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
                    # functions |P> are assumed to be real
                    lib.ddot(pqkR.T, GauxR[p0:p1], 1, j3cR[k], 1)
                    lib.ddot(pqkI.T, GauxI[p0:p1], 1, j3cR[k], 1)
                    if j3cI[k] is not None:
                        lib.ddot(pqkI.T, GauxR[p0:p1],  1, j3cI[k], 1)
                        lib.ddot(pqkR.T, GauxI[p0:p1], -1, j3cI[k], 1)

                dat = datR = datI = pqkR = pqkI = None

            for k, kk_idx in enumerate(kpt_ij_idx):
                if j3cI[k] is None:
                    j3c = j3cR[k]
                else:
                    j3c = j3cR[k] + j3cI[k] * 1j
                j3c = j3c.T

                if j2ctag == 'CD':
                    feri[f'{dataname}/{kk_idx}/{istep}'] = scipy.linalg.solve_triangular(
                        j2c, j3c, lower=True, overwrite_b=True)
                else:
                    feri[f'{dataname}/{kk_idx}/{istep}'] = lib.dot(j2c, j3c)
                    # low-dimension systems
                    if j2c_negative is not None:
                        feri[f'{dataname}-/{kk_idx}/{istep}'] = lib.dot(j2c_negative, j3c)
                j3c = None

    if j_only:
        uniq_kpts = np.zeros((1,3))
        j2c = rsdf.get_2c2e(uniq_kpts)[0]
        cpu1 = logger.timer(rsdf, 'short range part of int2c2e', *cpu1)
        cholesky_j2c = rsdf.cholesky_decomposed_metric(j2c)
        j2c = None
        if cholesky_j2c[2] == 'eig':
            log.warn('DF metric linear dependency at gamma point')
        ki = np.arange(nkpts)
        make_kpt(uniq_kpts[0], ki * nkpts + ki, cholesky_j2c)

    else:
        uniq_kpts, uniq_index, uniq_inverse = unique_with_wrap_around(
            cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
        scaled_uniq_kpts = cell.get_scaled_kpts(uniq_kpts).round(5)
        log.debug('Num uniq kpts %d', len(uniq_kpts))
        log.debug2('scaled unique kpts %s', scaled_uniq_kpts)

        for k, j2c in enumerate(rsdf.get_2c2e(uniq_kpts)):
            fswap[f'j2c/{k}'] = j2c
            j2c = None
        cpu1 = logger.timer(rsdf, 'short range part of int2c2e', *cpu1)

        def get_conj_j2c(cholesky_j2c):
            j2c, j2c_negative, j2ctag = cholesky_j2c
            if j2c_negative is None:
                return j2c.conj(), None, j2ctag
            else:
                return j2c.conj(), j2c_negative.conj(), j2ctag

        for k, k_conj in group_by_conj_pairs(cell, uniq_kpts)[0]:
            # Find ki's and kj's that satisfy k_aux = kj - ki
            log.debug1('Cholesky decomposition for j2c at kpt %s %s',
                       k, scaled_uniq_kpts[k])
            cholesky_j2c = rsdf.cholesky_decomposed_metric(fswap[f'j2c/{k}'])
            if cholesky_j2c[2] == 'eig':
                log.debug('DF metric linear dependency for unique kpt %s', k)
            kpt_ij_idx = np.where(uniq_inverse == k)[0]
            make_kpt(uniq_kpts[k], kpt_ij_idx, cholesky_j2c)

            if k_conj is None:
                continue

            # Swap ki, kj for the conjugated case
            log.debug1('Cholesky decomposition for the conjugated kpt %s %s',
                       k_conj, scaled_uniq_kpts[k_conj])
            kpt_ji_idx = np.where(uniq_inverse == k_conj)[0]
            cholesky_j2c = get_conj_j2c(cholesky_j2c)
            make_kpt(uniq_kpts[k_conj], kpt_ji_idx, cholesky_j2c)

    cpu1 = logger.timer(rsdf, 'pass2: AFT int3c2e', *cpu1)


class _RangeSeparationDFBuilder(lib.StreamObject):
    def __init__(self, cell, auxcell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.auxcell = auxcell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = None
        self.kpts = np.reshape(kpts, (-1, 3))

        self.omega = None
        self.rs_cell = None
        self.rs_auxcell = None
        # mesh to generate Born-von Karman supercell
        self.bvk_kmesh = None
        self.supmol_sr = None
        self.supmol_ft = None
        self.ke_cutoff = None
        self.precision = None

        self.cintopt = lib.c_null_ptr()
        self.q_cond = None  # integral screening condition

        # In outcore_auxe2, exclude smooth auxiliary basis (D|**) and evaluate
        # only the (C|CC), (C|CD), (C|DC) blocks
        self.compact_sr_j3c = True

        # to mimic molecular DF object
        self.blockdim = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)

        self._keys = set(self.__dict__.keys())

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.rs_cell = None
        self.rs_auxcell = None
        self.supmol_sr = None
        self.supmol_ft = None
        return self

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'omega = %s', self.omega)
        #logger.info(self, 'len(kpts) = %d', len(self.kpts))
        #logger.debug1(self, '    kpts = %s', self.kpts)
        return self

    def build(self, omega=None, precision=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        auxcell = self.auxcell
        kpts = self.kpts

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if omega is not None:
            self.omega = omega

        if self.omega is None:
            # Search a proper range-separation parameter omega that can balance the
            # computational cost between the real space integrals and moment space
            # integrals
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(auxcell, kpts, self.mesh)
            # TODO: self.omega = min(self.omega, 1.1) or min(self.mesh, [30, 30, 30]
        else:
            self.ke_cutoff = aft.estimate_ke_cutoff_for_omega(cell, self.omega)
            self.mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), self.ke_cutoff)

        log.info('omega = %.15g  ke_cutoff = %s  mesh = %s',
                 self.omega, self.ke_cutoff, self.mesh)

        if precision is None:
            precision = cell.precision**1.5
            logger.debug(self, 'Set precision %g', precision)
        self.precision = precision

        self.rs_cell = rs_cell = ft_ao._RangeSeperationCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, precision, log)
        self.rs_auxcell = rs_auxcell = ft_ao._RangeSeperationCell.from_cell(
            auxcell, self.ke_cutoff, precision, verbose=log)

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
            compact_aux_idx = np.where(rs_auxcell.bas_type != SMOOTH_BASIS)[0]
            exp_aux_min = min([rs_auxcell.bas_exp(ib).min() for ib in compact_aux_idx])
            theta = 1/(self.omega**-2 + 1./aij + 1./exp_aux_min)
            fac = ((8*np.pi*exp_d_min*exp_c_min/(aij*exp_aux_min)**2)**.75
                   / (theta * np.pi)**.5)
            # x = rcut * x_ratio for the distance between compact function
            # and smooth function (smooth function in the far end)
            # fac*erfc(\sqrt(theta)|rcut - x|) for the asymptotic value of short-range eri
            x_ratio = 1. / (exp_c_min/aij + exp_d_min/theta)
            exp_fac = eij * x_ratio**2 + theta * (1 - exp_c_min/aij*x_ratio)**2

            rcut_sr = cell.rcut  # initial guess
            rcut_sr = ((-np.log(precision * LATTICE_SUM_PENALTY
                                * rcut_sr / (2*np.pi*fac)) / exp_fac)**.5
                       + pbcgto.cell._rcut_penalty(cell))
            log.debug1('exp_d_min = %g, exp_c_min = %g, exp_aux_min = %g, rcut_sr = %g',
                       exp_d_min, exp_c_min, exp_aux_min, rcut_sr)

        self.supmol_ft = _ExtendedMoleFT.from_cell(rs_cell, kmesh, rcut_sr, log)
        self.supmol_sr = supmol = _ExtendedMoleSR.from_cell(
            rs_cell, kmesh, self.omega, rcut_sr, log)
        self.cintopt = _vhf.make_cintopt(supmol._atm, supmol._bas, supmol._env, 'int3c2e_sph')

        log.timer_debug1('initializing supmol', *cpu0)
        log.info('sup-mol nbas = %d cGTO = %d pGTO = %d',
                 supmol.nbas, supmol.nao, supmol.npgto_nr())
        return self

    weighted_coulG = aft.weighted_coulG

    gen_int3c_kernel = _gen_int3c_kernel
    outcore_auxe2 = _outcore_auxe2
    outcore_smooth_block = _outcore_smooth_block
    make_j3c = _make_j3c

    def get_qcond(self, intor='int3c2e_sph'):
        '''Integral screening condition max(sqrt((ij|ij))) inside supmol and
        max(sqrt((k|ii))) cross auxcell and supmol'''
        supmol = self.supmol_sr
        auxcell_s = self.rs_auxcell.copy()
        auxcell_s._bas[:,ANG_OF] = 0

        nbas = supmol.nbas
        q_cond = np.empty((nbas+auxcell_s.nbas, nbas))
        q_cond_supmol = q_cond[:nbas]
        q_cond_aux= q_cond[nbas:]
        with supmol.with_integral_screen(self.precision**2):
            atm, bas, env = gto.conc_env(supmol._atm, supmol._bas, supmol._env,
                                         auxcell_s._atm, auxcell_s._bas, auxcell_s._env)
            ao_loc = gto.moleintor.make_loc(bas, intor)
            libpbc.CVHFset_int2e_q_cond(
                getattr(libpbc, 'int2e_sph'), lib.c_null_ptr(),
                q_cond_supmol.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
                env.ctypes.data_as(ctypes.c_void_p))

            shls_slice = (0, supmol.nbas, supmol.nbas, len(bas))
            libpbc.PBC_nr3c_q_cond(
                getattr(libpbc, intor), self.cintopt,
                q_cond_aux.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.natm),
                bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(supmol.nbas),
                env.ctypes.data_as(ctypes.c_void_p))

        # Remove d-d block in supmol q_cond
        diffused_mask = supmol.bas_type_to_indices(SMOOTH_BASIS)
        q_cond_supmol[diffused_mask[:,None], diffused_mask] = 1e-200

        if self.compact_sr_j3c:
            # Assign a very small value to q_cond to avoid dividing 0 error
            q_cond_aux[self.rs_auxcell.bas_type == SMOOTH_BASIS] = 1e-200
        return q_cond

    def cholesky_decomposed_metric(self, j2c):
        cell = self.cell
        j2c = np.asarray(j2c)
        j2c_negative = None
        try:
            j2c = scipy.linalg.cholesky(j2c, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that mesh is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
            w, v = scipy.linalg.eigh(j2c)
            logger.debug(self, 'cond = %.4g, drop %d bfns',
                         w[-1]/w[0], np.count_nonzero(w<self.linear_dep_threshold))
            v1 = v[:,w>self.linear_dep_threshold].conj().T
            v1 /= np.sqrt(w[w>self.linear_dep_threshold]).reshape(-1,1)
            j2c = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = np.where(w < -self.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
            w = v = None
            j2ctag = 'eig'
        return j2c, j2c_negative, j2ctag

    def get_2c2e(self, uniq_kpts):
        # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
        auxcell = self.auxcell
        rs_auxcell = self.rs_auxcell
        auxcell_c = rs_auxcell.compact_basis_cell()

        if auxcell_c.nbas > 0:
            rcut_sr = auxcell_c.rcut
            rcut_sr = (-2*np.log(.225*self.precision * self.omega**4 * rcut_sr**2))**.5 / self.omega
            auxcell_c.rcut = rcut_sr
            logger.debug1(self, 'auxcell_c  rcut_sr = %g', rcut_sr)
            with auxcell_c.with_short_range_coulomb(self.omega):
                sr_j2c = auxcell_c.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
            recontract_1d = rs_auxcell.recontract()

            compact_bas_idx = np.where(rs_auxcell.bas_type != SMOOTH_BASIS)[0]
            compact_ao_idx = rs_auxcell.get_ao_indices(compact_bas_idx)
            ao_map = auxcell.get_ao_indices(rs_auxcell.bas_map[compact_bas_idx])

            def recontract_2d(j2c, j2c_cc):
                return lib.takebak_2d(j2c, j2c_cc, ao_map, ao_map, thread_safe=False)
        else:
            recontract_2d = None

        mesh = self.mesh
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        b = auxcell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

        ngrids = Gv.shape[0]
        naux_rs = rs_auxcell.nao
        naux = auxcell.nao
        max_memory = max(1000, self.max_memory - lib.current_memory()[0])
        blksize = min(ngrids, int(max_memory*.4e6/16/naux_rs), 200000)
        logger.debug2(self, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)
        j2c = []
        for k, kpt in enumerate(uniq_kpts):
            coulG = self.weighted_coulG(kpt, False, mesh)
            if recontract_2d is not None:
                # Add Long range part to compact subset of auxcell basis
                coulG_sr = self.weighted_coulG_LR(kpt, False, mesh) - coulG
                if auxcell.dimension >= 2 and is_zero(kpt):
                    G0_idx = 0  # due to np.fft.fftfreq convension
                    G0_weight = kws[G0_idx] if isinstance(kws, np.ndarray) else kws
                    coulG_sr[G0_idx] -= np.pi/self.omega**2 * G0_weight

            if is_zero(kpt):  # kpti == kptj
                j2c_k = np.zeros((naux, naux))
            else:
                j2c_k = np.zeros((naux, naux), dtype=np.complex128)

            for p0, p1 in lib.prange(0, ngrids, blksize):
                if recontract_2d is None:
                    auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                    if is_zero(kpt):  # kpti == kptj
                        j2c_k += lib.dot(auxG.conj() * coulG, auxG.T).real
                    else:
                        #j2cR, j2cI = zdotCN(LkR*coulG[p0:p1],
                        #                    LkI*coulG[p0:p1], LkR.T, LkI.T)
                        j2c_k += lib.dot(auxG.conj() * coulG, auxG.T)
                else:
                    auxG = ft_ao.ft_ao(rs_auxcell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
                    auxG_sr = auxG[compact_ao_idx]
                    if is_zero(kpt):
                        sr_j2c[k] += lib.dot(auxG_sr.conj() * coulG_sr, auxG_sr.T).real
                    else:
                        sr_j2c[k] += lib.dot(auxG_sr.conj() * coulG_sr, auxG_sr.T)
                    auxG = recontract_1d(auxG)
                    if is_zero(kpt):  # kpti == kptj
                        j2c_k += lib.dot(auxG.conj() * coulG, auxG.T).real
                    else:
                        j2c_k += lib.dot(auxG.conj() * coulG, auxG.T)
                auxG = auxG_sr = None

            if recontract_2d is not None:
                j2c_k = recontract_2d(j2c_k, sr_j2c[k])
            j2c.append(j2c_k)
        return j2c

    weighted_coulG_LR = aft.weighted_coulG_LR
    weighted_coulG_SR = aft.weighted_coulG_SR


class _ExtendedMoleSR(ft_ao._ExtendedMole):
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
            # compact_aux_idx = np.where(rs_auxcell.bas_type != SMOOTH_BASIS)[0]
            # exp_aux_min = min([rs_auxcell.bas_exp(ib).min() for ib in compact_aux_idx])
            # Is the exact exp_aux_min needed here?
            exp_aux_min = exp_min
            aij = exp_min + exps_c
            eij = exp_min * exps_c / aij
            theta = 1./(omega**-2 + 1./aij + 1./exp_aux_min)
            rLK = np.linalg.norm(LKs, axis=2) - pbcgto.cell._rcut_penalty(cell)
            rLK[rLK < 1e-2] = 1e-2  # avoid singularity in upper_bounds

            # x = rcut * x_ratio for the distance between compact function
            # and smooth function (compact function in the far end)
            # fac*erfc(\sqrt(theta)|rcut - x|) for the asymptotic value of short-range eri
            x_ratio = 1. / (exp_min/aij + exps_c/theta)
            exp_fac = eij * x_ratio**2 + theta * (1 - exp_min/aij*x_ratio)**2
            fac = ((8*np.pi*exp_min*exps_c/(aij*exp_aux_min)**2)**.75
                   / (theta * np.pi)**.5)
            # upper_bounds are the maximum values int3c2e can reach for each
            # basis in each repeated image. shape (bas_id, image_id, bvk_cell_id)
            upper_bounds = np.einsum('i,lk,ilk->ilk', fac, 2*np.pi/rLK,
                                     np.exp(-exp_fac[:,None,None]*rLK**2))
            cutoff = supmol.precision * LATTICE_SUM_PENALTY
            bas_mask[:,compact_bas_mask] = upper_bounds.transpose(2,0,1) > cutoff

            # determine rcut boundary for diffused functions
            exps_d = exps[~compact_bas_mask]
            if exps_d.size > 0:
                exp_c_min = exps_c.min()
                aij = exp_c_min + exps_d
                eij = exp_c_min * exps_d / aij
                theta = 1./(omega**-2 + 1./aij + 1./exp_aux_min)

                x_ratio = 1. / (exps_d/aij + exp_c_min/theta)
                exp_fac = eij * x_ratio**2 + theta * (1 - exps_d/aij*x_ratio)**2
                fac = ((8*np.pi*exps_d*exp_c_min/(aij*exp_aux_min)**2)**.75
                       / (theta * np.pi)**.5)
                # upper_bounds are the maximum values int3c2e can reach for each
                # basis in each repeated image. shape (bas_id, image_id, bvk_cell_id)
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

    def bas_type_to_indices(self, type_code=SMOOTH_BASIS):
        '''Return the basis indices of required bas_type'''
        cell0_mask = self.rs_cell.bas_type == type_code
        if np.any(cell0_mask):
            # (bvk_ncells, rs_cell.nbas, nimgs)
            bas_type_mask = np.empty_like(self.bas_mask)
            bas_type_mask[:] = cell0_mask[None,:,None]
            bas_type_mask = bas_type_mask[self.bas_mask]
            return np.where(bas_type_mask)[0]
        else:
            return np.arange(0)

class _ExtendedMoleFT(ft_ao._ExtendedMole):
    '''Extended Mole for Fourier Transform without dd-blocks'''

    def get_ovlp_mask(self):
        '''integral screening mask for basis product between cell and supmol.
        The diffused-diffused basis block are removed
        '''
        ovlp_mask = super().get_ovlp_mask()
        bvk_ncells, rs_nbas, nimgs = self.bas_mask.shape
        ovlp_mask_view = ovlp_mask.reshape(rs_nbas, bvk_ncells, rs_nbas, nimgs)
        smooth_idx = np.where(self.rs_cell.bas_type == SMOOTH_BASIS)[0]
        # Mute diffused-diffused block
        ovlp_mask_view[smooth_idx[:,None,None,None],:,smooth_idx[:,None]] = 0
        return ovlp_mask

# ngrids ~= 8*naux = prod(mesh)
def _guess_omega(auxcell, kpts, mesh=None):
    naux = auxcell.npgto_nr()
    if mesh is None:
        mesh = [max(4, int((8 * naux) ** (1./3) + .5))] * 3
    a = auxcell.lattice_vectors()
    ke_cutoff = min(pbctools.mesh_to_cutoff(a, mesh[:auxcell.dimension]))
    omega = aft.estimate_omega_for_ke_cutoff(auxcell, ke_cutoff)
    return omega, mesh, ke_cutoff

libpbc.GTOmax_cache_size.restype = ctypes.c_int
def _get_cache_size(cell, intor):
    '''Cache size for libcint integrals. Cache size cannot be accurately
    estimated in function PBC_ft_bvk_drv
    '''
    cache_size = libpbc.GTOmax_cache_size(
        getattr(libpbc, intor), (ctypes.c_int*2)(0, cell.nbas), ctypes.c_int(1),
        cell._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
        cell._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
        cell._env.ctypes.data_as(ctypes.c_void_p))
    return cache_size

def _conc_locs(cell_loc, auxcell_loc):
    '''auxiliary basis was appended to regular AO basis when calling int3c2e
    integrals. Composite loc combines locs from regular AO basis and auxiliary
    basis accordingly.'''
    comp_loc = np.append(cell_loc[:-1], cell_loc[-1] + auxcell_loc)
    return np.asarray(comp_loc, dtype=np.int32)

def _eval_gto(cell, mesh, kpts):
    coords = cell.get_uniform_grids(mesh)
    nkpts = len(kpts)
    nao = cell.nao
    ngrids = len(coords)

    ao_ks = cell.pbc_eval_gto('GTOval', coords, kpts=kpts)

    aoR_ks = np.empty((nkpts, nao, ngrids))
    aoI_ks = np.empty((nkpts, nao, ngrids))
    for k, dat in enumerate(ao_ks):
        aoR_ks[k] = dat.real.T
        aoI_ks[k] = dat.imag.T
    return aoR_ks, aoI_ks
