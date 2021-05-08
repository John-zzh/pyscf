/* Copyright 2014-2018, 2021 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "gto/ft_ao.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "pbc/pbc.h"
#include "np_helper/np_helper.h"

#define OF_CMPLX        2
#define BLOCK_SIZE      104

typedef int (*FPtr_intor)(double *outR, double *outI, int *shls, int *dims,
                          FPtr_eval_gz eval_gz, double complex fac,
                          double *Gv, double *b, int *gxyz, int *gs, int nGv,
                          int block_size,
                          int *atm, int natm, int *bas, int nbas, double *env);

static int _assemble2c(FPtr_intor intor, FPtr_eval_gz eval_gz,
                       double *eriR, double *eriI, //double *cache,
                       int grid0, int grid1, int ish_bvk, int jsh_bvk,
                       double complex fac, IntorEnvs *envs_cint, BVKEnvs *envs_bvk)
{
        int natm = envs_cint->natm;
        int nbas = envs_cint->nbas;
//        int ncomp = envs_cint->ncomp;
        int *sh_loc = envs_bvk->sh_loc;
        int ish0 = sh_loc[ish_bvk];
        int jsh0 = sh_loc[jsh_bvk];
        int ish1 = sh_loc[ish_bvk+1];
        int jsh1 = sh_loc[jsh_bvk+1];
        int empty = 1;
        if (ish0 == ish1 || jsh0 == jsh1) {
                return !empty;
        }

        int ngrids = envs_bvk->nGv;
        int dg = grid1 - grid0;
        int *atm = envs_cint->atm;
        int *bas = envs_cint->bas;
        double *env = envs_cint->env;
        double *Gv = envs_bvk->Gv;
        double *b = envs_bvk->b;
        int nimgs = envs_bvk->nimgs;
        int *gxyz = envs_bvk->gxyz;
        int *gs = envs_bvk->gs;
        char *ovlp_mask = envs_bvk->ovlp_mask;
        int shls[2];
        int ish, jsh, ishp;

        for (ish = ish0; ish < ish1; ish += nimgs) {
                ishp = ish / nimgs;
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        if (!ovlp_mask[ishp*nbas+jsh]) {
                                continue;
                        }
                        shls[0] = ish;
                        shls[1] = jsh;
                        if ((*intor)(eriR, eriI, shls, NULL, eval_gz,
                                     fac, Gv+grid0, b, gxyz+grid0, gs, ngrids, dg,
                                     atm, natm, bas, nbas, env)) {
                                empty = 0;
                        }
                }
        }
        return !empty;
}

/*
 * Multiple k-points for BvK cell
 */
void PBC_ft_bvk_ks1(FPtr_intor intor, FPtr_eval_gz eval_gz, void (*fsort)(),
                    double *out, int ish_cell0, int jsh_cell0, double *buf,
                    IntorEnvs *envs_cint, BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_cint->shls_slice;
        int *cell0_ao_loc = envs_cint->ao_loc;
        int ish0 = shls_slice[0];
        int jsh0 = shls_slice[2];
        ish_cell0 += ish0;
        jsh_cell0 += jsh0;

        int di = cell0_ao_loc[ish_cell0+1] - cell0_ao_loc[ish_cell0];
        int dj = cell0_ao_loc[jsh_cell0+1] - cell0_ao_loc[jsh_cell0];
        int dij = di * dj;
        char TRANS_N = 'N';
        char TRANS_T = 'T';
        double D0 = 0;
        double D1 = 1;
        double ND1 = -1;
        double complex Z1 = 1;

        int comp = envs_cint->ncomp;
        int nGv = envs_bvk->nGv;
        int bvk_ncells = envs_bvk->ncells;
        int nkpts = envs_bvk->nkpts;
        int nbasp = envs_bvk->nbasp;
        double *expLkR = envs_bvk->expLkR;
        double *expLkI = envs_bvk->expLkI;
        double *bufkR = buf;
        double *bufkI = bufkR + ((size_t)dij) * BLOCK_SIZE * comp * nkpts;
        double *bufLR = bufkI + ((size_t)dij) * BLOCK_SIZE * comp * nkpts;
        double *bufLI = bufLR + ((size_t)dij) * BLOCK_SIZE * comp * bvk_ncells;
        //double *cache = bufLI + ((size_t)dij) * BLOCK_SIZE * comp * bvk_ncells;
        double *pbufR, *pbufI;
        int grid0, grid1, dg, dijg, jL, jLmax, nLj;

        // TODO: precompute opts??

        for (grid0 = 0; grid0 < nGv; grid0 += BLOCK_SIZE) {
                grid1 = MIN(grid0+BLOCK_SIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg * comp;

                jLmax = 0;
                for (jL = 0; jL < bvk_ncells; jL++) {
                        pbufR = bufLR + jL * dijg;
                        pbufI = bufLI + jL * dijg;
                        NPdset0(pbufR, dijg);
                        NPdset0(pbufI, dijg);
                        if (_assemble2c(intor, eval_gz, pbufR, pbufI, grid0, grid1,
                                        ish_cell0, jL*nbasp+jsh_cell0, Z1,
                                        envs_cint, envs_bvk)) {
                                jLmax = jL;
                        }
                }

                nLj = jLmax + 1;
                dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                       &D1, bufLR, &dijg, expLkR, &nkpts, &D0, bufkR, &dijg);
                dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                       &ND1, bufLI, &dijg, expLkI, &nkpts, &D1, bufkR, &dijg);
                dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                       &D1, bufLR, &dijg, expLkI, &nkpts, &D0, bufkI, &dijg);
                dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                       &D1, bufLI, &dijg, expLkR, &nkpts, &D1, bufkI, &dijg);

                (*fsort)(out, bufkR, shls_slice, cell0_ao_loc,
                         nkpts, comp, nGv, ish_cell0, jsh_cell0, grid0, grid1);
        }
}

/*
 * Single k-point for BvK cell
 */
void PBC_ft_bvk_nk1s1(FPtr_intor intor, FPtr_eval_gz eval_gz, void (*fsort)(),
                      double *out, int ish_cell0, int jsh_cell0, double *buf,
                      IntorEnvs *envs_cint, BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_cint->shls_slice;
        int *cell0_ao_loc = envs_cint->ao_loc;
        int ish0 = shls_slice[0];
        int jsh0 = shls_slice[2];
        ish_cell0 += ish0;
        jsh_cell0 += jsh0;

        int di = cell0_ao_loc[ish_cell0+1] - cell0_ao_loc[ish_cell0];
        int dj = cell0_ao_loc[jsh_cell0+1] - cell0_ao_loc[jsh_cell0];
        int dij = di * dj;
        int comp = envs_cint->ncomp;
        int nGv = envs_bvk->nGv;
        int bvk_ncells = envs_bvk->ncells;
        int nkpts = envs_bvk->nkpts;
        int nbasp = envs_bvk->nbasp;
        double *expLkR = envs_bvk->expLkR;
        double *expLkI = envs_bvk->expLkI;
        double *bufR = buf;
        double *bufI = bufR + dij * BLOCK_SIZE * comp;
        double complex fac;
        int grid0, grid1, dg, jL, dijg;

        for (grid0 = 0; grid0 < nGv; grid0 += BLOCK_SIZE) {
                grid1 = MIN(grid0+BLOCK_SIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg * comp;
                NPdset0(bufR, dijg);
                NPdset0(bufI, dijg);

                for (jL = 0; jL < bvk_ncells; jL++) {
                        fac = expLkR[jL] + expLkI[jL] * _Complex_I;
                        _assemble2c(intor, eval_gz, bufR, bufI, grid0, grid1,
                                    ish_cell0, jL*nbasp+jsh_cell0, fac,
                                    envs_cint, envs_bvk);
                }

                (*fsort)(out, bufR, shls_slice, cell0_ao_loc,
                         nkpts, comp, nGv, ish_cell0, jsh_cell0, grid0, grid1);
        }
}

void PBC_ft_dsort_s1(double *out, double *in,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nijg = naoi * naoj * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int ip = ao_loc[ish] - ao_loc[ish0];
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        int dg = grid1 - grid0;
        int dij = di * dj;
        int dijg = dij * dg;
        double *outR = out + (ip * naoj + jp) * NGv + grid0;
        double *outI = outR + nijg * nkpts * comp;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int i, j, n, ic, kk;
        double *pinR, *pinI, *poutR, *poutI;

        for (kk = 0; kk < nkpts; kk++) {
        for (ic = 0; ic < comp; ic++) {
                for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        poutR = outR + (i*naoj+j) * NGv;
                        poutI = outI + (i*naoj+j) * NGv;
                        pinR  = inR + (j*di+i) * dg;
                        pinI  = inI + (j*di+i) * dg;
                        for (n = 0; n < dg; n++) {
                                poutR[n] = pinR[n];
                                poutI[n] = pinI[n];
                        }
                } }
                outR += nijg;
                outI += nijg;
                inR  += dijg;
                inI  += dijg;
        } }
}

void PBC_ft_dsort_s2(double *out, double *in,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        size_t nij  = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        size_t nijg = nij * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int dg = grid1 - grid0;
        size_t dijg = dij * dg;
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        double *outR = out + (((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * NGv + grid0;
        double *outI = outR + nijg * nkpts * comp;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int ip1 = ao_loc[ish] + 1;
        int i, j, n, ic, kk;
        double *pinR, *pinI, *poutR, *poutI;

        if (ish != jsh) {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        poutR = outR + (kk * comp + ic) * nijg;
                        poutI = outI + (kk * comp + ic) * nijg;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        pinR = inR + (j*di+i) * dg;
                                        pinI = inI + (j*di+i) * dg;
                                        for (n = 0; n < dg; n++) {
                                                poutR[j*NGv+n] = pinR[n];
                                                poutI[j*NGv+n] = pinI[n];
                                        }
                                }
                                poutR += (ip1 + i) * NGv;
                                poutI += (ip1 + i) * NGv;
                        }
                        inR += dijg;
                        inI += dijg;
                } }
        } else {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        poutR = outR + (kk * comp + ic) * nijg;
                        poutI = outI + (kk * comp + ic) * nijg;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        pinR = inR + (j*di+i) * dg;
                                        pinI = inI + (j*di+i) * dg;
                                        for (n = 0; n < dg; n++) {
                                                poutR[j*NGv+n] = pinR[n];
                                                poutI[j*NGv+n] = pinI[n];
                                        }
                                }
                                poutR += (ip1 + i) * NGv;
                                poutI += (ip1 + i) * NGv;
                        }
                        inR += dijg;
                        inI += dijg;
                } }
        }
}

void PBC_ft_zsort_s1(double *out, double *in,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nijg = naoi * naoj * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int ip = ao_loc[ish] - ao_loc[ish0];
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        int dg = grid1 - grid0;
        int dij = di * dj;
        int dijg = dij * dg;
        out += ((ip * naoj + jp) * NGv + grid0) * OF_CMPLX;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int i, j, n, ic, kk;
        double *pinR, *pinI, *pout;

        for (kk = 0; kk < nkpts; kk++) {
        for (ic = 0; ic < comp; ic++) {
                for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        pout = out + (i*naoj+j) * NGv * OF_CMPLX;
                        pinR = inR + (j*di+i) * dg;
                        pinI = inI + (j*di+i) * dg;
                        for (n = 0; n < dg; n++) {
                                pout[n*OF_CMPLX  ] = pinR[n];
                                pout[n*OF_CMPLX+1] = pinI[n];
                        }
                } }
                out += nijg * OF_CMPLX;
                inR += dijg;
                inI += dijg;
        } }
}

void PBC_ft_zsort_s2(double *out, double *in,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        size_t nij  = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        size_t nijg = nij * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int dg = grid1 - grid0;
        size_t dijg = dij * dg;
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += ((((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * NGv + grid0) * OF_CMPLX;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int ip1 = ao_loc[ish] + 1;
        int i, j, n, ic, kk;
        double *pinR, *pinI, *pout;

        if (ish != jsh) {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        pout = out + (kk * comp + ic) * nijg * OF_CMPLX;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        pinR = inR + (j*di+i) * dg;
                                        pinI = inI + (j*di+i) * dg;
                                        for (n = 0; n < dg; n++) {
                                                pout[(j*NGv+n)*OF_CMPLX  ] = pinR[n];
                                                pout[(j*NGv+n)*OF_CMPLX+1] = pinI[n];
                                        }
                                }
                                pout += (ip1 + i) * NGv * OF_CMPLX;
                        }
                        inR += dijg;
                        inI += dijg;
                } }
        } else {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        pout = out + (kk * comp + ic) * nijg * OF_CMPLX;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        pinR = inR + (j*di+i) * dg;
                                        pinI = inI + (j*di+i) * dg;
                                        for (n = 0; n < dg; n++) {
                                                pout[(j*NGv+n)*OF_CMPLX  ] = pinR[n];
                                                pout[(j*NGv+n)*OF_CMPLX+1] = pinI[n];
                                        }
                                }
                                pout += (ip1 + i) * NGv * OF_CMPLX;
                        }
                        inR += dijg;
                        inI += dijg;
                } }
        }
}

void PBC_ft_bvk_ks2(FPtr_intor intor, FPtr_eval_gz eval_gz, void (*fsort)(),
                    double *out, int ish, int jsh, double *buf,
                    IntorEnvs *envs_cint, BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_cint->shls_slice;
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2];
        if (ip >= jp) {
                PBC_ft_bvk_ks1(intor, eval_gz, fsort, out,
                               ish, jsh, buf, envs_cint, envs_bvk);
        }
}

void PBC_ft_bvk_nk1s2(FPtr_intor intor, FPtr_eval_gz eval_gz, void (*fsort)(),
                      double *out, int ish, int jsh, double *buf,
                      IntorEnvs *envs_cint, BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_cint->shls_slice;
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2];
        if (ip >= jp) {
                PBC_ft_bvk_nk1s1(intor, eval_gz, fsort, out,
                                 ish, jsh, buf, envs_cint, envs_bvk);
        }
}

void PBC_ft_bvk_nk1s1hermi(FPtr_intor intor, FPtr_eval_gz eval_gz, void (*fsort)(),
                           double *out, int ish, int jsh, double *buf,
                           IntorEnvs *envs_cint, BVKEnvs *envs_bvk)
{
        PBC_ft_bvk_nk1s2(intor, eval_gz, fsort, out,
                         ish, jsh, buf, envs_cint, envs_bvk);
}

void PBC_ft_zsort_s1hermi(double *out, double *in,
                          int *shls_slice, int *ao_loc, int nkpts, int comp,
                          int nGv, int ish, int jsh, int grid0, int grid1)
{
        PBC_ft_zsort_s1(out, in, shls_slice, ao_loc, nkpts, comp,
                        nGv, ish, jsh, grid0, grid1);
}

void PBC_ft_bvk_drv(FPtr_intor intor, FPtr_eval_gz eval_gz, void (*fill)(), void (*fsort)(),
                    double *out, double *expLkR, double *expLkI,
                    int *sh_loc, int *cell0_ao_loc, int *shls_slice,
                    int bvk_ncells, int nkpts, int nbasp, int nimgs, int comp,
                    char *ovlp_mask, char *cell0_ovlp_mask,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int nish = ish1 - ish0;
        int njsh = jsh1 - jsh0;
        int di = GTOmax_shell_dim(cell0_ao_loc, shls_slice, 2);
        IntorEnvs envs_cint = {natm, nbas, atm, bas, env, shls_slice, cell0_ao_loc,
                NULL, NULL, comp};
        BVKEnvs envs_bvk = {bvk_ncells, nkpts, nkpts, nbasp, sh_loc, expLkR, expLkI,
                ovlp_mask, NULL, 0., Gv, b, gxyz, gs, nGv, nimgs};

#pragma omp parallel
{
        int ish, jsh, ij;
        size_t count = nkpts + bvk_ncells;
        double *buf = malloc(sizeof(double) * di*di*BLOCK_SIZE*count*comp*OF_CMPLX);
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (!cell0_ovlp_mask[ish*nbasp+jsh]) {
                        continue;
                }
                (*fill)(intor, eval_gz, fsort, out,
                        ish, jsh, buf, &envs_cint, &envs_bvk);
        }
        free(buf);
}
}

void PBC_ft_fuse_dd_s1(double *outR, double *outI, double complex *pqG_dd,
                       int *ao_idx, int *grid_slice, int nao, int naod, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t Nao = nao;
        int ig0 = grid_slice[0];
        int ig1 = grid_slice[1];
        int ng = ig1 - ig0;
#pragma omp parallel
{
        size_t off_out, off_in;
        int i, j, ij, ip, jp, n;
#pragma omp for schedule(static)
        for (ij = 0; ij < naod*naod; ij++) {
                i = ij / naod;
                j = ij % naod;
                ip = ao_idx[i];
                jp = ao_idx[j];
                off_in = (i * naod + j) * Ngrids + ig0;
                off_out = (ip * Nao + jp) * ng;
                for (n = 0; n < ng; n++) {
                        outR[off_out+n] += creal(pqG_dd[off_in+n]);
                        outI[off_out+n] += cimag(pqG_dd[off_in+n]);
                }
        }
}
}

void PBC_ft_fuse_dd_s2(double *outR, double *outI, double complex *pqG_dd,
                       int *ao_idx, int *grid_slice, int nao, int naod, int ngrids)
{
        size_t Ngrids = ngrids;
        int ig0 = grid_slice[0];
        int ig1 = grid_slice[1];
        int ng = ig1 - ig0;
#pragma omp parallel
{
        size_t off_out, off_in;
        int i, j, ij, ip, jp, n;
#pragma omp for schedule(static)
        for (ij = 0; ij < naod*(naod+1)/2; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - i*(i+1)/2;
                ip = ao_idx[i];
                jp = ao_idx[j];
                off_in = (i * naod + j) * Ngrids + ig0;
                off_out = (ip*(ip+1)/2 + jp) * ng;
                for (n = 0; n < ng; n++) {
                        outR[off_out+n] += creal(pqG_dd[off_in+n]);
                        outI[off_out+n] += cimag(pqG_dd[off_in+n]);
                }
        }
}
}

void PBC_estimate_log_overlap(double *out, int *atm, int natm, int *bas, int nbas, double *env)
{
        size_t Nbas = nbas;
        double *exps = malloc(sizeof(double) * Nbas * 4);
        double *rx = exps + Nbas;
        double *ry = rx + Nbas;
        double *rz = ry + Nbas;
        int ptr_coord, nprim, ib;
        double log4 = log(4.) * .75;
        for (ib = 0; ib < nbas; ib++) {
                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, ib));
                rx[ib] = env[ptr_coord+0];
                ry[ib] = env[ptr_coord+1];
                rz[ib] = env[ptr_coord+2];
                nprim = bas(NPRIM_OF, ib);
                exps[ib] = env[bas(PTR_EXP, ib) + nprim - 1];
        }
#pragma omp parallel
{
        int i, j, li, lj;
        double dx, dy, dz, aij, a1, rr, rij, dri, drj;
#pragma omp for schedule(static)
        for (i = 0; i < nbas; i++) {
#pragma GCC ivdep
                for (j = 0; j < nbas; j++) {
                        li = bas(ANG_OF, i);
                        lj = bas(ANG_OF, j);
                        dx = rx[i] - rx[j];
                        dy = ry[i] - ry[j];
                        dz = rz[i] - rz[j];
                        rr = dx * dx + dy * dy + dz * dz;
                        rij = sqrt(rr);
                        aij = exps[i] + exps[j];
                        dri = exps[j] / aij * rij + 1.;
                        drj = exps[i] / aij * rij + 1.;
                        a1 = exps[i] * exps[j] / aij;
                        //out[i*Nbas+j] = -aij * rr;
                        out[i*Nbas+j] = log4 + .75 * log(a1/aij) - a1 * rr
                                + li * log(dri) + lj * log(drj);
                }
        }
}
        free(exps);
}
