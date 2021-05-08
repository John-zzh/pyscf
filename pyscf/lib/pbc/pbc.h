/* Copyright 2021 The PySCF Developers. All Rights Reserved.

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

#if !defined(HAVE_DEFINED_BVKENV_H)
#define HAVE_DEFINED_BVKENV_H
typedef struct {
        int ncells;
        int nkpts;
        int nbands;
        // nbas of primitive cell
        int nbasp;
        // indicates how to map basis in bvk-cell to supmol basis
        int *sh_loc;
        double *expLkR;
        double *expLkI;

        // Integral mask of SupMole based on s-function overlap
        char *ovlp_mask;
        // Integral screening condition
        double *q_cond;
        // cutoff for schwarz condtion
        double cutoff;

        // parameters for ft_ao
        double *Gv;
        double *b;
        int *gxyz;
        int *gs;
        int nGv;
        // number of repeated images associated to cell.rcut
        int nimgs;
} BVKEnvs;
#endif
