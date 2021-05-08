/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

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
        int nbasp;
        int *sh_loc;
        double *expLkR;
        double *expLkI;
        // the bvk-cell id for each shell in bvk-supcell
        int *cell_id;
        // shell id in the primitive cell for each shell in bvk-supcell
        int *cell0_shl_id;
} BVKEnvs;
#endif
