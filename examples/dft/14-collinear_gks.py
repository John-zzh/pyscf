#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
GKS with collinear functional
'''

from pyscf import gto

mol = gto.M(atom="O", basis='unc-sto3g', verbose=4)
mf = mol.GKS()
mf.xc = 'pbe'
# Enable collinear functional. GKS calls non-collinear functional by default
mf.collinear = True
mf.kernel()
