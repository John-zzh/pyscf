#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
DKS with collinear functional
'''

from pyscf import gto

mol = gto.M(atom="O", basis='unc-sto3g', verbose=4)
mf = mol.DKS()
mf.xc = 'pbe'
# Enable collinear functional. DKS calls non-collinear functional by default
mf.collinear = True
mf.kernel()
