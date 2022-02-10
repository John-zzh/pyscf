#!/usr/bin/env python

import numpy as np

def eval_xc_eff(fn_eval_xc, rho_tm, deriv=1,
                xctype='LDA', convention='ud', ang_samples=5810):
    '''
    convention:
        'ud': for spin-up and spin-density
        'ts': for total-density and spin-density
    '''
    assert deriv < 3

    ngrids = rho_tm[0].shape[-1]
    directions, weights = _make_ang_samples(ang_samples)
    blksize = int(np.ceil(1e5/ngrids)) * 8

    for p0, p1 in _prange(0, ang_samples, blksize):
        n_ang = p1 - p0
        p_directions = directions[p0:p1]
        p_weights = weights[p0:p1]
        rho = _project_spin(rho_tm, p_directions, xctype, convention)
        exc, vxc, fxc, kxc = fn_eval_xc(rho, deriv+1)

        nvar = vxc.shape[1]
        exc = exc.reshape(ngrids,n_ang)
        vxc = vxc.reshape(2,nvar,ngrids,n_ang)

        if convention == 'ud':
            s = (rho[0] - rho[1]).reshape(nvar,ngrids,n_ang)
            dvds = (vxc[1] - vxc[0]) * .5
            exc += np.einsum('xgo,xgo->go', dvds, s)
            exc = np.einsum('go,o->g', exc, p_weights)

            if deriv > 0:
                fxc = fxc.reshape(2,nvar,2,nvar,ngrids,n_ang)
                # The second part dV/ds * s
                dvds = (fxc[0] - fxc[1]) * .5
                vxc += np.einsum('xbygo,xgo->bygo', dvds, s)
                # Transform with the first part V
                c_tm = _ud2tm_transform(p_directions)
                cw_tm = c_tm * p_weights
                vxc = np.einsum('rao,axgo->rxg', cw_tm, vxc)

            if deriv > 1:
                kxc = kxc.reshape(2,nvar,2,nvar,2,nvar,ngrids,n_ang)
                dvds = (kxc[1] - kxc[0]) * .5
                fxc += np.einsum('xbyczgo,xgo->byczgo', dvds, s)
                fxc = np.einsum('rao,axbygo->rxbygo', c_tm, fxc)
                fxc = np.einsum('sbo,rxbygo->rxsyg', cw_tm, fxc)
        else:
            s = rho[1].rehsape(nvar,ngrids,n_ang)
            exc += np.einsum('xgo,xgo->go', vxc[1], s)
            exc = np.einsum('go,o->g', exc, p_weights)

            if deriv > 0:
                fxc = fxc.reshape(2,nvar,2,nvar,ngrids,n_ang)
                vxc += np.einsum('xbygo,xgo->bygo', fxc[1], s)
                c_tm = _ts2tm_transform(p_directions)
                cw_tm = c_tm * p_weights
                vxc = np.einsum('rao,axgo->rxg', cw_tm, vxc)

            if deriv > 1:
                kxc = kxc.reshape(2,nvar,2,nvar,2,nvar,ngrids,n_ang)
                fxc += np.einsum('xbyczgo,xgo->byczgo', kxc[1], s)
                fxc = np.einsum('rao,axbygo->rxbygo', c_tm, fxc)
                fxc = np.einsum('sbo,rxbygo->rxsyg', cw_tm, fxc)

    kxc = None
    return exc, vxc, fxc, kxc

def _make_ang_samples(ang_samples):
    import ctypes
    from pyscf.dft.gen_grid import libdft
    ang_grids = np.empty((ang_samples, 4))
    libdft.MakeAngularGrid(ang_grids.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(ang_samples))
    directions = ang_grids[:,:3].copy(order='F')
    weights = ang_grids[:,3].copy()
    return directions, weights

def _prange(start, end, step):
    if start < end:
        for i in range(start, end, step):
            yield i, min(i+step, end)

def _project_spin(rho_tm, directions, xctype, convention):
    rho, m = rho_tm
    n_ang = directions.shape[0]
    ngrids = rho.shape[-1]
    if xctype == 'LDA':
        s = np.einsum('mg,om->go', m, directions)
        if convention == 'ud':
            rhoa = (rho[:,None] + s) * .5
            rhob = (rho[:,None] - s) * .5
            rho = (rhoa.ravel(), rhob.ravel())
        else:
            rho = (np.repeat(rho[:,None], n_ang, axis=1), s)
    else:
        s = np.einsum('mxg,om->xgo', m, directions)
        if convention == 'ud':
            rhoa = (rho[:,:,None] + s) * .5
            rhob = (rho[:,:,None] - s) * .5
            rho = (rhoa.reshape(-1, ngrids*n_ang), rhob.reshape(-1, ngrids*n_ang))
        else:
            rho = (np.repeat(rho[:,:,None], n_ang, axis=2), s)
    return rho

def _ts2tm_transform(directions):
    '''
    projects v_ts(rho,s) in each directon to rho/m representation (rho,mx,my,mz)
    '''
    n_ang = directions.shape[0]
    c_tm = np.zeros((4, 2, n_ang))
    c_tm[0,0] = 1
    c_tm[1:3,1] = directions.T
    return c_tm

def _ud2tm_transform(directions):
    '''
    projects v_ud(rhou,rhod) in each directon to rho/m representation (rho,mx,my,mz)
    '''
    n_ang = directions.shape[0]
    c_ts = np.array([[.5,  .5],    # vrho = (va + vb) / 2
                     [.5, -.5]])   # vs   = (va - vb) / 2
    c_tm = np.empty((4, 2, n_ang))
    c_tm[0] = c_ts[0,:,None]
    c_tm[1] = c_ts[1,:,None] * directions.T[:,None,:]
    return c_tm
