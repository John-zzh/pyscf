#!/usr/bin/env python

import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy.special import roots_legendre

def eval_xc_eff(fn_eval_xc, rho_tm, deriv=1, spin_samples=14, max_workers=1,
                scheme='principal-axis'):
    '''
    scheme:
        Lebedev
        principal-axis
    '''
    assert deriv < 3

    if scheme.upper() == 'LEBEDEV':
        model = _eval_xc_lebedev
    else:
        model = _eval_xc_paxis

    if max_workers == 1:
        return model(fn_eval_xc, rho_tm, deriv, spin_samples)
    else:
        if getattr(fn_eval_xc, '__closure__', None):
            warnings.warn(f'Closure {fn_eval_xc} cannot be parallelized by multiprocessing module. '
                          'It is recommended to generate fn_eval_xc with functools.partial.')
            executor = ThreadPoolExecutor
        else:
            executor = ProcessPoolExecutor

        ngrids = rho_tm[0].shape[-1]
        with executor(max_workers=max_workers) as ex:
            futures = []
            for p0, p1 in _prange(0, ngrids, ngrids//max_workers+1):
                futures.append(ex.submit(model, fn_eval_xc,
                                         rho_tm[...,p0:p1], deriv, spin_samples))
            results = [f.result() for f in futures]
        return [None if x[0] is None else np.concatenate(x, axis=-1) for x in zip(*results)]


def _eval_xc_lebedev(fn_eval_xc, rho_tm, deriv, spin_samples):
    ngrids = rho_tm[0].shape[-1]
    sgrids, weights = _make_samples_grids(spin_samples)
    blksize = int(np.ceil(1e6/ngrids)) * 8

    exc_eff = 0
    vxc_eff = fxc_eff = kxc_eff = None
    if deriv > 0:
        vxc_eff = 0
    if deriv > 1:
        fxc_eff = 0
    if deriv > 2:
        kxc_eff = 0

    if rho_tm.ndim == 2:
        nvar = 1
    else:
        nvar = rho_tm.shape[1]

    for p0, p1 in _prange(0, weights.size, blksize):
        nsg = p1 - p0
        p_sgrids = sgrids[p0:p1]
        p_weights = weights[p0:p1]
        rho = _project_spin(rho_tm, p_sgrids)
        exc, vxc, fxc, kxc = fn_eval_xc(rho, deriv+1)

        exc = exc.reshape(ngrids,nsg)
        vxc = vxc.reshape(2,nvar,ngrids,nsg)

        s = rho[1].reshape(nvar,ngrids,nsg)
        exc += np.einsum('xgo,xgo->go', vxc[1], s)
        exc_eff += np.einsum('go,o->g', exc, p_weights)

        if deriv > 0:
            fxc = fxc.reshape(2,nvar,2,nvar,ngrids,nsg)
            # vs * 2 + s*f_s_st
            vxc[1] *= 2
            vxc += np.einsum('xbygo,xgo->bygo', fxc[1], s)
            c_tm = _ts2tm_transform(p_sgrids)
            cw_tm = c_tm * p_weights
            vxc_eff += np.einsum('rao,axgo->rxg', cw_tm, vxc)

        if deriv > 1:
            kxc = kxc.reshape(2,nvar,2,nvar,2,nvar,ngrids,nsg)
            fxc[1,:,1] *= 3
            fxc[:,:,1] *= 2
            fxc[1,:,:] *= 2
            fxc += np.einsum('xbyczgo,xgo->byczgo', kxc[1], s)
            fxc = np.einsum('rao,axbygo->rxbygo', c_tm, fxc)
            fxc_eff += np.einsum('sbo,rxbygo->rxsyg', cw_tm, fxc)

        if deriv > 2:
            raise NotImplementedError

    return exc_eff, vxc_eff, fxc_eff, kxc_eff

def _make_samples_grids(spin_samples):
    import ctypes
    from pyscf.dft.gen_grid import libdft
    ang_grids = np.empty((spin_samples, 4))
    libdft.MakeAngularGrid(ang_grids.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(spin_samples))
    sgrids = ang_grids[:,:3].copy(order='F')
    weights = ang_grids[:,3].copy()
    return sgrids, weights

def _prange(start, end, step):
    if start < end:
        for i in range(start, end, step):
            yield i, min(i+step, end)

def _project_spin(rho_tm, sgrids):
    rho = rho_tm[0]
    m = rho_tm[1:]
    nsg = sgrids.shape[0]
    ngrids = rho.shape[-1]
    if rho_tm.ndim == 2:
        rho_ts = np.empty((2, ngrids, nsg))
        rho_ts[0] = rho[:,None]
        rho_ts[1] = np.einsum('mg,om->go', m, sgrids)
        rho_ts = rho_ts.reshape(2, ngrids*nsg)
    else:
        nvar = rho_tm.shape[1]
        rho_ts = np.empty((2, nvar, ngrids, nsg))
        rho_ts[0] = rho[:,:,None]
        rho_ts[1] = np.einsum('mxg,om->xgo', m, sgrids)
        rho_ts = rho_ts.reshape(2, nvar, ngrids*nsg)
    return rho_ts

def _ts2tm_transform(sgrids):
    '''
    projects v_ts(rho,s) in each directon to rho/m representation (rho,mx,my,mz)
    '''
    nsg = sgrids.shape[0]
    c_tm = np.zeros((4, 2, nsg))
    c_tm[0,0] = 1
    c_tm[1:,1] = sgrids.T
    return c_tm

def _eval_xc_paxis(fn_eval_xc, rho_tm, deriv, spin_samples):
    '''Integration on principal axis'''
    ngrids = rho_tm[0].shape[-1]
    # samples on z=cos(theta) and their weights
    sgridz, weights = _make_paxis_samples(spin_samples)
    blksize = int(np.ceil(1e6/ngrids)) * 8

    exc_eff = 0
    vxc_eff = fxc_eff = kxc_eff = None
    if deriv > 0:
        vxc_eff = 0
    if deriv > 1:
        fxc_eff = 0
    if deriv > 2:
        kxc_eff = 0

    if rho_tm.ndim == 2:
        nvar = 1
    else:
        nvar = rho_tm.shape[1]

    for p0, p1 in _prange(0, weights.size, blksize):
        nsg = p1 - p0
        p_sgridz = sgridz[p0:p1]
        p_weights = weights[p0:p1]
        rho = _project_spin_paxis(rho_tm, p_sgridz)
        exc, vxc, fxc, kxc = fn_eval_xc(rho, deriv+1)

        exc = exc.reshape(ngrids,nsg)
        vxc = vxc.reshape(2,nvar,ngrids,nsg)

        m = rho_tm[1:].reshape(3,nvar,ngrids)
        s = rho[1].reshape(nvar,ngrids,nsg)
        omega = m / (np.linalg.norm(m, axis=0) + 1e-200)

        exc += np.einsum('xgo,xgo->go', vxc[1], s)
        exc_eff += np.einsum('go,o->g', exc, p_weights)

        if deriv > 0:
            fxc = fxc.reshape(2,nvar,2,nvar,ngrids,nsg)
            vxc[1] *= 2
            vxc += np.einsum('xbygo,xgo->bygo', fxc[1], s)
            vxc_eff += _affine_transform(vxc, omega, p_sgridz, weights)

        if deriv > 1:
            kxc = kxc.reshape(2,nvar,2,nvar,2,nvar,ngrids,nsg)
            fxc[1,:,1] *= 3
            fxc[:,:,1] *= 2
            fxc[1,:,:] *= 2
            fxc += np.einsum('xbyczgo,xgo->byczgo', kxc[1], s)
            fxc_eff += _affine_transform(fxc, omega, p_sgridz, weights)

        if deriv > 2:
            raise NotImplementedError

    return exc_eff, vxc_eff, fxc_eff, kxc_eff

def _make_paxis_samples(spin_samples):
    '''Returns z=cos(theta) and weights (normalized to 1) on z between the range [-1, 1]'''
    rt, wt = roots_legendre(spin_samples)
    wt *= .5  # normalized to 1
    return rt, wt

def _affine_transform(ts, omega, gridz, weights):
    # ts = [t, 0, 0, s] -> [t, s*omega_x, s*omega_y, s*omega_z]
    nvar = ts.shape[1]
    ngrids = ts.shape[-2]
    if ts.ndim == 4:
        # For vec{e} on unit sphere
        # vec{e} = [sin(theta)cos(varphi), sin(theta)sin(varphi), cos(theta)
        # 1/2pi int_0^{2\pi} \vec{e} d\varphi = [0, 0, cos(theta)]
        ts1d = (ts[1]*gridz).dot(weights)
        tm = np.empty((4,nvar,ngrids))
        tm[0] = ts[0].dot(weights)
        tm[1:] = np.einsum('xg,rxg->rxg', ts1d, omega)
    elif ts.ndim == 6:
        # gridzz, and gridxx are derived from integration of m*m on on the unit sphere
        c = _rotation_matrix(omega)
        gridzz = gridz**2
        gridxx = (1 - gridz**2) * .5
        nvar = ts.shape[1]
        ts1d = (ts[0,:,1]*gridz).dot(weights)
        ts2d_xx = (ts[1,:,1] * gridxx).dot(weights)
        ts2d_zz = (ts[1,:,1] * gridzz).dot(weights)
        ts2d = np.array([ts2d_xx, ts2d_xx, ts2d_zz])

        tm = np.empty((4,nvar, 4,nvar, ngrids))
        tm[0,:,0 ] = ts[0,:,0].dot(weights)
        ts1d = np.einsum('xyg,rxg->rxyg', ts1d, omega)
        tm[1:,:,0] = ts1d
        tm[0,:,1:] = ts1d.transpose(1,0,2,3)
        ts2d = np.einsum('rxyg,rsxg->rxsyg', ts2d, c)
        tm[1:,:,1:] = np.einsum('txsyg,trxg->rxsyg', ts2d, c)
    elif ts.ndim == 8:
        # gridxx = (1 - gridz**2) * .5
        # gridzz = gridz**2
        # gridxxz = gridxx * gridz
        # gridzzz = gridzz * gridz
        # ts2d_xx = (ts[0,:,1,:,1] * gridxx).dot(weights)
        # ts2d_zz = (ts[0,:,1,:,1] * gridzz).dot(weights)
        # ts3d_xxz = (ts[1,:,1,:,1] * gridxxz).dot(weights)
        # ts3d_zzz = (ts[1,:,1,:,1] * gridzzz).dot(weights)
        # ts1d = (ts[0,:,0,:,1]*gridz).dot(weights)
        # ts2d = np.array([ts2d_xx, ts2d_xx, ts2d_zz])
        # tm = np.empty((4,nvar, 4,nvar, 4,nvar, ngrids))
        # tm[0,:,0,:,0] = ts[0,:,0,:,0].dot(weights)
        # ts1d = np.einsum('xyzg,rxg->rxyzg', ts1d, omega)
        # tm[1:,:,0,:,0] = ts1d
        # tm[0,:,1:,:,0] = ts1d.transpose(1,0,2,3,4)
        # tm[0,:,0,:,1:] = ts1d.transpose(1,2,0,3,4)
        raise NotImplementedError
    else:
        raise NotImplementedError(f'ts dimension {ts.ndim}')
    return tm

def _rotation_matrix(omega):
    ox, oy, oz = omega
    c = oz
    s = (1-oz**2)**.5
    p = (1-c) / (s+1e-100)**2
    r00 = oz + p * oy**2
    r11 = oz + p * ox**2
    r10 = -p * ox * oy

    r = np.array([
        [r00, r10,-ox],
        [r10, r11,-oy],
        [ox , oy , oz],
    ])
    return r

def _project_spin_paxis(rho_tm, sgridz):
    '''Spin on the principal axis'''
    rho = rho_tm[0]
    m = rho_tm[1:]
    nsg = sgridz.shape[0]
    ngrids = rho.shape[-1]
    s = np.linalg.norm(m, axis=0)
    if rho_tm.ndim == 2:
        rho_ts = np.empty((2, ngrids, nsg))
        rho_ts[0] = rho[:,None]
        rho_ts[1] = s[:,None] * sgridz
        rho_ts = rho_ts.reshape(2,ngrids*nsg)
    else:
        nvar = rho_tm.shape[1]
        rho_ts = np.empty((2, nvar, ngrids, nsg))
        rho_ts[0] = rho[:,:,None]
        rho_ts[1] = s[:,:,None] * sgridz
        rho_ts = rho_ts.reshape(2, nvar, ngrids*nsg)
    return rho_ts
