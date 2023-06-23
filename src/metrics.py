"""
This module provides the metrics used in the evaluation of the
reconstruction.

Adapted from
https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import xrft

_PRECISION = 5  # Second argument of np.round


def psd_based_scores(da_rec, da_ref):
    # Compute error = rec - ref
    err = (da_rec - da_ref)
    err = err.chunk({
        'lat': 1, 'time': err['time'].size, 'lon': err['lon'].size,
    })

    # Rechunk ref
    signal = da_ref.chunk({
        'lat': 1, 'time': da_ref['time'].size, 'lon': da_ref['lon'].size,
    })

    # In days units
    err['time'] = (err.time - err.time[0]) / np.timedelta64(1, 'D')
    signal['time'] = (signal.time - signal.time[0]) / np.timedelta64(1, 'D')

    # Compute PSD_err and PSD_signal
    psd_err = xrft.power_spectrum(
        err, dim=['time', 'lon'], detrend='constant', window='hann',
    ).compute()
    psd_signal = xrft.power_spectrum(
        signal, dim=['time', 'lon'], detrend='constant', window='hann',
    ).compute()

    # Averaged over latitude
    mean_psd_signal = (
        psd_signal
        .mean(dim='lat')
        .where(
            (psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True,
        )
    )
    mean_psd_err = (
        psd_err
        .mean(dim='lat')
        .where(
            (psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True,
        )
    )

    # return PSD-based score
    psd_based_score = (1. - mean_psd_err/mean_psd_signal)

    # Find the key metrics: shortest temporal & spatial scales resolved
    # based on the 0.5 contour criterion of the PSD_score
    level = [.5]
    cs = plt.contour(
        1. / psd_based_score.freq_lon.values,
        1. / psd_based_score.freq_time.values,
        psd_based_score,
        level,
    )

    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)

    # psd_da = (1. - mean_psd_err/mean_psd_signal)
    # psd_da.name = 'psd_score'

    return (
        1. - mean_psd_err/mean_psd_signal,  # psd_da.to_dataset(),
        np.round(shortest_spatial_wavelength_resolved, _PRECISION),
        np.round(shortest_temporal_wavelength_resolved, _PRECISION),
    )


def rmse_based_scores(da_rec, da_ref):
    # RMSE(t) based score
    _rmse = (((da_rec - da_ref)**2).mean(dim=('lon', 'lat')))**.5
    _rms = ((da_ref**2).mean(dim=('lon', 'lat')))**.5
    rmse_t = 1. - _rmse / _rms
    rmse_t = rmse_t.rename('rmse_t')

    # RMSE(x, y) based score
    rmse_xy = (((da_rec - da_ref)**2).mean(dim=('time')))**.5
    rmse_xy = rmse_xy.rename('rmse_xy')

    # Show leaderboard SSH-RMSE metric (spatially and time averaged
    # normalized RMSE)
    _rmse = (((da_rec - da_ref)**2).mean())**.5
    _rms = ((da_ref**2).mean())**.5
    leaderboard_rmse = 1. - _rmse / _rms
    mu = np.round(leaderboard_rmse.values, _PRECISION)

    # Temporal stability of the error (sigma)
    reconstruction_error_stability_metric = rmse_t.std().values
    sigma = np.round(reconstruction_error_stability_metric, _PRECISION)

    # return rmse_t, rmse_xy, mu, sigma
    return rmse_xy, mu, sigma


def var_based_scores(da_rec, da_ref):
    """
    Return tau_uv, tau_vort, tau_div and tau_strain.
    """
    lat_rad = np.radians(da_rec.lat).values
    lon_rad = np.radians(da_rec.lon).values
    div_gt, curl_gt, strain_gt = _compute_div_curl_strain_with_lat_lon(
        da_ref.u, da_ref.v, lat_rad, lon_rad, sigma=1.,
    )
    div_uv_rec, curl_uv_rec, strain_uv_rec = _compute_div_curl_strain_with_lat_lon(
        da_rec.u, da_rec.v, lat_rad, lon_rad, sigma=1.,
    )

    # Compute tau_uv
    tau_uv = _var_mse_uv(da_rec.u, da_rec.v, da_ref.u, da_ref.v)

    # Compute tau_vort
    tau_vort = _compute_var_exp(curl_gt, curl_uv_rec)

    # Compute tau_div
    tau_div = _compute_var_exp(div_gt, div_uv_rec)

    # Compute tau_strain
    tau_strain = _compute_var_exp(strain_gt, strain_uv_rec)

    return (
        np.round(tau_uv, _PRECISION),
        np.round(tau_vort, _PRECISION),
        np.round(tau_div, _PRECISION),
        np.round(tau_strain, _PRECISION),
    )


def _var_mse_uv(u_rec, v_rec, u_ref, v_ref, dw=3):
    """
    Return tau_uv.
    """
    if dw == 0:
        mse_uv = np.nanmean((u_ref - u_rec)**2 + (v_ref - v_rec)**2)
        var_uv = np.nanmean(u_ref**2 + v_ref**2)
    else:
        _u = u_rec[:, dw:u_rec.shape[1]-dw, dw:u_rec.shape[2]-dw]
        _v = v_rec[:, dw:u_rec.shape[1]-dw, dw:u_rec.shape[2]-dw]
        _u_gt = u_ref[:, dw:u_rec.shape[1]-dw, dw:u_rec.shape[2]-dw]
        _v_gt = v_ref[:, dw:u_rec.shape[1]-dw, dw:u_rec.shape[2]-dw]

        mse_uv = np.nanmean((_u_gt - _u)**2 + (_v_gt - _v)**2)
        var_uv = np.nanmean((_u_gt)**2 + (_v_gt)**2)
    var_mse_uv = 100. * (1. - mse_uv / var_uv)

    return var_mse_uv


def _compute_var_exp(x, y, dw=3):
    """
    tau_div needs this.
    """
    if dw == 0:
        mse = np.nanmean((x - y)**2)
        var = np.nanvar(x)
    else:
        _x = x[:, dw:x.shape[1]-dw, dw:x.shape[2]-dw]
        _y = y[:, dw:x.shape[1]-dw, dw:x.shape[2]-dw]
        mse = np.nanmean((_x - _y)**2)
        var = np.nanvar(_x)

    return 100. * (1. - mse / var)


def _compute_div_curl_strain_with_lat_lon(u, v, lat, lon, sigma=1.):
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    # coriolis / lat/lon scaling
    grid_lat = lat.reshape((1, u.shape[1], 1))
    grid_lat = np.tile(grid_lat, (v.shape[0], 1, v.shape[2]))
    grid_lon = lon.reshape((1, 1, v.shape[2]))
    grid_lon = np.tile(grid_lon, (v.shape[0], v.shape[1], 1))

    dx_from_dlon , dy_from_dlat = _compute_dx_dy(grid_lat, grid_lon, dlat, dlon)

    du_dx = _compute_grad(u, 2, sigma=sigma)
    du_dx = du_dx / dx_from_dlon

    dv_dy = _compute_grad(v, 1, sigma=sigma)
    dv_dy = dv_dy / dy_from_dlat

    du_dy = _compute_grad(u, 1, sigma=sigma)
    du_dy = du_dy / dy_from_dlat

    dv_dx = _compute_grad(v, 2, sigma=sigma)
    dv_dx = dv_dx / dx_from_dlon

    strain = np.sqrt((dv_dx + du_dy)**2 + (du_dx - dv_dy)**2)
    curl = du_dy - dv_dx
    div = du_dx + dv_dy

    return div, curl, strain


def _compute_dx_dy(lat, lon, dlat, dlon):
    """
    Compute dx and dt from dlat and dlon.
    """
    def compute_c(lat, lon, dlat, dlon):
        a = np.sin(dlat / 2)**2 + np.cos(lat)**2 * np.sin(dlon / 2)**2
        return 2 * 6.371e6 * np.arctan2(np.sqrt(a), np.sqrt(1. - a))

    dy_from_dlat =  compute_c(lat, lon, dlat, 0.)
    dx_from_dlon =  compute_c(lat, lon, 0., dlon)

    return dx_from_dlon, dy_from_dlat


def _compute_grad(u, axis, alpha=1., sigma=0., _filter='diff-non-centered'):
    """
    If axis == 1, then compute grady
    Otherwise, if axis == 2, then compute gradx
    """
    if sigma > 0.:
        u = ndimage.gaussian_filter(u, sigma=sigma)

    if _filter == 'sobel':
        return alpha * ndimage.sobel(u, axis=axis)
    elif _filter == 'diff-non-centered':
        return alpha * ndimage.convolve1d(u, weights=[.3, .4, -.7], axis=axis)
