"""
Adapted from
https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60.
"""
# import holoviews as hv
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy import ndimage
import xarray as xr

# hv.extension('matplotlib')


def _anim_colormap(xr_ds, dvars, deriv=None, **kwargs):
    tpds = xr_ds[dvars]

    if deriv is None:
        clim = (
            tpds
            .to_array()
            .pipe(
                lambda da: (
                    da.quantile(0.005).item(), da.quantile(0.995).item()
                )
            )
        )
        cmap='RdBu'
    elif deriv == 'grad':
        tpds = tpds.pipe(_sobel)
        clim = (0, tpds.to_array().max().item())
        cmap = 'viridis'
    elif deriv == 'lap':
        tpds = tpds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1))
        clim = (
            tpds
            .to_array()
            .pipe(
                lambda da: (
                    da.quantile(0.005).item(), da.quantile(0.995).item()
                )
            )
        )
        cmap='RdGy'
    else:
        raise ValueError(f'unhandled value: `deriv`={deriv}')

    hvds = hv.Dataset(tpds)

    if len(dvars) == 1:
        return (
            hvds
            .to(hv.QuadMesh, ['lon', 'lat'], dvars[0])
            .relabel(dvars[0])
            .options(cmap=cmap, clim=clim, colorbar=True)
        )
    else:
        return hv.Layout([
            (
                hvds
                .to(hv.QuadMesh, ['lon', 'lat'], v)
                .relabel(v)
                .options(cmap=cmap, clim=clim, colorbar=True, axiswise=False)
            )
            for v in dvars
        ]).cols(kwargs.get('cols', 2)).opts(sublabel_format="")


def _sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) / 2
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) / 2
    return np.hypot(dx_ac, dx_al)


def animate_maps(
    xr_ds, dvars, save_location=None, deriv=None, domain=None, time_slice=None,
    **kwargs,
):
    if not isinstance(dvars, list):
        if isinstance(dvars, str):
            dvars = [dvars]
        else:
            raise ValueError('`dvars` must be a string or a list of string')
    if domain:
        xr_ds = xr_ds.sel(domain)
    if time_slice:
        xr_ds = xr_ds.isel(time=time_slice)

    img = _anim_colormap(xr_ds, dvars, deriv, **kwargs)

    if save_location:
        hv.save(img, save_location, fps=4, dpi=125)  # save file
    else:
        hv.output(img, holomap='gif', fps=4, dpi=125)  # display in notebook


if __name__ == '__main__':
    path = './core/UV/james_uv_test/version_1'
    ds = xr.open_dataset(f'{path}/test_res_all.nc')

    dvars = [
        'u_gt', 'u_rec', 'v_gt', 'v_rec', 'ssh_gt', 'ssh_rec', 'sst_feat'
    ]
    output_path = f'{path}/animation.gif'

    animate_maps(ds, dvars, output_path, cols=2)


def plot_psd_score(ds_psd):

    try:
        nb_experiment = len(ds_psd.experiment)
    except:
        nb_experiment = 1

    fig, ax0 =  plt.subplots(1, nb_experiment, sharey=True, figsize=(20, 5))
    #plt.subplots_adjust(right=0.1, left=0.09)
    for exp in range(nb_experiment):
        try:
            ctitle = ds_psd.experiment.values[exp]
        except:
            ctitle = ''

        if nb_experiment > 1:
            ax = ax0[exp]
            data = (ds_psd.isel(experiment=exp).values)
        else:
            ax = ax0
            data = (ds_psd.values)
        ax.invert_yaxis()
        ax.invert_xaxis()
        c1 = ax.contourf(1./(ds_psd.freq_lon), 1./ds_psd.freq_time, data,
                          levels=np.arange(0,1.1, 0.1), cmap='RdYlGn', extend='both')
        ax.set_xlabel('spatial wavelength (degree_lon)', fontweight='bold', fontsize=18)
        ax0[0].set_ylabel('temporal wavelength (days)', fontweight='bold', fontsize=18)
        #plt.xscale('log')
        ax.set_yscale('log')
        ax.grid(linestyle='--', lw=1, color='w')
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(f'PSD-based score ({ctitle})', fontweight='bold', fontsize=18)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        c2 = ax.contour(1./(ds_psd.freq_lon), 1./ds_psd.freq_time, data, levels=[0.5], linewidths=2, colors='k')

        cbar = fig.colorbar(c1, ax=ax, pad=0.01)
        cbar.add_lines(c2)

    bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
    ax0[-1].annotate('Resolved scales',
                    xy=(1.2, 0.8),
                    xycoords='axes fraction',
                    xytext=(1.2, 0.55),
                    bbox=bbox_props,
                    arrowprops=
                        dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left',
                        verticalalignment='center')

    ax0[-1].annotate('UN-resolved scales',
                    xy=(1.2, 0.2),
                    xycoords='axes fraction',
                    xytext=(1.2, 0.45),
                    bbox=bbox_props,
                    arrowprops=
                    dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left',
                        verticalalignment='center')

    plt.show()