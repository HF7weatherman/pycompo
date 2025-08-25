import numpy as np
import pycompo.ellipse
import xarray as xr
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.axes import Axes

import pycompo.coord as pccoord

def plot_preprocessing_overview_map(
        dset: xr.Dataset,
        var: str,
        dT_thresh: float,
        ano_range: Tuple[float, float],
        ano_cmap: str,
        ) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # filled contour plots
    dset['ts'].plot(ax=axs[0, 0]) # type: ignore
    axs[0, 0].set_title('Original SST')

    dset[f'{var}_detrend'].plot(
        ax=axs[0, 1], vmin=ano_range[0], vmax=ano_range[1], cmap=ano_cmap,
        extend='both',
        ) # type: ignore
    axs[0, 1].set_title('Detrended SST')

    dset[f'{var}_detrend_bg'].plot(
        ax=axs[1, 0], vmin=ano_range[0], vmax=ano_range[1], cmap=ano_cmap,
        extend='both',
        ) # type: ignore
    axs[1, 0].set_title('Detrended SST background (Lowpass)')

    dset[f'{var}_detrend_ano'].plot(
        ax=axs[1, 1], vmin=ano_range[0], vmax=ano_range[1], cmap=ano_cmap,
        extend='both',
        ) # type: ignore
    axs[1, 1].set_title('Detrended SST anomaly')

    # overlay contour lines for anomalies
    for i in range(0, 4):
        axs.ravel()[i].contour(
            dset['lon'], dset['lat'], dset[f'{var}_detrend_ano'],
            levels=[-dT_thresh, dT_thresh], colors='k',
            )

    plt.tight_layout()
    plt.show()


def plot_coord_trafo(
        featcen_dset: list,
        ellipse: dict,
        winds: xr.Dataset,
        feature_id: int,
        var: str,
        dT_thresh: float
        ) -> None:
    _, axs = plt.subplots(2, 2, figsize=(10, 8))

    dset = featcen_dset[feature_id]
    winds = winds.isel(feature=feature_id)

    # Plot data in regular geophysical coordinates
    pl0 = axs[0, 0].pcolormesh(
        dset['lon'], dset['lat'], dset[f'{var}_detrend_ano'],
        cmap='RdBu_r', shading='auto'
        )
    axs[0, 0].contour(
        dset['lon'], dset['lat'], dset[f'{var}_detrend_ano'],
        levels=[-dT_thresh, dT_thresh], colors='k',
        )
    axs[0, 0].scatter(
        ellipse['sphere']['centroid'].\
            sel(component='lon').isel(feature=feature_id),
        ellipse['sphere']['centroid'].\
            sel(component='lat').isel(feature=feature_id),
        color='k'
        )
    
    axs[0, 0].set_title('Regular geophysical coordinate')
    axs[0, 0].set_xlabel('Longitude / $^{\circ}\,$E')
    axs[0, 0].set_ylabel('Latitude / $^{\circ}\,$N')
    plt.colorbar(
        pl0, ax=axs[0, 0], orientation='vertical', extend='both',
        label=f'{var}_detrend_ano / K',
        )

    # Plot data in feature-centric geophysical coordinates
    pl1 = axs[0, 1].pcolormesh(
        dset['featcen_lon'], dset['featcen_lat'], dset[f'{var}_detrend_ano'],
        cmap='RdBu_r', shading='auto',
        )
    axs[0, 1].contour(
        dset['featcen_lon'], dset['featcen_lat'], dset[f'{var}_detrend_ano'],
        levels=[-dT_thresh, dT_thresh], colors='k',
        )
    axs[0, 1].scatter(
        ellipse['featcen_sphere']['centroid'].\
            sel(component='lon').isel(feature=feature_id),
        ellipse['featcen_sphere']['centroid'].\
            sel(component='lat').isel(feature=feature_id),
        color='k'
        )

    axs[0, 1].set_title('Feature-centric geophysical coordinate')
    axs[0, 1].set_xlabel('Feature-centric longitude / $^{\circ}\,$E')
    axs[0, 1].set_ylabel('Feature-centric latitude / $^{\circ}\,$N')
    plt.colorbar(
        pl1, ax=axs[0, 1], orientation='vertical', extend='both',
        label=f'{var}_detrend_ano / K',
        )
    
    # Plot data in feature-centric Cartesian coordinates
    pl1 = axs[1, 0].pcolormesh(
        dset['featcen_x'], dset['featcen_y'], dset[f'{var}_detrend_ano'],
        cmap='RdBu_r', shading='auto'
        )
    axs[1, 0].contour(
        dset['featcen_x'], dset['featcen_y'], dset[f'{var}_detrend_ano'],
        levels=[-dT_thresh, dT_thresh], colors='k',
        )
    axs[1, 0].scatter(
        ellipse['featcen_cart']['centroid'].\
            sel(component='x').isel(feature=feature_id),
        ellipse['featcen_cart']['centroid'].\
            sel(component='y').isel(feature=feature_id),
        color='k'
        )

    axs[1, 0].set_title('Feature-centric Cartesian coordinate')
    axs[1, 0].set_xlabel('Feature-centric distance / km')
    axs[1, 0].set_ylabel('Feature-centric distance / km')
    plt.colorbar(
        pl1, ax=axs[1, 0], orientation='vertical', extend='both',
        label=f'{var}_detrend_ano / K',
        )
    
    # Plot data in rotated feature-centric Cartesian coordinates
    pl1 = axs[1, 1].pcolormesh(
        dset['rota_featcen_x'], dset['rota_featcen_y'],
        dset[f'{var}_detrend_ano'],
        cmap='RdBu_r', shading='auto'
        )
    axs[1, 1].contour(
        dset['rota_featcen_x'], dset['rota_featcen_y'],
        dset[f'{var}_detrend_ano'],
        levels=[-dT_thresh, dT_thresh], colors='k',
        )
    axs[1, 1].scatter(
        ellipse['rota_featcen_cart']['centroid'].\
            sel(component='x').isel(feature=feature_id),
        ellipse['rota_featcen_cart']['centroid'].\
            sel(component='y').isel(feature=feature_id),
        color='k'
        )

    axs[1, 1].set_title('Rotated feature-centric Cartesian coordinate')
    axs[1, 1].set_xlabel('Feature-centric distance / km')
    axs[1, 1].set_ylabel('Feature-centric distance / km')
    plt.colorbar(
        pl1, ax=axs[1, 1], orientation='vertical', extend='both',
        label=f'{var}_detrend_ano / K',
        )

    # Plot grid lines
    ymin = dset['featcen_lat'].min()
    ymax = dset['featcen_lat'].max()
    xmin = dset['featcen_lon'].min()
    xmax = dset['featcen_lon'].max()
    axs[0, 1].vlines(x=0, ymin=ymin, ymax=ymax, lw=0.8, ls='--', color='gray')
    axs[0, 1].hlines(y=0, xmin=xmin, xmax=xmax, lw=0.8, ls='--', color='gray')

    ymin = dset['featcen_y'].min()
    ymax = dset['featcen_y'].max()
    xmin = dset['featcen_x'].min()
    xmax = dset['featcen_x'].max()
    axs[1, 0].vlines(x=0, ymin=ymin, ymax=ymax, lw=0.8, ls='--', color='gray')
    axs[1, 0].hlines(y=0, xmin=xmin, xmax=xmax, lw=0.8, ls='--', color='gray')

    ymin = dset['rota_featcen_y'].min()
    ymax = dset['rota_featcen_y'].max()
    xmin = dset['rota_featcen_x'].min()
    xmax = dset['rota_featcen_x'].max()
    axs[1, 1].vlines(x=0, ymin=ymin, ymax=ymax, lw=0.8, ls='--', color='gray')
    axs[1, 1].hlines(y=0, xmin=xmin, xmax=xmax, lw=0.8, ls='--', color='gray')
    
    # Plot ellipse features
    _plot_feature_ellipse(
        axs[0, 1], ellipse['featcen_sphere'].isel(feature=feature_id)
        )
    _plot_feature_ellipse(
        axs[1, 0], ellipse['featcen_cart'].isel(feature=feature_id)
        )
    _plot_feature_ellipse(
        axs[1, 1], ellipse['rota_featcen_cart'].isel(feature=feature_id)
        )
    
    # Plot wind features
    axs[0, 0].quiver(
        ellipse['sphere']['centroid'].sel(component='lon').isel(feature=feature_id),
        ellipse['sphere']['centroid'].sel(component='lat').isel(feature=feature_id),
        winds['bg_uas'], winds['bg_vas'], scale=50,
        )
    axs[0, 1].quiver(0, 0, winds['bg_uas'], winds['bg_vas'], scale=50)
    axs[1, 0].quiver(0, 0, winds['bg_uas'], winds['bg_vas'], scale=50)
    
    uas_rota, vas_rota = pccoord._calc_rota_featcen_cart_coords(
        winds['bg_uas'], winds['bg_vas'],
        ellipse['featcen_cart']['polar_angle_rad'].isel(feature=feature_id),
        )
    axs[1, 1].quiver(np.array([0]), np.array([0]), uas_rota, vas_rota, scale=50)

    plt.tight_layout()
    for i in range(0, len(axs.ravel())): axs.ravel()[i].set_aspect('equal')
    plt.show()


def _plot_feature_ellipse(
        axis: Axes,
        ellipse: xr.Dataset
        ) -> None:
    maj_end = ellipse['maj_end']
    min_end = ellipse['min_end']
    polar_angle_rad = ellipse['polar_angle_rad']

    ellipse_patch = Ellipse(
        xy=(0, 0),
        width=np.sqrt((2 * maj_end[0]) ** 2 + (2 * maj_end[1]) ** 2),
        height=np.sqrt((2 * min_end[0]) ** 2 + (2 * min_end[1]) ** 2),
        angle=np.rad2deg(polar_angle_rad),
        edgecolor='green', facecolor='none', lw=1, ls='-.',
    )
    axis.add_patch(ellipse_patch)
    
    axis.plot(
        [-maj_end.isel(component=1), maj_end.isel(component=1)],
        [-maj_end.isel(component=0), maj_end.isel(component=0)],
        )
    axis.plot(
        [-min_end.isel(component=1), min_end.isel(component=1)],
        [-min_end.isel(component=0), min_end.isel(component=0)],
        )