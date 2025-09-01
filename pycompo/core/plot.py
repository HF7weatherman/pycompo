import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Tuple
from matplotlib.patches import Ellipse
from matplotlib.axes import Axes

import pycompo.core.coord as pccoord

COMPO_PLOT_RANGE = {
    'ts_ano': [-0.3, 0.3],
    'dts_ano_dx': [-0.000003, 0.000003],
    'dts_ano_dy': [-0.000003, 0.000003],
    'ts_ano_laplacian': [-0.0000000001, 0.0000000001],
    'pr_ano': [-3, 3],
    'hfls_ano': [-6, 6],
    'hfss_ano': [-1.5, 1.5],
    'prw_ano': [-0.4, 0.4],
    'sfcwind_ano': [-0.15, 0.15],
}

CLABEL = {
    'ts_ano': "ts_ano / K",
    'dts_ano_dx': "dts_ano_dx / K m-1",
    'dts_ano_dy': "dts_ano_dy / K m-1",
    'ts_ano_laplacian': "ts_ano_laplacian / K m-2",
    'pr_ano': "pr_ano / mm day-1",
    'hfls_ano': "hfls_ano / W m-2",
    'hfss_ano': "hfss_ano / W m-2",
    'prw_ano': "prw_ano / mm",
    'sfcwind_ano': "sfcwind_ano / m s-1",
}


def plot_preprocessing_overview_map(
        dset: xr.Dataset,
        var: str,
        threshold: float,
        ) -> None:
    _, axs = plt.subplots(2, 2, figsize=(10, 8))

    ano_range = (-3 * threshold, 3 * threshold)
    ano_cmap = 'RdBu_r'

    # filled contour plots
    dset['ts'].plot(ax=axs[0, 0]) # type: ignore
    axs[0, 0].set_title('Original SST')

    dset[f'{var}'].plot(
        ax=axs[0, 1], vmin=ano_range[0], vmax=ano_range[1], cmap=ano_cmap,
        extend='both',
        ) # type: ignore
    axs[0, 1].set_title('Detrended SST')

    dset[f'{var}_bg'].plot(
        ax=axs[1, 0], vmin=ano_range[0], vmax=ano_range[1], cmap=ano_cmap,
        extend='both',
        ) # type: ignore
    axs[1, 0].set_title('Detrended SST background (Lowpass)')

    dset[f'{var}_ano'].plot(
        ax=axs[1, 1], vmin=ano_range[0], vmax=ano_range[1], cmap=ano_cmap,
        extend='both',
        ) # type: ignore
    axs[1, 1].set_title('Detrended SST anomaly')

    # overlay contour lines for anomalies
    for i in range(0, 4):
        axs.ravel()[i].contour(
            dset['lon'], dset['lat'], dset[f'{var}_ano'],
            levels=[threshold], colors='k',
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
    dset = featcen_dset[feature_id]
    winds = winds.isel(feature=feature_id)
    _, axs = plt.subplots(3, 2, figsize=(10, 12))

    clabel = f'{var} / K'

    # Plot data in regular geophysical coordinates
    pl0 = axs[0, 0].pcolormesh(
        dset['lon'], dset['lat'], dset[var], cmap='RdBu_r',
        )
    axs[0, 0].contour(
        dset['lon'], dset['lat'], dset[var], levels=[dT_thresh], colors='gray',
        )
    axs[0, 0].scatter(
        ellipse['sphere']['centroid'].\
            sel(component='lon').isel(feature=feature_id),
        ellipse['sphere']['centroid'].\
            sel(component='lat').isel(feature=feature_id),
        color='gray',
        )
    axs[0, 0].set_title('Regular geophysical coordinate')
    axs[0, 0].set_xlabel('Longitude / $^{\circ}\,$E')
    axs[0, 0].set_ylabel('Latitude / $^{\circ}\,$N')
    plt.colorbar(
        pl0, ax=axs[0, 0], orientation='vertical', extend='both', label=clabel,
        )

    # Plot data in feature-centric geophysical coordinates
    pl1 = axs[0, 1].pcolormesh(
        dset['featcen_lon'], dset['featcen_lat'], dset[var], cmap='RdBu_r',
        )
    axs[0, 1].contour(
        dset['featcen_lon'], dset['featcen_lat'], dset[var],
        levels=[dT_thresh], colors='gray',
        )
    axs[0, 1].scatter(
        ellipse['featcen_sphere']['centroid'].\
            sel(component='lon').isel(feature=feature_id),
        ellipse['featcen_sphere']['centroid'].\
            sel(component='lat').isel(feature=feature_id),
        color='gray',
        )
    axs[0, 1].set_title('Feature-centric geophysical coordinate')
    axs[0, 1].set_xlabel('Feature-centric longitude / $^{\circ}\,$E')
    axs[0, 1].set_ylabel('Feature-centric latitude / $^{\circ}\,$N')
    plt.colorbar(
        pl1, ax=axs[0, 1], orientation='vertical', extend='both', label=clabel,
        )
    
    # Plot data in feature-centric Cartesian coordinates
    pl1 = axs[1, 0].pcolormesh(
        dset['featcen_x'], dset['featcen_y'], dset[var], cmap='RdBu_r',
        )
    axs[1, 0].contour(
        dset['featcen_x'], dset['featcen_y'], dset[var],
        levels=[dT_thresh], colors='gray',
        )
    axs[1, 0].scatter(
        ellipse['featcen_cart']['centroid'].\
            sel(component='x').isel(feature=feature_id),
        ellipse['featcen_cart']['centroid'].\
            sel(component='y').isel(feature=feature_id),
        color='gray',
        )
    axs[1, 0].set_title('Feature-centric Cartesian coordinate')
    axs[1, 0].set_xlabel('Feature-centric distance / km')
    axs[1, 0].set_ylabel('Feature-centric distance / km')
    plt.colorbar(
        pl1, ax=axs[1, 0], orientation='vertical', extend='both', label=clabel,
        )
    
    # Plot data in rotated feature-centric Cartesian coordinates
    pl1 = axs[1, 1].pcolormesh(
        dset['rota_featcen_x'], dset['rota_featcen_y'], dset[var],
        cmap='RdBu_r',
        )
    axs[1, 1].contour(
        dset['rota_featcen_x'], dset['rota_featcen_y'], dset[var],
        levels=[dT_thresh], colors='gray',
        )
    axs[1, 1].scatter(
        ellipse['rota_featcen_cart']['centroid'].\
            sel(component='x').isel(feature=feature_id),
        ellipse['rota_featcen_cart']['centroid'].\
            sel(component='y').isel(feature=feature_id),
        color='gray',
        )
    axs[1, 1].set_title('Rotated feature-centric Cartesian coordinate')
    axs[1, 1].set_xlabel('Feature-centric distance / km')
    axs[1, 1].set_ylabel('Feature-centric distance / km')
    plt.colorbar(
        pl1, ax=axs[1, 1], orientation='vertical', extend='both', label=clabel,
        )
    
    # Plot data in normalized rotated feature-centric Cartesian coordinates
    pl1 = axs[2, 0].pcolormesh(
        dset['En_rota_featcen_x'], dset['En_rota_featcen_y'], dset[var],
        cmap='RdBu_r',
        )
    axs[2, 0].contour(
        dset['En_rota_featcen_x'], dset['En_rota_featcen_y'], dset[var],
        levels=[dT_thresh], colors='gray',
        )
    axs[2, 0].scatter([0], [0], color='gray')
    axs[2, 0].set_title('Norm. rotated feature-centric Cartesian coordinate')
    axs[2, 0].set_xlabel('Fractional distance')
    axs[2, 0].set_ylabel('Fractional distance')
    plt.colorbar(
        pl1, ax=axs[2, 0], orientation='vertical', extend='both', label=clabel,
        )

    # Plot data in normalized rotated feature-centric Cartesian coordinates
    pl1 = axs[2, 1].pcolormesh(
        dset['En_rota2_featcen_x'], dset['En_rota2_featcen_y'], dset[var],
        cmap='RdBu_r',
        )
    axs[2, 1].contour(
        dset['En_rota2_featcen_x'], dset['En_rota2_featcen_y'], dset[var],
        levels=[dT_thresh], colors='gray',
        )
    axs[2, 1].scatter([0], [0], color='gray')
    axs[2, 1].set_title('Norm. rotated feature-centric Cartesian coordinate')
    axs[2, 1].set_xlabel('Downwind fractional distance')
    axs[2, 1].set_ylabel('Downwind fractional distance')
    plt.colorbar(
        pl1, ax=axs[2, 1], orientation='vertical', extend='both', label=clabel,
        )

    # Plot grid lines
    _add_grid(axs[0, 1], dset['featcen_lon'], dset['featcen_lat'])
    _add_grid(axs[1, 0], dset['featcen_x'], dset['featcen_y'])
    _add_grid(axs[1, 1], dset['rota_featcen_x'], dset['rota_featcen_y'])
    _add_grid(axs[2, 0], dset['En_rota_featcen_x'], dset['En_rota_featcen_y'])
    _add_grid(axs[2, 1], dset['En_rota2_featcen_x'], dset['En_rota2_featcen_y'])
    
    # Plot feature ellipses/circles
    _plot_feature_ellipse(
        axs[0, 1], ellipse['featcen_sphere'].isel(feature=feature_id)
        )
    _plot_feature_ellipse(
        axs[1, 0], ellipse['featcen_cart'].isel(feature=feature_id)
        )
    _plot_feature_ellipse(
        axs[1, 1], ellipse['rota_featcen_cart'].isel(feature=feature_id)
        )
    _plot_feature_circle(axs[2, 0], (0, 0), 1)
    _plot_feature_circle(axs[2, 1], (0, 0), 1)
    
    # Plot wind features
    axs[0, 0].quiver(
        ellipse['sphere']['centroid'].sel(component='lon').\
            isel(feature=feature_id),
        ellipse['sphere']['centroid'].sel(component='lat').\
            isel(feature=feature_id),
        winds['bg_uas'], winds['bg_vas'], scale=50, zorder=2,
        )
    axs[0, 1].quiver(0, 0, winds['bg_uas'], winds['bg_vas'], scale=50, zorder=2)
    axs[1, 0].quiver(0, 0, winds['bg_uas'], winds['bg_vas'], scale=50, zorder=2)
    
    uas_rota, vas_rota = pccoord._calc_rota_featcen_cart_coords(
        winds['bg_uas'], winds['bg_vas'],
        ellipse['featcen_cart']['polar_angle_rad'].isel(feature=feature_id),
        )
    axs[1, 1].quiver(0, 0, uas_rota, vas_rota, scale=50, zorder=2)
    axs[2, 0].quiver(0, 0, uas_rota, vas_rota, scale=50, zorder=2)
    
    uas_rota2, vas_rota2 = pccoord._calc_rota_featcen_cart_coords(
        winds['bg_uas'], winds['bg_vas'],
        np.arctan2(winds['bg_vas'], winds['bg_uas']),
        )
    axs[2, 1].quiver(0, 0, uas_rota2, vas_rota2, scale=50, zorder=2)

    axs[2, 0].set_xlim([-2.5, 2.5])
    axs[2, 0].set_ylim([-2.5, 2.5])
    axs[2, 1].set_xlim([-2.5, 2.5])
    axs[2, 1].set_ylim([-2.5, 2.5])

    plt.tight_layout()
    for i in range(0, len(axs.ravel())): axs.ravel()[i].set_aspect('equal')
    plt.show()


def plot_composite(compo_data: xr.DataArray):
    var = compo_data.name
    _, axs = plt.subplots(1, 1, figsize=(4, 3))
    pl1 = axs.pcolormesh(
        compo_data['En_rota2_featcen_x'],
        compo_data['En_rota2_featcen_y'],
        compo_data.mean(dim='feature').transpose(),
        cmap="RdBu_r", vmin=COMPO_PLOT_RANGE[var][0],
        vmax=COMPO_PLOT_RANGE[var][1],
    )
    _add_grid(
        axs, compo_data['En_rota2_featcen_x'], compo_data['En_rota2_featcen_y'],
        )
    _plot_feature_circle(axs, (0, 0), 1)
    plt.gca().set_aspect('equal')
    plt.colorbar(pl1, ax=axs, label=CLABEL[var])

    plt.xlabel('Downwind fractional distance')
    plt.ylabel('Crosswind fractional distance')
    
    plt.tight_layout()
    axs.set_aspect('equal')
    plt.show()


# ------------------------------------------------------------------------------
# Helper functions
# ----------------
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
        edgecolor='k', facecolor='none', lw=1.5, ls='-.', zorder=2,
    )
    axis.add_patch(ellipse_patch)
    
    axis.plot(
        [-maj_end.isel(component=1), maj_end.isel(component=1)],
        [-maj_end.isel(component=0), maj_end.isel(component=0)],
        zorder=2,
        )
    axis.plot(
        [-min_end.isel(component=1), min_end.isel(component=1)],
        [-min_end.isel(component=0), min_end.isel(component=0)],
        zorder=2,
        )
    

def _plot_feature_circle(
    axis: Axes,
    center: Tuple[float | int, float | int],
    radius: float | int,
    ) -> None:
    circle = plt.Circle(
        center, radius, fill=False, color='k', ls='-.', lw=1.5, zorder=2,
    )
    axis.add_patch(circle)
    

def _add_grid(
        axis: Axes,
        x_coord: xr.DataArray,
        y_coord: xr.DataArray,
        ) -> None:
    axis.vlines(
        x=0, ymin=y_coord.min(), ymax=y_coord.max(), lw=0.8, ls='--',
        color='gray',
        )
    axis.hlines(
        y=0, xmin=x_coord.min(), xmax=x_coord.max(), lw=0.8, ls='--',
        color='gray',
        )