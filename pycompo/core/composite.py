import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from pathlib import Path
from typing import Tuple

import pycompo.core.filter as pcfilter
import pycompo.core.utils as pcutil


# ------------------------------------------------------------------------------
# Rempaping to shared composite coordinates
# -----------------------------------------
def get_compo_coords_ds(
        feature_data: list[xr.Dataset],
        feature_var: str,
        config: dict,
        ) -> xr.Dataset:
    compo_vars = _get_compo_vars(feature_var, config['data']['study_vars'])
    feature_compo_data = {
        var: interpolate2compo_coords(
            feature_data,
            (np.arange(*config['composite']['compo_x']),
             np.arange(*config['composite']['compo_y'])),
            var
            ) for var in compo_vars
        }
    return xr.merge([feature_compo_data[var] for var in compo_vars])


def interpolate2compo_coords(
        feature_data: list[xr.Dataset],
        compo_target_coords: Tuple[np.ndarray, np.ndarray],
        var: str,
        method: str='linear',
        ):
    compo_X, compo_Y = np.meshgrid(
        compo_target_coords[0], compo_target_coords[1],
        )

    compo_data = []
    feature_ids = []
    for data in feature_data:
        orig_x = data['En_rota2_featcen_x'].data.ravel()
        orig_y = data['En_rota2_featcen_y'].data.ravel()
        orig_data = data[var].data.ravel()
        
        try:
            grid_data = griddata(
                points=(orig_x, orig_y),   # original coords
                values=orig_data,        # values
                xi=(compo_X, compo_Y),
                method=method  # 'linear', 'nearest', or 'cubic'
            )
        except:
            continue

        # Don't include features that have NaNs within the remapped target data
        if not (~np.isnan(grid_data)).all(): continue
        
        compo_data.append(grid_data.transpose())
        feature_ids.append(data['feature_id'].values)

    return xr.DataArray(
        compo_data,
        coords={
            'feature_id': ('feature', feature_ids),
            'En_rota2_featcen_x': ('x', compo_target_coords[0]),
            'En_rota2_featcen_y': ('y', compo_target_coords[1]),
            },
        dims=('feature', 'x', 'y'),
        name=var,
    )


def _get_compo_vars(
        feature_var: str,
        study_vars: list[str],
        ) -> list[str]:
    feature_var_modes = [
        f"{feature_var}_ano",
        f"{feature_var}_ano_laplacian",
        f"downwind_{feature_var}_ano_grad",
        f"crosswind_{feature_var}_ano_grad",
        ]
    return feature_var_modes + [f"{var}_ano" for var in study_vars]


# ------------------------------------------------------------------------------
# Sampling on composite data
# --------------------------
def sample_features_geomask(
        features: xr.Dataset,
        mask: xr.DataArray,
        ) -> xr.Dataset:
    keep_idx = []
    for idx, _ in enumerate(features['feature_id']):
        feature = features.isel(feature=idx)
        if mask.sel(time=feature['time']).isel(
            lat=int(feature['centroid_idx'].sel(component='lat')),
            lon=int(feature['centroid_idx'].sel(component='lon')),
            ).values:
            keep_idx.append(int(feature['feature_id'].values))
    return features.where(features['feature_id'].isin(keep_idx), drop=True)



def get_rainbelt(
        analysis_times: list,
        config: dict,
        quantile: float=0.8
        ) -> xr.DataArray:
    # read in data
    inpath = Path(config['data']['inpaths']['pr'])
    in_pattern = f"{config['exp']}_tropical_pr_*.nc"
    infiles = sorted([str(f) for f in inpath.rglob(in_pattern)])
    pr_clim = xr.open_mfdataset(infiles, parallel=True).squeeze()['pr']

    # build climatology
    if config['composite']['rainbelt_subsampling']['mode'] == 'roll_avg_clim':
        pr_clim = pcfilter.build_hourly_climatology(
            pr_clim, clim_baseyear=str(config['detrend']['clim_baseyear'])
            )
        pr_clim = pcutil.circ_roll_avg(
            pr_clim, config['detrend']['clim_avg_days'], config['data']['spd'],
            )
        
    elif config['composite']['rainbelt_subsampling']['mode'] == 'roll_avg':
        pr_clim = pcutil.roll_avg(
            pr_clim, config['detrend']['clim_avg_days'], config['data']['spd'],
            )

    else:
        raise ValueError(
            "Please provide a valid mode for 'rainbelt_subsampling'! " +
            "Valid modes are 'roll_avg_clim' and 'roll_avg'."
            )
    
    # build rainbelt from climatology
    pr_clim = pr_clim.sel(time=slice(analysis_times[0], analysis_times[-1]))
    pr_clim = pr_clim.persist()
    pr_quantile = pr_clim.quantile(quantile, dim=['lat', 'lon'])
    pr_clim = pr_clim.sel(lat=slice(*config['lat_range']), drop=True)
    return xr.where(pr_clim >= pr_quantile, True, False)


def adjust_units(
    data: xr.Dataset,
    vars: list,
    ) -> xr.Dataset:
    data_adjusted = data.copy()
    for var in vars:
        if var in ['downwind_ts_ano_grad', 'crosswind_ts_ano_grad']:
            data_adjusted[var] = data_adjusted[var] * 1e5
        if var in ['ts_ano_laplacian']:
            data_adjusted[var] = data_adjusted[var] * 1e10
        if var in ['cllvi_ano', 'clivi_ano']:
            data_adjusted[var] = data_adjusted[var] * 1e3
        if var in ['sfcwind_conv_ano']:
            data_adjusted[var] = data_adjusted[var] * 1e5
    return data_adjusted


def get_quartile_compos_per_ts(
        features: xr.Dataset,
        feature_props_quartiles: xr.DataArray,
        var: str,
        ) -> dict:
    q25 = feature_props_quartiles.sel(quantile=0.25)
    q50 = feature_props_quartiles.sel(quantile=0.5)
    q75 = feature_props_quartiles.sel(quantile=0.75)

    return {
        "q1": features.where(
            features[var] <= q25, drop=True
            ).mean(dim='feature'),
        "q2": features.where(
            (features[var] > q25) & (features[var] <= q50), drop=True,
            ).mean(dim='feature'),
        "q3": features.where(
            (features[var] > q50) & (features[var] <= q75), drop=True,
            ).mean(dim='feature'),
        "q4": features.where(
            features[var] > q75, drop=True,
            ).mean(dim='feature'),
        }


def get_full_quartile_compos(
        quartile_compo: list[dict],
        ) -> xr.Dataset:
    quartile_compo_dict = {
        key: [d[key] for d in quartile_compo] for key in quartile_compo[0]
        }
    for key, data in quartile_compo_dict.items():
        quartile_compo_dict[key] = \
            xr.concat(data, dim='month').mean(dim='month')
        
    return xr.concat(
        quartile_compo_dict.values(),
        dim='quartile', coords='minimal', compat='override',
        ).drop(['quantile']).assign_coords(quartile=[1, 2, 3, 4])