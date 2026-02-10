import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from pathlib import Path
from typing import Tuple

import pycompo.core.filter as pcfilter
import pycompo.core.utils as pcutil


# ------------------------------------------------------------------------------
# Remapping to shared composite coordinates
# -----------------------------------------
def get_compo_coords_ds(
        feature_data: list[xr.Dataset],
        config: dict,
        ) -> xr.Dataset:
    compo_vars = _get_compo_vars(feature_data[0].data_vars)

    feature_compo_data = {
        var: interpolate2compo_coords(
            feature_data,
            (np.arange(*config['composite']['compo_x']),
             np.arange(*config['composite']['compo_y'])),
            var
            ) for var in compo_vars
        }
    feature_compo_data = _check_feature_id_consistency_across_vars(
        feature_compo_data
        )
    return xr.merge([feature_compo_data[var] for var in compo_vars])


def interpolate2compo_coords(
        feature_data: list[xr.Dataset],
        compo_target_coords: Tuple[np.ndarray, np.ndarray],
        var: str,
        method: str='linear',
        ):
    if "height" in feature_data[0][var].dims:
        return _interpolate2compo_coords_3d(
            feature_data, compo_target_coords, var, method
            )
    else:
        return _interpolate2compo_coords_2d(
            feature_data, compo_target_coords, var, method
            )
    

def _interpolate2compo_coords_2d(
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
                values=orig_data,          # values
                xi=(compo_X, compo_Y),     # target coords
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


def _interpolate2compo_coords_3d(
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
        orig_height = data['height'].data
        try:
            N_height_levels = len(data['height'].values)
            grid_data = np.empty(
                (compo_X.shape[0], compo_Y.shape[1], N_height_levels)
                )
            for h in range(N_height_levels):
                orig_data = data[var].isel(height=h).data.ravel()
                grid_data[:, :, h] = griddata(
                    points=(orig_x, orig_y),   # original coords
                    values=orig_data,          # values
                    xi=(compo_X, compo_Y),     # target coords
                    method=method  # 'linear', 'nearest', or 'cubic'
                    )
        except:
            continue

        # Don't include features that have NaNs within the remapped target data
        if not (~np.isnan(grid_data)).all(): continue
        
        compo_data.append(grid_data.transpose(1, 0, 2))
        feature_ids.append(data['feature_id'].values)

    return xr.DataArray(
        compo_data,
        coords={
            'feature_id': ('feature', feature_ids),
            'En_rota2_featcen_x': ('x', compo_target_coords[0]),
            'En_rota2_featcen_y': ('y', compo_target_coords[1]),
            'height': ('height', orig_height),
            },
        dims=('feature', 'x', 'y', 'height'),
        name=var,
    )


def _get_compo_vars(
        data_vars: list[str],
        ) -> list[str]:
    non_keep_list = [
        'cell_area', 'ts_feature', 'featcen_x', 'featcen_y', 'rota_featcen_x',
        'rota_featcen_y', 'En_rota_featcen_x', 'En_rota_featcen_y',
        'En_rota2_featcen_x', 'En_rota2_featcen_y',
    ]
    return [v for v in data_vars if v not in non_keep_list]
    

def _check_feature_id_consistency_across_vars(
        feature_compo_data: dict[str, xr.Dataset]
        ) -> dict[str, xr.Dataset]:
    N_features = [data.sizes['feature'] for data in feature_compo_data.values()]
    while len(set(N_features)) > 1:
        print("Warning: Different number of features for different variables!")
        feature_ids = {
            var: data['feature_id'].values
            for var, data in feature_compo_data.items()
            }
        min_key = min(feature_ids, key=lambda k: len(feature_ids[k]))
        
        remove_ids = []
        for var, ids in feature_ids.items():
            if var == min_key: continue
            for id in ids:
                if id not in feature_ids[min_key]: remove_ids.append(id)

        for id in set(remove_ids):
            feature_compo_data = {
                var: data.where(data['feature_id'] != id, drop=True)
                for var, data in feature_compo_data.items()
            }

        N_features = [
            data.sizes['feature'] for data in feature_compo_data.values()
            ]

    return feature_compo_data


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
    pr_clim = xr.open_mfdataset(infiles).squeeze()['pr']

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
        if var in [
            'downwind_ts_ano_grad', 'crosswind_ts_ano_grad',
            'downwind_ts_grad', 'crosswind_ts_grad',
            ]:
            data_adjusted[var] = data_adjusted[var] * 1e5
        if var in ['ts_laplacian', 'ts_ano_laplacian']:
            data_adjusted[var] = data_adjusted[var] * 1e10
        if var in ['cllvi_ano', 'clivi_ano']:
            data_adjusted[var] = data_adjusted[var] * 1e3
        if var in ['sfcwind_conv', 'sfcwind_conv_ano']:
            data_adjusted[var] = data_adjusted[var] * 1e5
        if var in ['wa', 'wa_ano']:
            data_adjusted[var] = data_adjusted[var] * 1e3
        if var in ['hus', 'clw', 'cli', 'hus_ano', 'clw_ano', 'cli_ano']:
            data_adjusted[var] = data_adjusted[var] * 1e6
        if var in ['hfls', 'hfss']:
            data_adjusted[var] = data_adjusted[var] * -1
        if var in ['ps']:
            data_adjusted[var] = data_adjusted[var] * 0.01
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