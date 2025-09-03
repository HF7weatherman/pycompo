import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from typing import Tuple


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