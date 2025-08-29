import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from typing import Tuple


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

        grid_data = griddata(
            points=(orig_x, orig_y),   # original coords
            values=orig_data,        # values
            xi=(compo_X, compo_Y),
            method=method  # 'linear', 'nearest', or 'cubic'
        )

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