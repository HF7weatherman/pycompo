import xarray as xr
import numpy as np
from typing import Tuple


def calc_feature_bg_wind(
        feature_props: xr.Dataset,
        feature_centric_data: list[xr.Dataset],
        wind_vars: Tuple[str],
        ) -> xr.Dataset:
    # Calculate mean wind components for cutout region
    for var in wind_vars:
        bg_wind_component = [
            (data[var] * data['cell_area']/data['cell_area'].sum()).sum().values
            for data in feature_centric_data
            ]
        feature_props[f'bg_{var}'] = ('feature', bg_wind_component)

    # Calculate sfcwind speed and direction
    feature_props['bg_sfcwind'] = _calc_sfcwind_speed(
        feature_props[f'bg_{wind_vars[0]}'], feature_props[f'bg_{wind_vars[1]}']
    )
    feature_props['bg_sfcwind_dir'] = _calc_sfcwind_dir(
        feature_props[f'bg_{wind_vars[0]}'], feature_props[f'bg_{wind_vars[1]}']
    )
    return feature_props

    
def _calc_sfcwind_speed(
        u: xr.DataArray | float,
        v: xr.DataArray | float,
        ) -> xr.DataArray | float:
    """Calculate surface wind speed from u and v components."""
    sfcwind_speed = np.sqrt(u**2 + v**2)
    
    if type(sfcwind_speed) == xr.DataArray:
        attrs = {
            'standard_name': "sfcwind",
            'units': "m s-1",
            'long_name': "10m windspeed",
        }
        sfcwind_speed = sfcwind_speed.rename('sfcwind').assign_attrs(attrs)
    return sfcwind_speed


def _calc_sfcwind_dir(
        u: xr.DataArray | float,
        v: xr.DataArray | float,
        unit: str="deg"
        ) -> xr.DataArray | float:
    """Calculate surface wind direction from u and v components."""
    if unit not in ["rad", "deg"]:
        raise ValueError("Unit must be 'rad' or 'deg'.")
    
    if unit == "deg":
        wind_dir = np.arctan2(u, v) * 180. / np.pi + 180.
        wind_dir = xr.where(wind_dir == 360., 0., wind_dir)
    elif unit == "rad":
        wind_dir = np.arctan2(u, v) + np.pi
        wind_dir = xr.where(wind_dir == 2 *np.pi, 0., wind_dir)

    if type(wind_dir) == xr.DataArray:
        attrs = {
            'standard_name': "sfcwind_dir",
            'units': unit,
            'long_name': "10m wind direction",
        }
        wind_dir = wind_dir.rename('sfcwind').assign_attrs(attrs)
        
    return wind_dir