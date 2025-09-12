import xarray as xr
import numpy as np
from typing import Tuple


def calc_feature_bg_wind(
        feature_props: xr.Dataset,
        feature_centric_data: list[xr.Dataset],
        wind_vars: Tuple[str, str],
        calc_sfcwind: bool=False,
        ) -> xr.Dataset:
    # Calculate mean wind components for cutout region
    for var in wind_vars:
        bg_wind_component = [
            (data[var] * data['cell_area']/data['cell_area'].sum()).sum().values
            for data in feature_centric_data
            ]
        feature_props[f'bg_{var}'] = ('feature', bg_wind_component)

    if calc_sfcwind:
        # Calculate sfcwind speed and direction
        feature_props['bg_sfcwind'] = _calc_windspeed(
            feature_props[f'bg_{wind_vars[0]}'],
            feature_props[f'bg_{wind_vars[1]}'],
        )
        feature_props['bg_sfcwind_dir'] = _calc_winddir(
            feature_props[f'bg_{wind_vars[0]}'],
            feature_props[f'bg_{wind_vars[1]}'],
        )
    return feature_props


def add_wind_grads(
        feature_data_in: list,
        feature_props: xr.Dataset,
        feature_var: str,
        ) -> list:
    feature_data_out = []
    for idx, _ in enumerate(feature_props['feature_id']):
        props = feature_props.isel(feature=idx)
        data = feature_data_in[idx]
        wind_grads = _calc_wind_grads(props, data, feature_var)

        data[f'downwind_{feature_var}_ano_grad'] = wind_grads[0]
        data[f'crosswind_{feature_var}_ano_grad'] = wind_grads[1]
        feature_data_out.append(data)

    return feature_data_out


def add_rotate_winds(
        feature_data_in: list,
        feature_props: xr.Dataset,
        ) -> list:
    feature_data_out = []
    for idx, _ in enumerate(feature_props['feature_id']):
        props = feature_props.isel(feature=idx)
        data = feature_data_in[idx]
        rota_winds = _rotate_winds(props, data)

        data[f'downwind_ano'] = rota_winds[0]
        data[f'crosswind_ano'] = rota_winds[1]
        feature_data_out.append(data)

    return feature_data_out

    
def _calc_windspeed(
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


def _calc_winddir(
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


def _calc_wind_grads(
        feature_props: xr.Dataset,
        feature_data: xr.Dataset,
        grad_var: str,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    cos_winddir = feature_props['bg_uas'] / feature_props['bg_sfcwind']
    sin_winddir = feature_props['bg_vas'] / feature_props['bg_sfcwind']
    downwind_sst_grad = \
        cos_winddir * feature_data[f'd{grad_var}_ano_dx'] + \
        sin_winddir * feature_data[f'd{grad_var}_ano_dy']
    crosswind_sst_grad = \
        -sin_winddir * feature_data[f'd{grad_var}_ano_dx'] + \
         cos_winddir * feature_data[f'd{grad_var}_ano_dy']
    
    return (downwind_sst_grad, crosswind_sst_grad)


def _rotate_winds(
        feature_props: xr.Dataset,
        feature_data: xr.Dataset,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    cos_winddir = feature_props['bg_uas'] / feature_props['bg_sfcwind']
    sin_winddir = feature_props['bg_vas'] / feature_props['bg_sfcwind']
    downwind = \
        cos_winddir * feature_data[f'uas_ano'] + \
        sin_winddir * feature_data[f'vas_ano']
    crosswind = \
        -sin_winddir * feature_data[f'uas_ano'] + \
         cos_winddir * feature_data[f'vas_ano']
    
    return (downwind, crosswind)