from typing import Tuple
import xarray as xr
import numpy as np

KM_PER_DEGREE_EQ = 111.195  # km per degree of latitude/longitude at the equator


def get_coords_orig(dset: xr.Dataset) -> xr.Dataset:
    coords_orig = dset[['lat', 'lon', 'cell_area']].drop_vars('height_2')
    coords_orig = coords_orig.assign_coords(
        {'component': ('component', ['lat', 'lon'])}
        )
    coords_orig['origin'] = xr.DataArray(
        coords = {'component': (('component',), ['lat', 'lon'])},
        data = [
            coords_orig['lat'].isel(lat=0).values,
            coords_orig['lon'].isel(lon=0).values,
        ]
    )
    coords_orig['dsphere'] = (
        'component',
        np.array([
            coords_orig['lat'].diff('lat').mean().values,
            coords_orig['lon'].diff('lon').mean().values,
            ]),
        )

    return coords_orig


# ------------------------------------------------------------------------------
# Coordinate system transformation
# --------------------------------
def add_featcen_coords(
        coords_sphere: xr.Dataset,
        feature_data_in: list[xr.Dataset],
        feature_props: xr.Dataset,
        feature_ellipse: dict,
        ) -> list[xr.Dataset]:
    feature_data_out = []
    for idx, _ in enumerate(feature_props['feature_id']):
        data = feature_data_in[idx]
        props = feature_props.isel(feature=idx)
        ellipse = feature_ellipse['featcen_cart'].isel(feature=idx)

        data = featcen_sphere_coords(coords_sphere, data, props['centroid_idx'])
        data = featcen_cart_coords(data)
        data = rota_featcen_cart_coords(data, ellipse['polar_angle_rad'])
        feature_data_out.append(data)

    return feature_data_out


def featcen_sphere_coords(
        coords_sphere: xr.Dataset,
        data: xr.Dataset,
        centroid_idx: xr.Dataset,
        ) -> xr.Dataset:
    centroid_coords = _get_centroid_coords(coords_sphere, centroid_idx)
    centroid_lat = centroid_coords.sel(component='lat').values
    centroid_lon = centroid_coords.sel(component='lon').values
    
    if data['lon'][0] > data['lon'][-1]:
        data['lon'] = _adjust_lon_jump(data['lon'], centroid_lon)
    featcen_lat = data['lat'] - centroid_lat
    featcen_lon = data['lon'] - centroid_lon
    data = data.assign_coords({
        'featcen_lat': ('lat', featcen_lat.data),
        'featcen_lon': ('lon', featcen_lon.data),
    })
    return data


def featcen_cart_coords(data: xr.Dataset) -> xr.Dataset:
    dx = KM_PER_DEGREE_EQ * np.cos(np.deg2rad(data['lat']))
    dy = KM_PER_DEGREE_EQ

    featcen_x = data['featcen_lon'] * dx
    featcen_y = data['featcen_lat'] * dy
    featcen_y = featcen_y.broadcast_like(featcen_x)
    data['featcen_x'] = featcen_x.transpose()
    data['featcen_y'] = featcen_y.transpose()
    return data
    

def rota_featcen_cart_coords(
        data: xr.Dataset,
        rot_angle_rad: xr.DataArray,
        ) -> list[xr.Dataset]:
    data['rota_featcen_x'], data['rota_featcen_y'] = \
        _calc_rota_featcen_cart_coords(
            data['featcen_x'], data['featcen_y'], rot_angle_rad,
            )
    return data


# ------------------------------------------------------------------------------
# Helper functions
# ----------------
def _calc_rota_featcen_cart_coords(
        x: xr.DataArray,
        y: xr.DataArray,
        rot_angle_rad: xr.DataArray,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    x_new = x * np.cos(rot_angle_rad) + y * np.sin(rot_angle_rad)
    y_new = -x * np.sin(rot_angle_rad) + y * np.cos(rot_angle_rad)
    return x_new, y_new


def _adjust_lon_jump(
        data_lon: xr.DataArray,
        centroid_lon: float,
        ) -> xr.DataArray:
    if centroid_lon > 0:
        data_lon = xr.where(data_lon < 0, data_lon + 360., data_lon)
    elif centroid_lon < 0:
        data_lon = xr.where(data_lon > 0, data_lon - 360., data_lon)
    else:
        raise ValueError("Please doublecheck what's going on here!")
    return data_lon


def _get_centroid_coords(
        coords_sphere: xr.Dataset,
        centroid_idx: xr.DataArray
        ) -> xr.DataArray:
    return centroid_idx * coords_sphere['dsphere'] + coords_sphere['origin']