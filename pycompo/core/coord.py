from typing import Tuple
import xarray as xr
import numpy as np

from grid_toolbox.spherical_derivatives_latlon import \
    compute_gradient_and_laplacian_on_latlon

KM_PER_DEGREE_EQ = 111.195  # km per degree of latitude/longitude at the equator


# ------------------------------------------------------------------------------
# General coordinate functions
# --------------------------------
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


def calc_sphere_gradient_laplacian(
        dset: xr.DataArray | xr.Dataset,
        var: str,
        ) -> xr.Dataset:
    gradient, laplacian = compute_gradient_and_laplacian_on_latlon(dset[var])
    dset[f'd{var}_dx'] = gradient[0]
    dset[f'd{var}_dy'] = gradient[1]
    dset[f'{var}_laplacian'] = laplacian
    for v in [var, f'd{var}_dx', f'd{var}_dy']:
        dset[v] = dset[v].where(~np.isnan(dset[f'{var}_laplacian']), np.NaN)
    return dset


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
        ellipse_cart = feature_ellipse['featcen_cart'].isel(feature=idx)
        ellipse_rota_cart = feature_ellipse['rota_featcen_cart'].isel(feature=idx)

        data = featcen_sphere_coords(coords_sphere, data, props['centroid_idx'])
        data = featcen_cart_coords(data)
        data = rota_featcen_cart_coords(data, ellipse_cart['polar_angle_rad'])
        data = ellipse_norm_rota_featcen_cart_coords(data, ellipse_rota_cart)
        data = ellipse_norm_rota2_featcen_cart_coords(
            data, props[['bg_uas', 'bg_vas']], ellipse_cart['polar_angle_rad'],
            )
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


def ellipse_norm_rota_featcen_cart_coords(
        data: xr.Dataset,
        ellipse: xr.Dataset,
        ) -> xr.Dataset:
    len_maj = _ax_cart_endpoints2length(ellipse['maj_end'])
    len_min = _ax_cart_endpoints2length(ellipse['min_end'])

    R_featcen_cart = np.sqrt(
        data['rota_featcen_x']**2 + data['rota_featcen_y']**2
        )
    tan_featcen_cart = np.tan(data['rota_featcen_y'] / data['rota_featcen_x'])

    # Determine the boundary point ('bp'), where teh ellipse intersects the ray
    #   towards point (x, y)
    xbp = (len_maj * len_min) / \
        np.sqrt(len_min**2 + len_maj**2 * tan_featcen_cart**2)
    ybp = xbp * tan_featcen_cart
    Rbp = np.sqrt(xbp**2 + ybp**2)

    # Compute normalized radius ('Rn')
    Rn = R_featcen_cart / Rbp
    thp = np.arctan2(data['rota_featcen_y'], data['rota_featcen_x'])
    data['En_rota_featcen_x'] = Rn * np.cos(thp)
    data['En_rota_featcen_y'] = Rn * np.sin(thp)

    return data


def ellipse_norm_rota2_featcen_cart_coords(
        data: xr.Dataset,
        winds: xr.Dataset,
        polar_angle_rad: xr.DataArray,
        ) -> list[xr.Dataset]:
    wind_angle_rad = np.arctan2(winds['bg_vas'], winds['bg_uas'])
    rot_angle_rad = wind_angle_rad - polar_angle_rad
    
    data['En_rota2_featcen_x'], data['En_rota2_featcen_y'] = \
        _calc_rota_featcen_cart_coords(
            data['En_rota_featcen_x'], data['En_rota_featcen_y'],
            rot_angle_rad,
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


def _ax_cart_endpoints2length(endpoints: xr.DataArray) -> np.ndarray:
    return np.sqrt(
        endpoints.sel(component='x')**2 + endpoints.sel(component='y')**2
        )