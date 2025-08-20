from typing import Tuple
import xarray as xr
import numpy as np

# ------------------------------------------------------------------------------
# Coordinate system transformation
# --------------------------------
def spherical2cartesian_featcen(
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        feature_data_in: list[xr.Dataset],
        feature_props: xr.Dataset,
        ) -> list[xr.Dataset]:
    lat = basic_coords[0]
    lon = basic_coords[1]
    dlat = lat.diff('lat').mean().values
    dlon = lon.diff('lon').mean().values

    feature_data_out = []
    for idx, _ in enumerate(feature_props['feature_id']):
        props = feature_props.isel(feature=idx)
        centroid_lat, centroid_lon = _get_centroid_coords(
            basic_coords, (dlat, dlon), props['centroid_idx']
            )
        data = feature_data_in[idx]
        
        if data['lon'][0] > data['lon'][-1]:
            data['lon'] = _adjust_lon_jump(data['lon'], centroid_lon)
    
        feature_centric_lat = data['lat'] - centroid_lat
        feature_centric_lon = data['lon'] - centroid_lon
        
        feature_centric_x = feature_centric_lon * 111.195 * \
            np.cos(np.deg2rad(data['lat']))
        feature_centric_y = feature_centric_lat * 111.195
        feature_centric_y = feature_centric_y.broadcast_like(feature_centric_x)

        # Add calculated quantities to xr.Dataset that contains the feature data
        data = data.assign_coords({
            'featcen_lat': ('lat', feature_centric_lat.data),
            'featcen_lon': ('lon', feature_centric_lon.data),
        })
        data['featcen_x'] = feature_centric_x.transpose()
        data['featcen_y'] = feature_centric_y.transpose()

        feature_data_out.append(data)

    return feature_data_out


def rota_featcen_cart_coords(
        featcen_data_in: list[xr.Dataset],
        feature_props: xr.Dataset,
        ) -> list[xr.Dataset]:
    featcen_data_out = []
    for idx, _ in enumerate(feature_props['feature_id']):
        props = feature_props.isel(feature=idx)
        data = featcen_data_in[idx]

        data['rota_featcen_x'], data['rota_featcen_y'] = \
            _calc_rota_featcen_cart_coords(
                data['featcen_x'], data['featcen_y'],
                props['polar_angle_rad_idx']
                )
        featcen_data_out.append(data)
    
    return featcen_data_out


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
        data_lon = xr.where(data_lon < 0, data_lon+360., data_lon)
    elif centroid_lon < 0:
        data_lon = xr.where(data_lon > 0, data_lon-360., data_lon)
    else:
        raise ValueError("Please doublecheck what's going on here!")
    return data_lon


def _get_centroid_coords(
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        basic_dcoords: Tuple[xr.DataArray, xr.DataArray],
        centroid_idxs: xr.DataArray
        ) -> Tuple[float, float]:
    lat = basic_coords[0]
    lon = basic_coords[1]
    dlat = basic_dcoords[0]
    dlon = basic_dcoords[1]

    centroid_exact_lat = lat[0].values + dlat*centroid_idxs[0]
    centroid_exact_lon = lon[0].values + dlon*centroid_idxs[1]

    return (float(centroid_exact_lat.values), float(centroid_exact_lon.values))


# ------------------------------------------------------------------------------
# Get coordinates of ellipse in various coordinate systems
# --------------------------------------------------------
def get_ellipse_featcen_spher_coords(
        props: xr.Dataset,
        basic_dcoords: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dlat = basic_dcoords[0]
    dlon = basic_dcoords[1]
    maj_end_idx = props['axis_major_end_idx'].values
    min_end_idx = props['axis_minor_end_idx'].values

    maj_end_spher = (maj_end_idx[0] * dlon, maj_end_idx[1] * dlat)
    min_end_spher = (min_end_idx[0] * dlon, min_end_idx[1] * dlat)
    return (maj_end_spher, min_end_spher)


def get_ellipse_featcen_cart_coords(
        props: xr.Dataset,
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        basic_dcoords: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dlat = basic_dcoords[0]
    dlon = basic_dcoords[1]
    maj_end_idx = props['axis_major_end_idx'].values
    min_end_idx = props['axis_minor_end_idx'].values

    centroid_lat, _ = pccoord._get_centroid_coords(
        basic_coords, basic_dcoords, props['centroid_idx'],
        )
    dx = dlon * 111.195 * np.cos(np.deg2rad(centroid_lat + dlat))
    dy = dlat * 111.195

    maj_end_cart = (maj_end_idx[0] * dx, maj_end_idx[1] * dy)
    min_end_cart = (min_end_idx[0] * dx, min_end_idx[1] * dy)
    return (maj_end_cart, min_end_cart)


def get_ellipse_featcen_rota_cart_coords(
        props: xr.Dataset,
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        basic_dcoords: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    maj_end_cart, min_end_cart = get_ellipse_featcen_cart_coords(
        props, basic_coords, basic_dcoords,
        )

    maj_end_rota_cart = _calc_rota_featcen_cart_coords(
        maj_end_cart[0], maj_end_cart[1], props['polar_angle_rad_idx']
        )
    min_end_rota_cart = _calc_rota_featcen_cart_coords(
        min_end_cart[0], min_end_cart[1], props['polar_angle_rad_idx']
        )
    return (maj_end_rota_cart, min_end_rota_cart)