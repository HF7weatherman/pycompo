from typing import Tuple
import pycompo.coord as pccoord
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
            'feature_centric_lat': ('lat', feature_centric_lat.data),
            'feature_centric_lon': ('lon', feature_centric_lon.data),
        })
        data['feature_centric_x'] = feature_centric_x.transpose()
        data['feature_centric_y'] = feature_centric_y.transpose()

        feature_data_out.append(data)

    return feature_data_out


def rotated_feature_centric_cartesian_coords(
        feature_centric_data_in: list[xr.Dataset],
        feature_props: xr.Dataset,
        ) -> list[xr.Dataset]:
    feature_centric_data_out = []
    for idx, _ in enumerate(feature_props['feature_id']):
        props = feature_props.isel(feature=idx)
        data = feature_centric_data_in[idx]

        rotated_feature_centric_coords = _calc_rotated_feature_centric_coords(
            data['feature_centric_x'], data['feature_centric_y'],
            props['polar_angle_rad']
        )
        data['rotated_feature_centric_x'] = rotated_feature_centric_coords[0]
        data['rotated_feature_centric_y'] = rotated_feature_centric_coords[1]
        feature_centric_data_out.append(data)
    
    return feature_centric_data_out


def _calc_rotated_feature_centric_coords(
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
def get_ellipse_featcen_spherical_coords(
        props: xr.Dataset,
        basic_dcoords: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dlat = basic_dcoords[0]
    dlon = basic_dcoords[1]
    major_end_idx = props['axis_major_end_idx'].values
    minor_end_idx = props['axis_minor_end_idx'].values

    major_end_spherical = (major_end_idx[0] * dlon, major_end_idx[1] * dlat)
    minor_end_spherical = (minor_end_idx[0] * dlon, minor_end_idx[1] * dlat)
    return (major_end_spherical, minor_end_spherical)


def get_ellipse_featcen_cartesian_coords(
        props: xr.Dataset,
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        basic_dcoords: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dlat = basic_dcoords[0]
    dlon = basic_dcoords[1]
    major_end_idx = props['axis_major_end_idx'].values
    minor_end_idx = props['axis_minor_end_idx'].values

    centroid_lat, _ = pccoord._get_centroid_coords(
        basic_coords, basic_dcoords, props['centroid_idx'],
        )
    dx = dlon * 111.195 * np.cos(np.deg2rad(centroid_lat + dlat))
    dy = dlat * 111.195

    major_end_cartesian = (major_end_idx[0] * dx, major_end_idx[1] * dy)
    minor_end_cartesian = (minor_end_idx[0] * dx, minor_end_idx[1] * dy)
    return (major_end_cartesian, minor_end_cartesian)


def get_ellipse_featcen_rotated_cartesian_coords(
        props: xr.Dataset,
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        basic_dcoords: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    major_end_cartesian, minor_end_cartesian = \
        get_ellipse_featcen_cartesian_coords(props, basic_coords, basic_dcoords)

    major_end_rotated_cartesian = pccoord._calc_rotated_feature_centric_coords(
        major_end_cartesian[0], major_end_cartesian[1], props['polar_angle_rad']
        )
    minor_end_rotated_cartesian = pccoord._calc_rotated_feature_centric_coords(
        minor_end_cartesian[0], minor_end_cartesian[1], props['polar_angle_rad']
        )
    return (major_end_rotated_cartesian, minor_end_rotated_cartesian)