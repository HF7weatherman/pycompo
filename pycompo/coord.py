from typing import Tuple
import xarray as xr
import numpy as np


def spherical2cartesian_featurecentric(
        basic_coords: Tuple[xr.DataArray, xr.DataArray],
        feature_data_in: list[xr.Dataset],
        feature_props: xr.Dataset,
        ) -> list[xr.Dataset]:
    lat = basic_coords[0]
    lon = basic_coords[1]
    dlat = lat.diff('lat').mean().values
    dlon = lon.diff('lon').mean().values

    feature_data_out = []
    for idx, feature_id in enumerate(feature_props['feature_id']):
        feature = feature_props.isel(feature=idx)
        centroid_lat, centroid_lon = _get_centroid_coords(
            basic_coords, (dlat, dlon), feature['centroid_idx']
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
    for idx, feature_id in enumerate(feature_props['feature_id']):
        feature = feature_props.isel(feature=idx)
        data = feature_centric_data_in[idx]

        rotated_feature_centric_coords = _calc_rotated_feature_centric_coords(
            data['feature_centric_x'], data['feature_centric_y'],
            feature['polar_angle_rad']
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