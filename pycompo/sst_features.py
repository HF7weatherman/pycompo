import numpy as np
from typing import Tuple
import xarray as xr
from skimage.measure import regionprops

from pycompo.utils import round_away_from_zero

from pyorg.core.geometry import get_cells_area
from pyorg.core.convection import convective_regions
from pyorg.core.clusters import get_clusters, get_clusters_areas


# ------------------------------------------------------------------------------
# Functions to get SST features and basic statistics
# --------------------------------------------------
def extract_sst_features(
        sst_dset: xr.DataArray,
        threshold: float,
        connectivity: int,
        property_list: list[str],
        feature_min_area: float=0.,
        ) -> Tuple[xr.DataArray, xr.Dataset]:
    feature_map, feature_props = _get_sst_features(
        sst_dset, threshold=threshold, connectivity=connectivity
        )
    feature_props = xr.merge([
        feature_props, _get_feature_props_idx_space(feature_map, property_list)
        ])
    feature_map, feature_props = _remove_small_features(
        feature_map, feature_props, feature_min_area,
        )
    return feature_map, feature_props


def _get_sst_features(
        data: xr.DataArray,
        threshold: float,
        connectivity: int,
        ) -> xr.DataArray:
    structure_element =_build_structure_element(connectivity)

    # Initialisation
    features_list, radii_list, areas_list, times_list = ([] for _ in range(4))
    total_features = 0
    cell_area = get_cells_area(data)
    
    def _get_sst_features_single_timestep(data, threshold):
        nonlocal total_features
        
        # Create threshold-based binary map and extract clusters from it
        threshold, regions = convective_regions(data, threshold=threshold)
        features, features_number = get_clusters(
            regions, periodic_longitude_clustering=True,
            remove_edge_clusters=True, structure=structure_element,
            )
        
        feature_indices = list(range(1, features_number + 1))
        feature_areas = get_clusters_areas(features, feature_indices, cell_area)
        feature_areas = feature_areas/1000**2  # km^2
        feature_radii = np.sqrt(feature_areas / np.pi) # km

        features += total_features
        features = features.where(features != total_features, 0)
        features_list.append(features)
        total_features += features_number
        
        # append that stuff to list
        areas_list.extend(feature_areas.tolist())
        radii_list.extend(feature_radii.tolist())

        return feature_radii

    if "time" in data.dims and data.sizes["time"] > 1:
        for time in data.time:
            feature_radii = _get_sst_features_single_timestep(
                data.sel(time=time), threshold
                )
            times_list = times_list + [time.values]*len(feature_radii)
            features = xr.concat(features_list, "time") 

    else: 
        _get_sst_features_single_timestep(data, threshold)
    
    feature_props = xr.Dataset(
        coords={
            'feature_id': ("feature", list(range(1, total_features + 1)))
        },
        data_vars={
            'radius_km': ("feature", radii_list),
            'area_km2': ("feature", areas_list),
            'time': ("feature", times_list),
            }
        )
    return features, feature_props


def _get_feature_props_idx_space(
        feature_map: xr.DataArray,
        props_lst: list,
        ) -> xr.Dataset:
    feature_props = []
    if "time" in feature_map.dims:
        for time in feature_map['time']:
            feature_props.extend(
                regionprops(feature_map.sel(time=time).values)
                )
    else:
        feature_props.extend(regionprops(feature_map.values))

    if 'label' not in props_lst: props_lst.extend(['label'])
    feature_props_dict = {
        prop: [feature[prop] for feature in feature_props] for prop in props_lst
        }
    
    return _transform_feature_props_idx_space_to_xrarray(feature_props_dict)


def _remove_small_features(
        feature_map: xr.DataArray,
        feature_props: xr.Dataset,
        feature_min_area: float,
        area_varname: str='area_km2',
        ) -> Tuple[xr.DataArray, xr.Dataset]:
    feature_props = feature_props.where(
        feature_props[area_varname]>feature_min_area, drop=True,
        )
    feature_map = _update_feature_map(feature_map, feature_props)
    return feature_map, feature_props


def _transform_feature_props_idx_space_to_xrarray(feature_props_dict):
    data_vars = {}

    for prop, values in feature_props_dict.items():
        first_val = values[0]
        array = np.array(values, dtype=float)
        if isinstance(first_val, tuple):
            data_vars[f'{prop}_idx'] = (("feature", f"{prop}_component"), array)
        else:
            data_vars[f'{prop}_idx'] = (("feature",), array)

    return xr.Dataset(
        coords={'feature_id': ("feature", feature_props_dict['label'])},
        data_vars=data_vars
        ).drop('label_idx')


def _build_structure_element(connectivity: int=4) -> list:
    if connectivity not in [4, 8]:
        raise ValueError(
            "Please provide a valid feature connectivity! Only 4- and 8-" +
            "connectivity supported."
            )
    if connectivity == 4:
        structure_element = [[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]]
    elif connectivity == 8:
        structure_element = [[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]
    return structure_element


# ------------------------------------------------------------------------------
# Extract data around SST features
# --------------------------------
def cutout_feature_data(
        data: xr.Dataset,
        feature_props: xr.Dataset,
        search_RadRatio: float,
        ) -> list[xr.Dataset]:
    N_lat = data.sizes['lat']
    N_lon = data.sizes['lon']

    feature_data = []
    for idx, feature_id in enumerate(feature_props['feature_id']):
        feature = feature_props.isel(feature=idx)
        data_sample = data.sel(time=feature['time'])

        feature_data_bbox = _get_feature_data_bbox(feature, search_RadRatio)
        if _is_lat_edge_bbox(feature_data_bbox, N_lat):
            continue
        
        feature_data.append(
            _cutout_feature_data(data_sample, feature_data_bbox, N_lon)
            )
    return feature_data


def _get_feature_data_bbox(
        feature: xr.Dataset,
        search_RadRatio: float,
        ) -> xr.DataArray:
    R_maj = search_RadRatio/2 * feature['axis_major_length_idx']
    feature_data_bbox = {
        'lat_lower': round_away_from_zero(feature['centroid_idx'][0]-R_maj),
        'lat_upper': round_away_from_zero(feature['centroid_idx'][0]+R_maj),
        'lon_left':  round_away_from_zero(feature['centroid_idx'][1]-R_maj),
        'lon_right': round_away_from_zero(feature['centroid_idx'][1]+R_maj),
    }
    return feature_data_bbox


def _is_lat_edge_bbox(
        feature_data_bbox: dict[str, int],
        N_lat: int,
        ) -> bool:
    m1 = (feature_data_bbox['lat_lower'] >= 0)
    m2 = (feature_data_bbox['lat_upper'] < N_lat)
    return not (m1 and m2)


def _cutout_feature_data(
        data: xr.Dataset,
        feature_data_bbox: dict[str, int],
        N_lon: int,
        ) -> xr.Dataset:
    lat_lower = feature_data_bbox['lat_lower']
    lat_upper = feature_data_bbox['lat_upper']
    lon_left  = feature_data_bbox['lon_left']
    lon_right = feature_data_bbox['lon_right']

    lat_select_idxs = np.arange(lat_lower, lat_upper+1)    
    if lon_right+1 >= N_lon:
        lon_select_idxs = np.concatenate(
            [np.arange(lon_left, N_lon), np.arange(0, (lon_right+1) - N_lon)]
            )
    else: 
        lon_select_idxs = np.arange(lon_left, lon_right+1)

    return data.isel(lat=lat_select_idxs, lon=lon_select_idxs)


# ------------------------------------------------------------------------------
# Update generic feature information based on selected features
# -------------------------------------------------------------
def update_features(
        feature_map: xr.DataArray,
        feature_props: xr.Dataset,
        feature_data: list[xr.Dataset],
        ) -> Tuple[xr.DataArray, xr.Dataset]:
    keep_features = [int(data['feature_id'].values) for data in feature_data]
    feature_props = feature_props.where(
        feature_props['feature_id'].isin(keep_features), drop=True,
        )
    feature_map = _update_feature_map(feature_map, feature_props)
    return feature_map, feature_props


def _update_feature_map(
    feature_map: xr.DataArray,
    feature_props: xr.Dataset,
    ) -> xr.DataArray:
    return feature_map.where(
        feature_map.isin(feature_props['feature_id']) | feature_map.isnull(), 0
        )