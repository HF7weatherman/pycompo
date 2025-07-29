import math
import numpy as np
from typing import Tuple
import xarray as xr
from skimage.measure import regionprops

from pyorg.core.geometry import get_cells_area
from pyorg.core.convection import convective_regions
from pyorg.core.clusters import get_clusters, get_clusters_areas, \
    get_clusters_centroids


# ------------------------------------------------------------------------------
# Functions to get SST features and basic statistics
# --------------------------------------------------
def get_sst_features(
        data: xr.DataArray,
        threshold: float,
        connectivity: int,
        ) -> xr.DataArray:
    structure_element =_build_structure_element(connectivity)

    # Initialisation
    clusters_list, centroids_list, radiuses_list, areas_list, times_list \
        = ([] for _ in range(5))
    total_clusters = 0
    cells_area = get_cells_area(data)
    
    def _get_sst_features_single_timestep(data, threshold):
        nonlocal total_clusters
        
        # Create threshold-based binary map and extract clusters from it
        threshold, regions = convective_regions(data, threshold=threshold)
        clusters, clusters_number = get_clusters(
            regions,
            periodic_longitude_clustering=True, remove_edge_clusters=True,
            structure=structure_element,
            )
        
        cluster_indices = list(range(1, clusters_number + 1))
        cluster_centroids = get_clusters_centroids(
            clusters, cluster_indices, regions
            )
        cluster_areas = get_clusters_areas(
            clusters, cluster_indices, cells_area
            )
        cluster_areas = cluster_areas/1000**2  # km^2
        cluster_radii = np.sqrt(cluster_areas / np.pi) # km

        clusters += total_clusters
        clusters = clusters.where(clusters != total_clusters, 0)
        clusters_list.append(clusters)
        total_clusters += clusters_number
        
        # append that stuff to list
        areas_list.extend(cluster_areas.tolist())
        centroids_list.extend(cluster_centroids)
        radiuses_list.extend(cluster_radii.tolist())

        return cluster_radii

    if "time" in data.dims and data.sizes["time"] > 1:
        for time in data.time:
            clusters_radiuses = _get_sst_features_single_timestep(
                data.sel(time=time), threshold
                )
            times_list = times_list + [time.values]*len(clusters_radiuses)
            clusters = xr.concat(clusters_list, "time") 

    else: 
        _get_sst_features_single_timestep(data, threshold)
    
    centroids_lats, centroids_lons = zip(*centroids_list)
    cluster_props = xr.Dataset(
        coords={
            'cluster_id': ("cluster", list(range(1, total_clusters + 1)))
        },
        data_vars={
            'centroid_lat': ("cluster", list(centroids_lats)),
            'centroid_lon': ("cluster", list(centroids_lons)),
            'radius_km': ("cluster", radiuses_list),
            'area_km2': ("cluster", areas_list),
            'time': ("cluster", times_list),
            }
        )
    return clusters, cluster_props


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
# Functions to get more advanced statistics of SST features
# ---------------------------------------------------------
def get_feature_props_idx_space(
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


def _transform_feature_props_idx_space_to_xrarray(feature_props_dict):
    data_vars = {}

    for prop, values in feature_props_dict.items():
        first_val = values[0]
        array = np.array(values, dtype=float)
        if isinstance(first_val, tuple):
            data_vars[prop] = (("cluster", f"{prop}_component"), array)
        else:
            data_vars[prop] = (("cluster",), array)

    return xr.Dataset(
        coords={'cluster_id': ("cluster", feature_props_dict['label'])},
        data_vars=data_vars
        ).drop('label')


def get_cutout_data_box_idxs(
        feature_map: xr.DataArray,
        feature_props: xr.Dataset,
        search_RadRatio: float,
        ) -> Tuple[xr.DataArray, xr.Dataset]:
    feature_props['cutout_idxs'] = _get_cutout_data_box_idxs(
        feature_props, search_RadRatio,
    )
    return remove_lat_edge_boxes(feature_map, feature_props)


def _get_cutout_data_box_idxs(
        feature_props: xr.Dataset,
        search_RadRatio: float,
        ) -> xr.DataArray:
    cutout_idxs_list = []
    for cluster in feature_props['cluster_id']:
        sample = feature_props.where(
            feature_props['cluster_id']==cluster, drop=True
            ).squeeze()
        R_maj = search_RadRatio/2 * sample['axis_major_length']
        cutout_idxs = {
            'lat_lower': _round_away_from_zero(sample['centroid'][0] - R_maj),
            'lat_upper': _round_away_from_zero(sample['centroid'][0] + R_maj),
            'lon_left': _round_away_from_zero(sample['centroid'][1] - R_maj),
            'lon_right': _round_away_from_zero(sample['centroid'][1] + R_maj),
        }
        cutout_idxs_list.append(
            (cutout_idxs['lat_lower'], cutout_idxs['lat_upper'],
             cutout_idxs['lon_left'], cutout_idxs['lon_right'])
             )
    
    return xr.DataArray(
        name = 'cutout_idxs',
        dims = ['cluster', 'cutout_idx_component'],
        coords = {
            'cluster_id': ('cluster', feature_props['cluster_id'].data),
            'cutout_idx_component': (
                'cutout_idx_component',
                ['lat_lower', 'lat_upper', 'lon_left', 'lon_right'],
                )
            },
        data = np.array(cutout_idxs_list)
        )


def _round_away_from_zero(x: float) -> int:
    return int(math.copysign(math.ceil(abs(x)), x))


# ------------------------------------------------------------------------------
# Subsampling of detected features
# --------------------------------
def remove_small_features(
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
    

def remove_lat_edge_boxes(
        feature_map: xr.DataArray,
        feature_props: xr.Dataset,
        ):
    N_lat = feature_map.sizes['lat']
    m1 = (feature_props['cutout_idxs'].sel(cutout_idx_component='lat_lower') >= 0)
    m2 = (feature_props['cutout_idxs'].sel(cutout_idx_component='lat_upper') <= N_lat)
    mask = m1 & m2
    feature_props = feature_props.sel(cluster=mask.values)
    feature_map = _update_feature_map(feature_map, feature_props)
    return feature_map, feature_props


def _update_feature_map(
    feature_map: xr.DataArray,
    feature_props: xr.Dataset,
    ) -> xr.DataArray:
    return feature_map.where(
        feature_map.isin(feature_props['cluster_id']) | feature_map.isnull(), 0
        )