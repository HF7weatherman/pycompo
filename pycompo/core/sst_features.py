import numpy as np
import xarray as xr
from skimage.measure import regionprops
from typing import Tuple

from pyorg.core.geometry import get_cells_area
from pyorg.core.clusters import get_clusters, get_clusters_areas

from pycompo.core.utils import area_weighted_avg
from pycompo.core.coord import _get_centroid_coords


# ------------------------------------------------------------------------------
# Functions to get SST features and basic statistics
# --------------------------------------------------
def extract_sst_features(
        sst_dset: xr.DataArray,
        type: str,
        threshold: float,
        connectivity: int,
        min_area: float,
        property_list: list[str],
        ) -> Tuple[xr.DataArray, xr.Dataset]:
    feature_map, basic_feature_props = _get_sst_features(
        sst_dset, type=type, threshold=threshold, connectivity=connectivity,
        )
    advanced_feature_props = _get_feature_props_idx_space(
        feature_map, property_list,
        )
    feature_props = xr.merge([basic_feature_props, advanced_feature_props])
    feature_props = feature_props.assign({
        f'{sst_dset.name}_mean': _area_weighted_feature_mean(
            feature_map, sst_dset, feature_props,
            )
        })
    feature_map, feature_props = _remove_small_features(
        feature_map, feature_props, min_area,
        )
    return feature_map, feature_props


def _get_sst_features(
        data: xr.DataArray,
        type: str,
        threshold: float,
        connectivity: int,
        ) -> xr.DataArray:
    structure_element =_build_structure_element(connectivity)

    # Initialisation
    features_list, radii_list, areas_list, times_list = ([] for _ in range(4))
    total_features = 0
    cell_area = get_cells_area(data)
    
    def _get_sst_features_single_timestep(data, type, threshold):
        nonlocal total_features
        
        # Create threshold-based binary map and extract clusters from it
        if type == "warm":
            regions = data > threshold
        elif type == "cold":
            regions = data < threshold
        else:
            raise ValueError(
                "Please provide a valid feature type! Only 'warm' and " +
                "'cold' are supported."
                )
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
                data.sel(time=time), type, threshold,
                )
            times_list = times_list + [time.values]*len(feature_radii)
            features = xr.concat(features_list, "time")

    else:
        feature_radii = _get_sst_features_single_timestep(data, type, threshold)
        times_list = times_list + [data.time.values]*len(feature_radii)
        features = xr.concat(features_list, "time")
    
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


def _area_weighted_feature_mean(
        label_image: xr.DataArray,
        intensity_image: xr.DataArray,
        feature_props: xr.Dataset,
        ):
    cell_area_km2 = get_cells_area(intensity_image)/1000**2

    feature_mean = []
    for i, feature_id in enumerate(feature_props['feature_id']):
        feature = xr.where(label_image == feature_id, 1, 0)
        feature_area = feature_props['area_km2'].isel(feature=i)
        area_weight = feature * cell_area_km2/feature_area
        feature_mean.append(np.sum(area_weight * intensity_image).values)

    return xr.DataArray(
        data=np.array(feature_mean),
        dims=('feature',),
        coords={'feature_id': ('feature', feature_props['feature_id'].values)}
        )


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
            data_vars[f'{prop}_idx'] = (("feature", "component"), array)
        else:
            data_vars[f'{prop}_idx'] = (("feature",), array)

    return xr.Dataset(
        coords={
            'feature_id': ("feature", feature_props_dict['label']),
            'component': ("component", ['lat', 'lon'])
            },
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
# Update generic feature information based on selected features
# -------------------------------------------------------------
def _update_features(
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


# ------------------------------------------------------------------------------
# Helper functions
# ----------------
def set_global_feature_id(feature_list: list) -> list:
    global_feature_id = 1
    for idx, feature in enumerate(feature_list):
        global_feature_id_old = global_feature_id
        global_feature_id = global_feature_id + feature.sizes['feature']
        global_feature_ids = range(global_feature_id_old, global_feature_id)
        feature_list[idx]['feature_id'] = ('feature', global_feature_ids)

    return feature_list


# ------------------------------------------------------------------------------
# Functions to calculate feature quantities
# ------------------------------------------
def add_more_feature_props(
        feature_props: xr.Dataset,
        feature_centric_data: list[xr.Dataset],
        orig_coords: xr.Dataset,
):
    feature_props['centroid_sphere'] = _get_centroid_coords(
        orig_coords, feature_props['centroid_idx'],
        )
    for data in feature_centric_data:
        if 'tas-ts_bg' in data.data_vars:
            feature_props = _calc_feature_bg_field(
                feature_props, feature_centric_data, 'tas-ts',
                )
    return feature_props


def _calc_feature_bg_field(
        feature_props: xr.Dataset,
        feature_centric_data: list[xr.Dataset],
        var: str,
        ) -> xr.Dataset:
    bg_field = [
        area_weighted_avg(data[f"{var}_bg"], data['cell_area']).values
        for data in feature_centric_data
        ]
    feature_props[f'bg_{var}'] = ('feature', bg_field)

    return feature_props