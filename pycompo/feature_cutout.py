import numpy as np
import xarray as xr
from typing import Tuple

import pycompo.coord as pccoord
from pycompo.sst_features import update_features
from pycompo.utils import round_away_from_zero


# ------------------------------------------------------------------------------
# Extract data around SST features
# --------------------------------
def get_featcen_data_cutouts(
        dset: xr.Dataset,
        feature_props: xr.Dataset,
        ellipse: dict,
        search_RadRatio: float,
        ) -> Tuple[xr.Dataset, xr.Dataset, list[xr.Dataset]]:
    """
    Extracts and processes feature-centric data cutouts from a given dataset.

    This function performs the following steps:
    1. Extracts feature data cutouts from the input dataset based on provided 
        feature properties and a search radius ratio.
    2. Updates the dataset and feature properties based on the extracted
        feature data.
    3. Adds a feature-centric cartesian system to the feature data.
    4. Adds a rotated feature-centric cartesian coordinates to the feature data.

    Parameters
    ----------
    dset : xr.Dataset
        The input xarray Dataset containing the data from which features will
        be extracted.
    feature_props : xr.Dataset
        An xarray Dataset containing properties of the features to be extracted.
    search_RadRatio : float
        The ratio used to determine the search radius for extracting featur
        data.

    Returns
    -------
    dset : xr.Dataset
        The updated xarray Dataset with feature data cutouts.
    feature_props : xr.Dataset
        The updated feature properties Dataset.
    feature_centric_data : list[xr.Dataset]
        A list of xarray Datasets containing the feature-centric data cutouts
        in rotated cartesian coordinates.
    """
    feature_data = cutout_feature_data(dset, feature_props, search_RadRatio)
    dset['sst_feature'], feature_props, ellipse = update_features(
        dset['sst_feature'], feature_props, ellipse, feature_data,
        )

    feature_centric_data = pccoord.add_featcen_coords(
        pccoord.get_coords_orig(dset), feature_data, feature_props, ellipse,
        )

    return dset, feature_props, feature_centric_data


def cutout_feature_data(
        data: xr.Dataset,
        feature_props: xr.Dataset,
        search_RadRatio: float,
        ) -> list[xr.Dataset]:
    """
    Extracts cutout regions of data around detected features from an xr.Dataset.

    Given a dataset and a set of feature properties, this function extracts
    sub-datasets (centered on each feature) within a bounding box determined by
    `search_RadRatio`.
    Features located at the latitude edge are skipped.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset containing the data to be cut out.
    feature_props : xr.Dataset
        Dataset containing properties of detected features, including their 
        positions and times.
    search_RadRatio : float
        Ratio used to determine the size of the bounding box around each 
        feature.

    Returns
    -------
    list[xr.Dataset]
        A list of xarray Datasets, each corresponding to a cutout region around 
        a feature.
    """
    N_lat = data.sizes['lat']
    N_lon = data.sizes['lon']

    feature_data = []
    for idx, _ in enumerate(feature_props['feature_id']):
        feature = feature_props.isel(feature=idx)
        data_sample = data.sel(time=feature['time'])

        feature_data_bbox = _get_feature_data_bbox(feature, search_RadRatio)
        if _is_lat_edge_bbox(feature_data_bbox, N_lat):
            continue

        feature_data.append(
            _cutout_feature_data(data_sample, feature_data_bbox, N_lon)
            )
    return feature_data


def _cutout_feature_data(
        data: xr.Dataset,
        feature_data_bbox: dict[str, int],
        N_lon: int,
        ) -> xr.Dataset:
    """
    Extracts a subset of the input xarray Dataset based on specified latitude 
    and longitude bounds.

    This function selects a rectangular region from the input dataset using the 
    provided bounding box.
    If the longitude bounds cross the dateline (i.e., wrap around), the 
    selection is handled accordingly.

    Parameters
    ----------
    data : xr.Dataset
        The input xarray Dataset containing 'lat' and 'lon' dimensions.
    feature_data_bbox : dict[str, int]
        A dictionary specifying the bounding box with keys:
            - 'lat_lower': Lower index for latitude (inclusive).
            - 'lat_upper': Upper index for latitude (inclusive).
            - 'lon_left': Left index for longitude (inclusive).
            - 'lon_right': Right index for longitude (inclusive).
    N_lon : int
        The total number of longitude points in the dataset (used for 
        wrap-around handling).

    Returns
    -------
    xr.Dataset
        The subset of the input dataset corresponding to the specified bounding 
        box.
    """
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


def _get_feature_data_bbox(
        feature: xr.Dataset,
        search_RadRatio: float,
        ) -> xr.DataArray:
    """
    Calculate the bounding box of a feature in index space based on its 
    centroid and major axis length.

    Parameters
    ----------
    feature : xr.Dataset
        An xarray Dataset containing at least the following fields:
            - 'centroid_idx': A sequence or array-like with two elements 
            representing the (latitude, longitude) indices of the feature's 
            centroid.
            - 'axis_major_length_idx': The length of the major axis of the 
            feature in index units.
    search_RadRatio : float
        A scaling factor to determine the search radius as a ratio of the major 
        axis length.

    Returns
    -------
    xr.DataArray
        A dictionary-like object with the bounding box indices:
            - 'lat_lower': Lower latitude index bound.
            - 'lat_upper': Upper latitude index bound.
            - 'lon_left':  Left longitude index bound.
            - 'lon_right': Right longitude index bound.

    Notes:
    - The bounding box is computed by expanding from the centroid by half the 
    scaled major axis length in all directions.
    - The `round_away_from_zero` function is used to round the bounds.
    """
    R_maj = search_RadRatio / 2 * feature['axis_major_length_idx']
    feature_data_bbox = {
        'lat_lower': round_away_from_zero(feature['centroid_idx'][0] - R_maj),
        'lat_upper': round_away_from_zero(feature['centroid_idx'][0] + R_maj),
        'lon_left':  round_away_from_zero(feature['centroid_idx'][1] - R_maj),
        'lon_right': round_away_from_zero(feature['centroid_idx'][1] + R_maj),
    }
    return feature_data_bbox


def _is_lat_edge_bbox(
        feature_data_bbox: dict[str, int],
        N_lat: int,
        ) -> bool:
    """
    Determine if a feature's bounding box touches or exceeds the latitude edges.

    Parameters
    ----------
    feature_data_bbox : dict[str, int]
        Dictionary containing the bounding box coordinates with keys 'lat_lower'
        and 'lat_upper'.
    N_lat : int
        The total number of latitude grid points.

    Returns
    -------
    bool
        True if the bounding box touches or exceeds the latitude boundaries
        (i.e., at the edge), False otherwise.
    """
    m1 = (feature_data_bbox['lat_lower'] >= 0)
    m2 = (feature_data_bbox['lat_upper'] < N_lat)
    return not (m1 and m2)