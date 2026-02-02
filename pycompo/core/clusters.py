from typing import Tuple, List
from scipy import ndimage
import xarray as xr
import numpy as np


def get_clusters(
    convective_regions: xr.DataArray,
    periodic_longitude_clustering: bool=False,
    remove_edge_clusters: bool=False,
    **kwargs,
) -> Tuple[xr.DataArray, int]:
    """Get the convective clusters from convective regions."""
    clusters = xr.zeros_like(convective_regions)
    clusters_array, clusters_number = ndimage.label(
        convective_regions.values, **kwargs
        )
    if periodic_longitude_clustering:
        clusters_array, clusters_number = _periodic_longitude_clustering(
            clusters_array
            )
    if remove_edge_clusters:
        clusters_array, clusters_number = \
            _remove_latitude_edge_clusters(clusters_array)
    clusters.values = clusters_array
    return clusters, clusters_number


def get_clusters_areas(
    clusters: xr.DataArray, clusters_index: List[int], cells_area: xr.DataArray
) -> List[float]:
    """Get a list with the area of each convective core."""
    return ndimage.sum(cells_area, clusters, clusters_index)


def _periodic_longitude_clustering(
        clusters_array: np.ndarray
        ) -> Tuple[np.ndarray, int]:
    for lat in range(clusters_array.shape[0]):
        if clusters_array[lat, 0] > 0 and clusters_array[lat, -1] > 0:
            clusters_array[clusters_array == clusters_array[lat, -1]] = \
                clusters_array[lat, 0]
        # Prevent rare finger lake case
        if clusters_array[lat, 0] > 0 and clusters_array[lat, -1] > 0:
            clusters_array[clusters_array == clusters_array[lat, 0]] = \
                clusters_array[lat, -1]
    clusters_array = _relabel_clusters(clusters_array)
    clusters_number = clusters_array.max()
    return clusters_array, clusters_number


def _remove_latitude_edge_clusters(
        clusters_array: np.ndarray
        ) -> Tuple[np.ndarray, int]:
    for lon in range(clusters_array.shape[1]):
        if clusters_array[0, lon] > 0:
            clusters_array[clusters_array == clusters_array[0, lon]] = 0
        if clusters_array[-1, lon] > 0:
            clusters_array[clusters_array == clusters_array[-1, lon]] = 0
    clusters_array = _relabel_clusters(clusters_array)
    clusters_number = clusters_array.max()
    return clusters_array, clusters_number


def _relabel_clusters(clusters_array: np.ndarray) -> np.ndarray:
    return np.unique(clusters_array, return_inverse=True)[1].reshape(
        clusters_array.shape)