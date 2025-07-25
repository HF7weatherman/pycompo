import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from typing import Tuple


# ------------------------------------------------------------------------------
# Functions for building a climatology
# ------------------------------------
def build_hourly_climatology(
        dset: xr.DataArray | xr.Dataset,
        clim_baseyear: str,
        ) -> xr.DataArray | xr.Dataset:
    """
    Computes the hourly climatology of the input xarray DataArray or Dataset.

    Parameters
    ----------
    dset : xr.DataArray or xr.Dataset
        Input data with a 'time' dimension to compute climatology from.
    clim_baseyear : str, default="2000"
        The base year to use when converting group coordinates to datetime64.
        This year is assigned to the time coordinate of the climatology.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Hourly climatology.
    """
    dset = dset.assign_coords(group_time=_create_grouper_coord(dset))
    climatology = dset.groupby('group_time').mean(dim='time')
    climatology = _grouper_coord2datetime64(climatology, clim_baseyear)

    return climatology


def _create_grouper_coord(dset: xr.DataArray | xr.Dataset):
    return xr.apply_ufunc(
        _format_time_label,
        dset['time'].dt.month,
        dset['time'].dt.day,
        dset['time'].dt.hour,
        vectorize=True,
        dask='allowed',
        output_dtypes=[str]
    )


def _format_time_label(month, day, hour):
    return f"{month:02d}-{day:02d}_{hour:02d}"


def _grouper_coord2datetime64(
        climatology: xr.DataArray | xr.Dataset,
        clim_baseyear: str
        ) -> xr.DataArray | xr.Dataset:
    # Split group_time into month, day, and hour and assign as a coordinate
    group_time_str = climatology['group_time'].astype(str)
    month = group_time_str.str.slice(0, 2).astype(int)
    day   = group_time_str.str.slice(3, 5).astype(int)
    hour  = group_time_str.str.slice(6, 8).astype(int)
    climatology = climatology.assign_coords(month=month, day=day, hour=hour)

    # Build np.datetime64 coordinate
    date_strings = [
        f"{clim_baseyear}-{m:02d}-{d:02d}T{h:02d}:00"
        for m, d, h in zip(
            climatology['month'].values,
            climatology['day'].values,
            climatology['hour'].values
            )
    ]
    clim_time = np.array(date_strings, dtype='datetime64[m]')

    # Assign as a new coordinate
    climatology = climatology.assign_coords(time=("group_time", clim_time))
    climatology = climatology.swap_dims({'group_time': 'time'}).\
        drop(['group_time', 'month', 'day', 'hour'])
    
    return climatology


# ------------------------------------------------------------------------------
# Functions for Gaussian filtering
# --------------------------------
def get_gaussian_filter_bg_ano(
        dset: xr.Dataset | xr.DataArray,
        **kwargs,
        ) -> xr.Dataset:
    """
    Applies a Gaussian lowpass filter to each variable in the input dataset and
    computes the background and anomaly fields.

    Parameters
    ----------
    dset : xr.Dataset or xr.DataArray
        Input dataset or data array containing the variables to be filtered.
    **kwargs
        Additional keyword arguments passed to `gaussian_lowpass_filter`.

    Returns
    -------
    xr.Dataset
        Dataset with added variables for each input variable:
        - '<var>_bg': the background field after Gaussian filtering.
        - '<var>_ano': the anomaly field (original minus background).

    Notes
    -----
    If a DataArray is provided, it is converted to a Dataset. The anomaly is
    computed as the difference between the original variable and its background
    field.
    """
    if isinstance(dset, xr.DataArray):
        dset = dset.to_dataset()

    for var in dset.data_vars:
        dset[f'{var}_bg'] = (gaussian_lowpass_filter(dset[var], **kwargs))
        dset[f'{var}_ano'] = dset[var] - dset['ts_bg']

    return dset


def gaussian_lowpass_filter(
        dset: xr.DataArray,
        lat_mid: float,
        Lc_km: float,
        truncate: float, 
        ) -> Tuple[list, np.ndarray]:
    """
    Applies a Gaussian lowpass filter to the input DataArray.

    Parameters
    ----------
    dset : xr.DataArray
        Input data array containing the variable to be filtered. Must have 'lat' and 'lon' dimensions.
    lat_mid : float
        Latitude at which to compute the conversion from kilometers to degrees.
    Lc_km : float
        Cutoff length scale in kilometers for the Gaussian filter.
    truncate : float
        Truncate value for the Gaussian filter.

    Returns
    -------
    list
        List of dimension names of the input DataArray, in the order used for filtering.
    np.ndarray
        Filtered data as a NumPy array after applying the Gaussian filter.

    Raises
    ------
    ValueError
        If the input DataArray does not contain 'lat' and 'lon' dimensions.
    """
    # Some checks and preprocessing of the dataset
    if not all([dim in dset.dims for dim in ['lat', 'lon']]):
        raise ValueError(
            "Please provide a dataset that contains 'lat' and 'lon' as " +
            "dimension names for latitude and longitude, respectively."
            )
    dset = dset.transpose('lat', 'lon', ...)
    n_dims = len(dset.dims)

    # Calculate degrees per gridpoint
    dlat = dset['lat'].diff('lat').mean().values
    dlon = dset['lon'].diff('lon').mean().values

    # Define cutoff wavenumber (in radians per degree)
    Lc_deg_lat, Lc_deg_lon = _Lc_km2deg(Lc_km, lat_mid)  # degrees
    kc_deg_lat = 2 * np.pi / Lc_deg_lat  # radians per degree
    kc_deg_lon = 2 * np.pi / Lc_deg_lon  # radians per degree

    # Compute sigma in grid points
    sigma_x = (2 * np.pi) / (kc_deg_lon * np.sqrt(2) * dlon)
    sigma_y = (2 * np.pi) / (kc_deg_lat * np.sqrt(2) * dlat)
    sigmas = [sigma_y, sigma_x]
    while len(sigmas) < n_dims: sigmas.append(0)

    return list(dset.dims), gaussian_filter(
        dset, sigma=sigmas, truncate=truncate
        )


def _Lc_km2deg(
        Lc_km: float,
        lat_mid: float
        ) -> Tuple[float, float]:
    """
    Convert a length in kilometers to degrees of latitude and longitude at a
    given latitude.

    Parameters
    ----------
    Lc_km : float
        Length in kilometers to be converted.
    lat_mid : float
        Latitude (in degrees) at which the conversion is performed.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - The equivalent length in degrees of latitude.
        - The equivalent length in degrees of longitude at the specified
        latitude.

    Notes
    -----
    - The conversion uses the average value of 1 degree latitude ≈ 111.195 km.
    - The longitude conversion accounts for the cosine of the latitude.
    """
    deg_per_km_lat = 1/111.195
    deg_per_km_lon = 1/(111.195 * np.cos(np.deg2rad(lat_mid)))
    return (Lc_km * deg_per_km_lat, Lc_km * deg_per_km_lon)