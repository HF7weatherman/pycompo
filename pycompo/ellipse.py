import numpy as np
import xarray as xr
from typing import Tuple

from pycompo.coord import _calc_rota_featcen_cart_coords, _get_centroid_coords
from pycompo.coord import KM_PER_DEGREE_EQ


def get_ellipse_params(
        feature_props: xr.Dataset,
        coords_sphere: xr.Dataset,
        ) -> dict:
    ellipse_idx = _get_ellipse_params_idx(feature_props)
    ellipse_sphere = _ellipse_idx2sphere(ellipse_idx, coords_sphere)
    ellipse_featcen_sphere = _ellipse_sphere2featcen_sphere(ellipse_sphere)
    ellipse_featcen_cart = _ellipse_sphere2featcen_cart(
        ellipse_sphere, coords_sphere
        )
    ellipse_rota_featcen_cart = _ellipse_featcen_cart2rota_featcen_cart(
        ellipse_featcen_cart
        )
    
    return {
        'idx': ellipse_idx,
        'sphere': ellipse_sphere,
        'featcen_sphere': ellipse_featcen_sphere,
        'featcen_cart': ellipse_featcen_cart,
        'rota_featcen_cart': ellipse_rota_featcen_cart,
    }


# ------------------------------------------------------------------------------
# Get ellipse parameters in index space from skimage output
# ---------------------------------------------------------
def _get_ellipse_params_idx(
        feature_props: xr.Dataset,
        ) -> xr.Dataset:
    centroid_idx = feature_props['centroid_idx'].values
    orientation_idx = feature_props['orientation_idx'].values
    maj_len_idx = feature_props['axis_major_length_idx'].values
    min_len_idx = feature_props['axis_minor_length_idx'].values

    polar_angle_rad = [_calc_polar_angle_rad(x) for x in orientation_idx]
    maj_end = [
        _calc_axis_major_end(length, angle) \
            for length, angle in zip(maj_len_idx, polar_angle_rad)
        ]
    min_end = [
        _calc_axis_minor_end(length, angle) \
            for length, angle in zip(min_len_idx, polar_angle_rad)
        ]
    
    return _create_ellipse_dataset(
        centroid_idx, polar_angle_rad, maj_end, min_end, feature_props.coords,
    )


def _calc_polar_angle_rad(
        x: float
        ) -> float:
    """
    Calculate polar angle in radians from ellipse orientation

    This function converts an orientation angle `x` (in radians), measured
    counter-clockwise from the x-axis, to the corresponding polar angle,
    measured clockwise from the x-axis, for the major axis of an ellipse.
    The conversion is performed as follows:
    - If `x` is positive, the polar angle is computed as (π/2 - x).
    - If `x` is zero or negative, the polar angle is computed as (-π/2 - x).

    Parameters
    ----------
    x : float
        The orientation angle of the ellipse in radians.

    Returns
    -------
    float
        The corresponding polar angle in radians.
    """
    return (np.pi/2 - x) if x > 0 else (-np.pi/2 - x)


def _calc_axis_major_end(
        axis_major_length: float,
        polar_angle_rad: float,
        ) -> Tuple[float, float]:
    """
    Calculate the end point coordinates of the major axis of an ellipse.

    Given the length of the major axis and the polar angle (in radians),
    this function computes the (x, y) coordinates of the end point of the
    major axis, assuming the ellipse is centered at the origin.

    Parameters
    ----------
    axis_major_length : float
        The total length of the major axis of the ellipse.
    polar_angle_rad : float
        The polar angle (in radians) from the x-axis to the major axis.

    Returns
    -------
    Tuple[float, float]
        The (x, y) coordinates of the end point of the major axis.
    """
    lon_end = axis_major_length / 2 * np.cos(polar_angle_rad)
    lat_end = axis_major_length / 2 * np.sin(polar_angle_rad)
    return (lat_end, lon_end)


def _calc_axis_minor_end(
        axis_minor_length: float,
        polar_angle_rad: float,
        ) -> Tuple[float, float]:
    """
    Calculate the end point coordinates of the minor axis of an ellipse.

    Given the length of the minor axis and the polar angle (in radians),
    this function computes the (x, y) coordinates of the end point of the
    minor axis, assuming the ellipse is centered at the origin.

    Parameters
    ----------
    axis_minor_length : float
        The length of the minor axis of the ellipse.
    polar_angle_rad : float
        The orientation angle of the ellipse in radians (measured from the x-axis).

    Returns
    -------
    Tuple[float, float]
        The (x, y) coordinates of the end point of the minor axis.
    """
    lon_end = axis_minor_length / 2 * np.cos(polar_angle_rad + np.pi / 2)
    lat_end = axis_minor_length / 2 * np.sin(polar_angle_rad + np.pi / 2)
    return (lat_end, lon_end)


# ------------------------------------------------------------------------------
# Get coordinates of ellipse in various coordinate systems
# --------------------------------------------------------
def _ellipse_idx2sphere(
        ellipse_idx: xr.Dataset,
        coords_sphere: xr.Dataset,
    ) -> xr.Dataset:    
    centr_sphere = _get_centroid_coords(coords_sphere, ellipse_idx['centroid'])
    polar_angle_rad_sphere = ellipse_idx['polar_angle_rad']
    maj_end_sphere = ellipse_idx['maj_end'] * coords_sphere['dsphere']
    min_end_sphere = ellipse_idx['min_end'] * coords_sphere['dsphere']

    return _create_ellipse_dataset(
        centr_sphere.data,
        polar_angle_rad_sphere.data,
        maj_end_sphere.data,
        min_end_sphere.data,
        ellipse_idx.coords,
    )


def _ellipse_sphere2featcen_sphere(
        ellipse_sphere: xr.Dataset,
    ) -> xr.Dataset:
    ellipse_featcen_sphere = ellipse_sphere.copy()
    ellipse_featcen_sphere['centroid'] = ellipse_featcen_sphere['centroid'] * 0.
    return ellipse_featcen_sphere


def _ellipse_sphere2featcen_cart(
        ellipse_sphere: xr.Dataset,
        coords_sphere: xr.Dataset,
    ) -> xr.Dataset:
    dcart = _get_dcart(ellipse_sphere, coords_sphere)

    centroid_cart = ellipse_sphere['centroid'] * 0.
    maj_end_cart = ellipse_sphere['maj_end'] * dcart
    min_end_cart = ellipse_sphere['min_end'] * dcart
#    polar_angle_rad_cart = np.arctan(
#        maj_end_cart.sel(component='lat') / maj_end_cart.sel(component='lon')
#        )
    polar_angle_rad_cart = ellipse_sphere['polar_angle_rad']
    
    coords = ellipse_sphere.assign_coords(
        {'component': ('component', ['y', 'x'])}
        ).coords
    
    return _create_ellipse_dataset(
        centroid_cart.data,
        polar_angle_rad_cart.data,
        maj_end_cart.data,
        min_end_cart.data,
        coords,
    )


def _ellipse_featcen_cart2rota_featcen_cart(
        ellipse_cart: xr.Dataset,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    centroid_rota_cart = ellipse_cart['centroid']
    polar_angle_rad_rota_cart = ellipse_cart['polar_angle_rad'] * 0.
    maj_end_rota_cart = _calc_ellipse_rota_cart(ellipse_cart, axis='maj_end')
    min_end_rota_cart = _calc_ellipse_rota_cart(ellipse_cart, axis='min_end')
    
    return _create_ellipse_dataset(
        centroid_rota_cart.data,
        polar_angle_rad_rota_cart.data,
        maj_end_rota_cart.data,
        min_end_rota_cart.data,
        ellipse_cart.coords,
    )


def _calc_ellipse_rota_cart(
        ellipse_cart: xr.Dataset,
        axis: str
        ) -> xr.DataArray:
    end_rota_cart = _calc_rota_featcen_cart_coords(
        ellipse_cart[axis].sel(component='x'),
        ellipse_cart[axis].sel(component='y'),
        ellipse_cart['polar_angle_rad'],
        )
    return xr.DataArray(
        coords = ellipse_cart[axis].coords,
        data = np.array([end_rota_cart[1], end_rota_cart[0]]).transpose()
    )


# ------------------------------------------------------------------------------
# Helper functions
# ----------------
def _create_ellipse_dataset(
        centroid, polar_angle_rad, maj_end, min_end, coords,
        ):
    return xr.Dataset(
        data_vars = {
            'centroid': (('feature', 'component'), centroid),
            'polar_angle_rad': (('feature',), polar_angle_rad),
            'maj_end': (('feature', 'component'), maj_end),
            'min_end': (('feature', 'component'), min_end),
        },
        coords = coords,
    )


def _get_dcart(
        ellipse_sphere: xr.Dataset,
        coords_sphere: xr.Dataset,
        ) -> xr.Dataset:
    centroid_sphere = ellipse_sphere['centroid']
    centroid_lat = centroid_sphere.sel(component='lat')
    dlat = coords_sphere['dsphere'].sel(component='lat')
    dx = KM_PER_DEGREE_EQ * np.cos(np.deg2rad(centroid_lat + dlat))
    dy = dx.copy()
    dy[:] = KM_PER_DEGREE_EQ
    return xr.DataArray(
        coords = centroid_sphere.coords,
        data = np.array([dy, dx]).transpose()
    )