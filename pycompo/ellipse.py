import numpy as np
import xarray as xr
from typing import Tuple

import pycompo.coord as pccoord


def calc_polar_angle_rad(
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


def calc_axis_major_end(
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
    x_end = axis_major_length / 2 * np.cos(polar_angle_rad)
    y_end = axis_major_length / 2 * np.sin(polar_angle_rad)
    return (x_end, y_end)


def calc_axis_minor_end(
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
    x_end = axis_minor_length / 2 * np.cos(polar_angle_rad + np.pi / 2)
    y_end = axis_minor_length / 2 * np.sin(polar_angle_rad + np.pi / 2)
    return (x_end, y_end)


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