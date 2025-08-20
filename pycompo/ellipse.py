import numpy as np
from typing import Tuple



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