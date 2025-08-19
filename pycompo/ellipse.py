import numpy as np
from typing import Tuple

def calc_polar_angle_rad(
        x: float
        ) -> float:
    return (np.pi/2 - x) if x > 0 else (-np.pi/2 - x)


def calc_axis_major_end(
        axis_major_length: float,
        polar_angle_rad: float,
        ) -> Tuple[float, float]:
    x_end = axis_major_length / 2 * np.cos(polar_angle_rad)
    y_end = axis_major_length / 2 * np.sin(polar_angle_rad)
    return (x_end, y_end)


def calc_axis_minor_end(
        axis_minor_length: float,
        polar_angle_rad: float,
        ) -> Tuple[float, float]:
    x_end = axis_minor_length / 2 * np.cos(polar_angle_rad + np.pi / 2)
    y_end = axis_minor_length / 2 * np.sin(polar_angle_rad + np.pi / 2)
    return (x_end, y_end)