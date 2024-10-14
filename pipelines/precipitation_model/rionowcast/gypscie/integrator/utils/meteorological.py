# -*- coding: utf-8 -*-
# pylint: disable= invalid-name, inconsistent-return-statements, broad-except
"""
ADD
"""

import numpy as np
import pandas as pd
from metpy.calc import wind_components
from metpy.units import units


def cyclic_time_encoding(s: pd.Series):
    """Transforms a datetime series into a cyclic encoding of the time of day.

    Args:
        s (pd.Series): a pandas series with datetime values

    Returns:
        tuple(pandas.Series, pandas.Series): a tuple of two
        pandas series with the sine and cosine encoding of the time of day
    """
    hour_float = s.dt.hour + s.dt.minute / 60.0
    return cyclic_encoding(hour_float, 24.0)


def cyclic_month_encoding(s: pd.Series):
    """Transforms a datetime series into a cyclic encoding of the month of year.

    Args:
        s (pd.Series): a pandas series with datetime values

    Returns:
        tuple(pandas.Series, pandas.Series): a tuple of two pandas series
         with the sine and cosine encoding of the month of year
    """
    month_float = s.dt.month.astype("float")
    return cyclic_encoding(month_float, 12.0)


def cyclic_encoding(s: pd.Series, max_value):
    """Transforms a series into a cyclic encoding.

    Args:
        s (pd.Series): a pandas series with values

    Returns:
        tuple(pandas.Series, pandas.Series): a tuple of two
                pandas series with the sine and cosine encoding
    """
    s_float = s.astype("float")
    s_sin = np.sin(2.0 * np.pi * s_float / max_value)
    s_cos = np.cos(2.0 * np.pi * s_float / max_value)
    return s_sin, s_cos


def cyclic_wind_encoding(wind_speed: pd.Series, wind_direction: pd.Series):
    """Calculates U and V wind vector components from the speed and direction.

    Args:
        wind_speed (pd.Series): wind speed series

        wind_direction (pd.Series): wind direction series

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): a tuple of two numpy.ndarray
        with U and V wind components
    """
    wind_u, wind_v = wind_components(
        wind_speed.to_numpy() * units.meter_per_second,
        wind_direction.to_numpy() * units.deg,
    )
    return wind_u.magnitude, wind_v.magnitude
