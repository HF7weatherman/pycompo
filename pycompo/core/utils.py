import math
import numpy as np
import yaml
import xarray as xr


def read_yaml_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def np_datetime2file_datestr(time_np64: np.datetime64) -> str:
    format = '%Y%m%dT%H%M%SZ'
    return time_np64.astype('datetime64[us]').astype('O').strftime(format)


def circ_roll_avg(
        dset: xr.DataArray | xr.Dataset,
        clim_avg_days: int | float,
        spd: int,
        ) -> xr.DataArray | xr.Dataset:
    window_size = int(clim_avg_days*spd + 1)

    # Build pseudo-circular extended array
    extend_len = (window_size - 1) // 2
    extended = xr.concat(
        [dset.isel(time=slice(-extend_len, None)),
         dset,
         dset.isel(time=slice(0, extend_len))
         ], dim='time'
         )
    
    rolling_avg = extended.rolling(time=window_size, center=True).mean()
    result = rolling_avg.isel(
        time=slice(extend_len, extend_len + dset.sizes['time'])
        )
    return result


def round_away_from_zero(x: float) -> int:
    return int(math.copysign(math.ceil(abs(x)), x))