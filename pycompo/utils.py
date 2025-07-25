import xarray as xr

def build_circular_rolling_avg(
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