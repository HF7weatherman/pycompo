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


def roll_avg(
        dset: xr.DataArray | xr.Dataset,
        clim_avg_days: int | float,
        spd: int,
        ) -> xr.DataArray | xr.Dataset:
    window_size = int(clim_avg_days*spd + 1)
    return dset.rolling(time=window_size, center=True).mean()


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


def add_metric_altitude(
        compo: xr.Dataset,
        config: dict,
        ) -> xr.Dataset:
    if config['exp'] == "ngc5004":
        # add altitude coordinate aligned with the existing 'height' dimension
        altitude = np.array(
            [73727.2891, 71235.4219, 68814.0938, 66443.7891, 64140.6133,
             61902.7656, 59728.5625, 57616.4141, 55560.3828, 53551.6094,
             51589.1094, 49671.9727, 47803.7930, 45991.2344, 44233.1914,
             42528.6211, 40876.5273, 39275.9688, 37726.0352, 36225.8750,
             34774.6602, 33371.6172, 32015.9883, 30707.0684, 29444.1738,
             28226.6543, 27053.8906, 25927.4180, 24851.4746, 23829.5391,
             22864.3750, 21955.9531, 21100.9395, 20296.1895, 19538.7520,
             18825.8438, 18154.8477, 17523.2988, 16928.8789, 16369.4053,
             15842.8232, 15347.1982, 14880.7109, 14441.6494, 14022.5205,
             13616.2744, 13216.2744, 12816.2744, 12416.2744, 12016.2744,
             11616.2744, 11216.2744, 10816.2744, 10416.2744, 10016.2744,
              9616.2744,  9216.2744,  8816.2744,  8416.2744,  8016.2739,
              7616.2739,  7216.2739,  6816.2739,  6416.2739,  6016.2739,
              5616.2739,  5220.4321,  4835.6538,  4464.6704,  4107.4575,
              3764.0017,  3434.3030,  3118.3750,  2816.2468,  2527.9646,
              2253.5935,  1993.2206,  1746.9575,  1514.9450,  1297.3583,
              1094.4139,   906.3797,   733.5876,   576.4535,   435.5065,
               311.4361,   205.1747,   118.0616,    52.2459,    12.5000,
            ]
        )
        altitude = altitude[[int(idx-1) for idx in compo['height'].values]]
        return compo.assign_coords(altitude=('height', altitude))
    else:
        raise ValueError(
            f"Conversion to metric altitude is currently not implemented " +
            f"for experiment {config['exp']}!"
            )