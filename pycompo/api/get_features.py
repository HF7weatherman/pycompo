import gc
import numpy as np
import sys
import traceback
import xarray as xr
import warnings
from joblib import Parallel, delayed
from pathlib import Path
from pandas import date_range

from pyorg.core.geometry import get_cells_area

import pycompo.core.coord as pccoord
import pycompo.core.filter as pcfilter
import pycompo.core.sst_features as pcsst
import pycompo.core.utils as pcutil
import pycompo.core.wind as pcwind

from pycompo.core.composite import get_compo_coords_ds
from pycompo.core.feature_cutout import get_featcen_data_cutouts
from pycompo.core.ellipse import get_ellipse_params

warnings.filterwarnings(action='ignore')

# ------------------------------------------------------------------------------
def main():
    # read in settings
    config_file = sys.argv[1]
    config = pcutil.read_yaml_config(config_file)

    start_time = config['data']['analysis_time'][0]
    end_time = config['data']['analysis_time'][1]
    analysis_times = [
        np.datetime64(t) for t in date_range(
            np.datetime64(start_time), np.datetime64(end_time), freq='MS',
            )
        ]
    analysis_idf = f"{config['exp']}_{config['pycompo_name']}"
    
    for i in range (len(analysis_times)-1):
        file_time_string = \
            f"{pcutil.np_datetime2file_datestr(analysis_times[i])}-" + \
            f"{pcutil.np_datetime2file_datestr(analysis_times[i+1])}"
        
        # ----------------------------------------------------------------------
        # read in data
        # ------------
        feature_var = config['data']['feature_var']
        varlist = [feature_var] + config['data']['wind_vars'] + \
            config['data']['study_vars']
        infiles = []
        for var in varlist:
            inpath = Path(config['data']['inpaths'][var])
            in_pattern = f"{config['exp']}_tropical_{var}_{file_time_string}.nc"
            infiles.extend(sorted([str(f) for f in inpath.rglob(in_pattern)]))
        dset = xr.open_mfdataset(infiles, parallel=True).squeeze()
        if config['test']: dset = dset.isel(time=slice(0, 2))
            
        # ----------------------------------------------------------------------
        # precprocessing
        # --------------
        # detrending
        if config['detrend']['switch']:
            # Read in data for building the climatology
            inpath = Path(config['data']['inpaths'][feature_var])
            in_pattern = f"{config['exp']}_tropical_{feature_var}_*.nc"
            infiles = sorted([str(f) for f in inpath.rglob(in_pattern)])
            dset_clim = xr.open_mfdataset(infiles, parallel=True).squeeze()
            
            # Detrend dataset with multiyear monthly climatology
            climatology = pcfilter.build_hourly_climatology(
                dset_clim, clim_baseyear=str(config['detrend']['clim_baseyear'])
                )
            rolling_climatology = pcutil.circ_roll_avg(
                climatology, config['detrend']['clim_avg_days'],
                config['data']['spd'],
                )
            dset[f'{feature_var}_detrend'] = \
                dset[feature_var] - rolling_climatology[feature_var]
            feature_var = f'{feature_var}_detrend'
            varlist = [feature_var] + config['data']['wind_vars']

        for var in varlist: dset[var] = dset[var].compute()
        
        # scale separation
        filter_vars = [feature_var] + config['data']['wind_vars']
        if config['composite']['type'] == 'anomaly':
            filter_vars += config['data']['study_vars']
        dset_filter = pcfilter.get_gaussian_filter_bg_ano(
            dset[filter_vars], **config['filter']
            )

        if config['composite']['type'] == 'anomaly':
            merge_dsets = [
                dset_filter[
                    [f"{var}_bg" for var in config['data']['wind_vars']]
                    ],
                dset_filter[
                    [f"{var}_ano" for var in filter_vars]
                    ],
                ]
        elif config['composite']['type'] == 'absolute':
            merge_dsets = [
                dset_filter[
                    [f"{var}_bg" for var in config['data']['wind_vars']]
                    ],
                dset_filter[f"{feature_var}_ano"],
                dset,
                ]
        dset = xr.merge(merge_dsets)

        if config['composite']['type'] == 'anomaly':
            dset = pccoord.calc_sphere_gradient_laplacian(dset, f'{feature_var}_ano')
        elif config['composite']['type'] == 'absolute':
            dset = pccoord.calc_sphere_gradient_laplacian(dset, f'{feature_var}')
        dset['cell_area'] = get_cells_area(dset)
        dset = dset.sel(lat=slice(*config['lat_range']), drop=True)

        # ----------------------------------------------------------------------
        # extract and save anomaly features
        # ---------------------------------
        # extract anomaly features per timestep
        features = Parallel(n_jobs=config['parallel']['n_jobs_get_features'])(
            delayed(process_one_timestep_safe)(dset, time, config)
            for time in dset['time']
            )
        
        # merge features per timestep into one file and set global feature id
        features = pcsst.set_global_feature_id(features)
        features = xr.concat(features, dim='feature')
        features.attrs["identifier"] = analysis_idf

        # save feature composite data
        outpath = Path(f"{config['data']['outpath']}/{analysis_idf}/features/")
        outpath.mkdir(parents=True, exist_ok=True)
        outfile = Path(f"{analysis_idf}_features_{file_time_string}.nc")
        features.to_netcdf(str(outpath/outfile))

        # ----------------------------------------------------------------------
        # clean up
        # --------
        del dset
        del features
        gc.collect()

        if config['test']: break


def process_one_timestep_safe(
        dset: xr.Dataset,
        time: np.datetime64,
        config: dict,
        ) -> xr.Dataset:
    try:
        return process_one_timestep(dset, time, config)
    except Exception as e:
        error_time = time.values if hasattr(time, 'values') else time
        print(f"Error at time={error_time}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


def process_one_timestep(
        dset: xr.Dataset,
        time: np.datetime64,
        config: dict,
        ) -> xr.Dataset:
    data = dset.sel(time=time)
    feature_var = config['data']['feature_var']
    data[f"{feature_var}_feature"], feature_props = pcsst.extract_sst_features(
        data[f"{feature_var}_ano"], **config['feature']
        )
    data, feature_props, feature_data = get_featcen_data_cutouts(
        data, feature_props, feature_var, config['cutout']['search_RadRatio'],
        )
    feature_props = pcwind.calc_feature_bg_wind(
        feature_props, feature_data, config['data']['wind_vars'],
        )
    
    # coordinate transformation
    orig_coords = pccoord.get_coords_orig(data)
    feature_ellipse = get_ellipse_params(feature_props, orig_coords)
    feature_data = pccoord.add_featcen_coords(
        orig_coords, feature_data, feature_props, feature_ellipse,
        )
    if config['composite']['type'] == 'anomaly':
        grad_var = f"{feature_var}_ano"
    elif config['composite']['type'] == 'absolute':
        grad_var = feature_var
    feature_data = pcwind.add_wind_grads(feature_data, feature_props, grad_var)
    feature_data = pcwind.add_rotate_winds(feature_data, feature_props)

    print_time = time.values if hasattr(time, 'values') else time
    print(f"{print_time}: {feature_data[0].data_vars}",
          file=sys.stderr, flush=True)
    
    # remapping to composite coordinate and creating consistent output array
    feature_compo_data = get_compo_coords_ds(feature_data, config)
    feature_props = feature_props.where(
        feature_props['feature_id'].isin(feature_compo_data['feature_id']),
        drop=True,
    )
    return xr.merge([feature_props, feature_compo_data])
    

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
