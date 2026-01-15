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
import pycompo.core.significance_testing as pcsig

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
    feature_var = config['data']['feature_var']
    varlist = [feature_var] + config['data']['wind_vars'] + \
        config['data']['study_vars']
    
    outpath_feat = Path(f"{config['data']['outpath']}/{analysis_idf}/features/")
    outpath_feat.mkdir(parents=True, exist_ok=True)
    outpath_popm = Path(f"{config['data']['outpath']}/{analysis_idf}/popmeans/")
    outpath_popm.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # read in data
    # ------------
    infiles = []
    for var in varlist:
        inpath = Path(config['data']['inpaths'][var])
        in_pattern = f"{config['exp']}_tropical_{var}_*.nc"
        infiles.extend(sorted([str(f) for f in inpath.rglob(in_pattern)]))
    dset = xr.open_mfdataset(infiles, parallel=True).squeeze()
    
    n_new_files = 0
    for i in range (len(analysis_times)-1):
        file_time_string = \
            f"{pcutil.np_datetime2file_datestr(analysis_times[i])}-" + \
            f"{pcutil.np_datetime2file_datestr(analysis_times[i+1])}"    
        outfile_feat = Path(f"{analysis_idf}_features_{file_time_string}.nc")
        if (outpath_feat/outfile_feat).exists():
            continue
        else:
            n_new_files += 1


        dset_sample = pcutil.subsample_analysis_data(
            dset, analysis_times[i], analysis_times[i+1], config,
            )
        if config['test']: dset_sample = dset_sample.isel(time=slice(0, 2))
            
        # ----------------------------------------------------------------------
        # precprocessing
        # --------------
        # detrending
        if config['detrend']['switch']:
            dset_sample, feature_var, varlist = \
                pcfilter.detrend_with_hourly_climatology(
                    dset_sample, feature_var, config,
                    )
        for var in varlist: dset_sample[var] = dset_sample[var].compute()
        
        # scale separation
        filter_vars = [feature_var] + config['data']['wind_vars']
        if config['composite']['type'] == 'anomaly':
            filter_vars += config['data']['study_vars']
        dset_filter = pcfilter.get_gaussian_filter_bg_ano(
            dset_sample[filter_vars], **config['filter']
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
            grad_var = f"{feature_var}_ano"
        elif config['composite']['type'] == 'absolute':
            merge_dsets = [
                dset_filter[
                    [f"{var}_bg" for var in config['data']['wind_vars']]
                    ],
                dset_filter[f"{feature_var}_ano"],
                dset_sample,
                ]
            grad_var = feature_var
        dset_sample = xr.merge(merge_dsets)

        # add timelag and calculate gradients
        dset_sample = pcutil.add_timelag_idx_space(
            dset_sample, f"{feature_var}_ano", config['data']['timelag_idx'],
            )
        dset_sample = pccoord.calc_sphere_gradient_laplacian(
            dset_sample, grad_var,
            )
        dset_sample['cell_area'] = get_cells_area(dset_sample)
        dset_sample = dset_sample.sel(
            lat=slice(*config['lat_range']), drop=True
            )
        
        # ----------------------------------------------------------------------
        # calculate population mean for correct Null hypothesis in sigtests
        # -----------------------------------------------------------------
        outfile_popm = Path(
            f"{analysis_idf}_popmeans_alltrops_{file_time_string}.nc"
            )
        popmeans_alltrops = pcsig.calc_popmeans(dset, feature_var)
        popmeans_alltrops.to_netcdf(str(outpath_popm/outfile_popm))

        if config['composite']['rainbelt_subsampling']['switch']:
            outfile_popm = Path(
                f"{analysis_idf}_popmeans_rainbelt_{file_time_string}.nc"
                )
            rainbelt = pccompo.get_rainbelt(
                analysis_times, config, quantile=0.8,
                ).compute()
            if config['test']: rainbelt = rainbelt.isel(time=slice(0, 2))
            popmeans_rainbelt = pcsig.calc_popmeans(
                dset.where(rainbelt), feature_var,
                )
            popmeans_rainbelt.to_netcdf(str(outpath_popm/outfile_popm))

        # ----------------------------------------------------------------------
        # extract and save anomaly features
        # ---------------------------------
        # extract anomaly features per timestep
        features = Parallel(n_jobs=config['parallel']['n_jobs_get_features'])(
            delayed(process_one_timestep_safe)(dset_sample, time, config)
            for time in dset_sample['time']
            )
        
        # merge features per timestep into one file and set global feature id
        features = pcsst.set_global_feature_id(features)
        features = xr.concat(features, dim='feature')
        features.attrs["identifier"] = analysis_idf

        # save feature composite data
        features.to_netcdf(str(outpath_feat/outfile_feat))

        # ----------------------------------------------------------------------
        # clean up
        # --------
        del dset_sample
        del features
        gc.collect()

        if config['test']: break

    if n_new_files == 0:
        (outpath_feat/Path("get_features_finished.txt")).touch()
        print(
            "All feature files already exist, no new ones created.", flush=True,
            )


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
    orig_coords = pccoord.get_coords_orig(data.drop('time'))
    feature_var = config['data']['feature_var']
    if config['composite']['type'] == 'anomaly':
        grad_var = f"{feature_var}_ano"
    elif config['composite']['type'] == 'absolute':
        grad_var = feature_var

    data[f"{feature_var}_feature"], feature_props = pcsst.extract_sst_features(
        data[f"{feature_var}_ano_detect"], **config['feature']
        )
    data, feature_props, feature_data = get_featcen_data_cutouts(
        data, feature_props, feature_var, config['cutout']['search_RadRatio'],
        )
    feature_props = pcwind.calc_feature_bg_wind(
        feature_props, feature_data, config['data']['wind_vars'],
        )
    feature_props['centroid_sphere'] = pccoord._get_centroid_coords(
        orig_coords, feature_props['centroid_idx'],
        )
    
    # coordinate transformation
    feature_ellipse = get_ellipse_params(feature_props, orig_coords)
    feature_data = pccoord.add_featcen_coords(
        orig_coords, feature_data, feature_props, feature_ellipse,
        )
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
