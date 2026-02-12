import gc
import numpy as np
import sys
import traceback
import xarray as xr
import warnings
from joblib import Parallel, delayed
from pathlib import Path

from grid_toolbox.basic_latlon import get_cells_area

import pycompo.core.coord as pccoord
import pycompo.core.filter as pcfilter
import pycompo.core.sst_features as pcsst
import pycompo.core.utils as pcutil
import pycompo.core.wind as pcwind

from pycompo.core.composite import get_rainbelt, get_compo_coords_ds
from pycompo.core.ellipse import get_ellipse_params
from pycompo.core.feature_cutout import get_featcen_data_cutouts
from pycompo.core.sigtest import calc_popmeans

warnings.filterwarnings(action='ignore')

# ------------------------------------------------------------------------------
def main():
    config_file = sys.argv[1]
    config = pcutil.read_yaml_config(config_file)

    ana_times = pcutil.create_analysis_times(config)
    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    feat_var = config['data']['feature_var']
    wind_vars = config['data']['wind_vars']
    study_vars = config['data']['study_vars']
    varlist = [feat_var] + wind_vars + study_vars
    
    opath_feat = Path(f"{config['data']['outpath']}/{ana_idf}/features/")
    opath_feat.mkdir(parents=True, exist_ok=True)
    opath_popm = Path(f"{config['data']['outpath']}/{ana_idf}/popmeans/")
    opath_popm.mkdir(parents=True, exist_ok=True)

    ifiles = []
    for var in varlist:
        ipath = Path(config['data']['inpaths'][var])
        ipattern = f"{config['exp']}_tropical_{var}_*.nc"
        ifiles.extend(sorted([str(f) for f in ipath.rglob(ipattern)]))
    dset = xr.open_mfdataset(ifiles, parallel=False).squeeze()
    
    # --------------------------------------------------------------------------
    # main logical loop
    # -----------------
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ofile_feat = opath_feat/Path(f"{ana_idf}_features_{fdate_str}.nc")
        ofile_popm_at = opath_popm/Path(
            f"{ana_idf}_popmeans_alltrops_{fdate_str}.nc"
            )
        ofiles_exist = all([ofile_feat.exists(), ofile_popm_at.exists()])
        if ofiles_exist: continue

        dsample = pcutil.subsample_data(dset, start_time, end_time, config)
        if config['test']: dsample = dsample.isel(time=slice(0, 2))

        if config['detrend']['switch']:
            dsample, feat_var, varlist = \
                pcfilter.detrend_with_hourly_climatology(
                    dsample, feat_var, config,
                    )
        for var in varlist: dsample[var] = dsample[var].compute()
        
        # ----------------------------------------------------------------------
        # scale separation
        # ----------------
        filter_vars = [feat_var] + wind_vars
        if config['composite']['type'] == 'anomaly': filter_vars += study_vars
        dfilter = pcfilter.get_gaussian_filter_bg_ano(
            dsample[filter_vars], **config['filter']
            )

        if config['composite']['type'] == 'anomaly':
            merge_dsets = [
                dfilter[[f"{var}_bg" for var in wind_vars]],
                dfilter[[f"{var}_ano" for var in filter_vars]],
                ]
            grad_var = f"{feat_var}_ano"
        elif config['composite']['type'] == 'absolute':
            merge_dsets = [
                dfilter[[f"{var}_bg" for var in wind_vars]],
                dfilter[f"{feat_var}_ano"],
                dsample,
                ]
            grad_var = feat_var
        
        if 'ts_bg' in dfilter and 'tas_bg' in dfilter:
            dfilter['tas-ts_bg'] = dfilter['tas_bg'] - dfilter['ts_bg']
            merge_dsets.append(dfilter['tas-ts_bg'])
        if 'sfc_rho_bg' in dfilter:
            merge_dsets.append(dfilter['sfc_rho_bg'])
        
        dsample = xr.merge(merge_dsets)

        # add timelag and calculate gradients
        dsample = pcutil.add_timelag_idx_space(
            dsample, f"{feat_var}_ano", config['data']['timelag_idx'],
            )
        dsample = pccoord.calc_sphere_gradient_laplacian(dsample, grad_var)
        if 'ps' in config['data']['study_vars']:
            if config['composite']['type'] == 'anomaly':
                dsample = pccoord.calc_sphere_laplacian(dsample, 'ps_ano')
            elif config['composite']['type'] == 'absolute':
                dsample = pccoord.calc_sphere_laplacian(dsample, 'ps')
                
        dsample['cell_area'] = get_cells_area(dsample)
        dsample = dsample.sel(lat=slice(*config['lat_range']), drop=True)
        
        # ----------------------------------------------------------------------
        # calculate population mean for correct Null hypothesis in sigtests
        # -----------------------------------------------------------------
        if config['composite']['rainbelt_subsampling']['switch']:
            ofile_popm_rb = Path(f"{ana_idf}_popmeans_rainbelt_{fdate_str}.nc")
            rainbelt = get_rainbelt(ana_times, config, quantile=0.8).compute()
            if config['test']: rainbelt = rainbelt.isel(time=slice(0, 2))
            popmeans_rb = calc_popmeans(dsample.where(rainbelt), feat_var)
            popmeans_rb.to_netcdf(str(opath_popm/ofile_popm_rb))

        popmeans_at = calc_popmeans(dsample, feat_var)
        popmeans_at.to_netcdf(str(ofile_popm_at))

        # ----------------------------------------------------------------------
        # extract and save anomaly features
        # ---------------------------------
        features = Parallel(n_jobs=config['parallel']['n_jobs_get_features'])(
            delayed(process_one_timestep_safe)(dsample, time, config)
            for time in dsample['time']
            )
        features = pcsst.set_global_feature_id(features)
        features = xr.concat(features, dim='feature')
        features.attrs["identifier"] = ana_idf
        features.to_netcdf(str(ofile_feat))

        # clean up
        del dsample
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
    orig_coords = pccoord.get_coords_orig(data.drop('time'))
    feat_var = config['data']['feature_var']
    if config['composite']['type'] == 'anomaly':
        grad_var = f"{feat_var}_ano"
    elif config['composite']['type'] == 'absolute':
        grad_var = feat_var

    data[f"{feat_var}_feature"], featprops = pcsst.extract_sst_features(
        data[f"{feat_var}_ano_detect"], **config['feature']
        )
    data, featprops, featdata = get_featcen_data_cutouts(
        data, featprops, feat_var, config['cutout']['search_RadRatio'],
        )
    featprops = pcwind.calc_feature_bg_wind(
        featprops, featdata, config['data']['wind_vars'],
        )
    featprops = pcsst.add_more_feature_props(featprops, featdata, orig_coords)
    
    # coordinate transformation
    feature_ellipse = get_ellipse_params(featprops, orig_coords)
    featdata = pccoord.add_featcen_coords(
        orig_coords, featdata, featprops, feature_ellipse,
        )
    featdata = pcwind.add_wind_grads(featdata, featprops, grad_var)
    featdata = pcwind.add_rotate_winds(featdata, featprops)

    print_time = time.values if hasattr(time, 'values') else time
    print(f"{print_time}: {featdata[0].data_vars}",
          file=sys.stderr, flush=True)
    
    # remapping to composite coordinate and creating consistent output array
    compo_data = get_compo_coords_ds(featdata, config)
    featprops = featprops.where(
        featprops['feature_id'].isin(compo_data['feature_id']), drop=True,
    )
    return xr.merge([featprops, compo_data])
    

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
