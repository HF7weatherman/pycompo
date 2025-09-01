import gc
import numpy as np
import sys
import xarray as xr
import warnings
from pathlib import Path
from pandas import date_range

from pyorg.core.geometry import get_cells_area

import pycompo.core.coord as pccoord
import pycompo.core.filter as pcfilter
import pycompo.core.utils as pcutil

from pycompo.core.feature_cutout import get_featcen_data_cutouts
from pycompo.core.ellipse import get_ellipse_params
from pycompo.core.sst_features import extract_sst_features
from pycompo.core.wind import calc_feature_bg_wind

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
    
    for i in range (len(analysis_times)-1):
        # ----------------------------------------------------------------------
        # read in data
        # ------------
        feature_var = config['data']['feature_var']
        varlist = [feature_var] + config['data']['wind_vars']
        time_string = \
            f"{pcutil.np_datetime2file_datestr(analysis_times[i])}-" + \
            f"{pcutil.np_datetime2file_datestr(analysis_times[i+1])}"
        infiles = []
        for var in varlist:
            inpath = Path(config['data']['inpaths'][var])
            in_pattern = f"{config['exp']}_tropical_{var}_{time_string}.nc"
            infiles.extend(sorted([str(f) for f in inpath.rglob(in_pattern)]))
        dset = xr.open_mfdataset(infiles, parallel=True).squeeze()
        dset['cell_area'] = get_cells_area(dset)

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
        dset = xr.merge([
            dset,
            pcfilter.get_gaussian_filter_bg_ano(
                dset[feature_var], **config['filter'],
                )
            ])
        dset = pccoord.calc_sphere_gradient_laplacian(
            dset, f'{feature_var}_ano',
            )
        dset = dset.sel(lat=slice(*config['lat_range']), drop=True)

        # ----------------------------------------------------------------------
        # extract anomaly features
        # ------------------------
        # detect anomaly features and get corresponding data cutouts
        dset[f"{feature_var}_feature"], feature_props = extract_sst_features(
            dset[f"{feature_var}_ano"], **config['feature']
            )
        dset, feature_props, feature_data = get_featcen_data_cutouts(
            dset, feature_props, feature_var,
            config['cutout']['search_RadRatio'],
            )
        feature_props = calc_feature_bg_wind(
            feature_props, feature_data, config['data']['wind_vars'],
            )
        
        # coordinate transformation
        orig_coords = pccoord.get_coords_orig(dset)
        feature_ellipse = get_ellipse_params(feature_props, orig_coords)
        feature_data = pccoord.add_featcen_coords(
            orig_coords, feature_data, feature_props, feature_ellipse,
            )
        
        # clean up
        del dset
        del feature_ellipse
        gc.collect()

    # TODO: MERGE TOGETHER ALL MONTHS IF POSSIBLE

    # ----------------------------------------------------------------------
    # write output
    # ------------
    analysis_identifier = f"{config['exp']}_{config['pycompo_name']}"

    # save feature props
    outpath = Path(f"{config['data']['outpath']}/{analysis_identifier}/")
    outpath.mkdir(parents=True, exist_ok=True)
    outfile = Path(
        f"{analysis_identifier}_feature_props_{start_time}-{end_time}.nc"
        )
    feature_props.attrs["identifier"] = analysis_identifier
    feature_props.to_netcdf(str(outpath/outfile))

    # save feature data
    outpath = Path(
        f"{config['data']['outpath']}/{analysis_identifier}/" + \
        f"{analysis_identifier}_feature_data_{start_time}-{end_time}/"
        )
    outpath.mkdir(parents=True, exist_ok=True)

    for data in feature_data[:3]:
        feature_id = data['feature_id'].values
        outfile = Path(
            f"{analysis_identifier}_feature_data_{start_time}-{end_time}_" + \
            f"feature{feature_id}.nc"
            )
        data.attrs["identifier"] = analysis_identifier
        data.drop(['height_2', 'uas', 'vas']).to_netcdf(str(outpath/outfile))


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()