import numpy as np
import xarray as xr
import warnings
from pathlib import Path
from pandas import date_range

import pycompo.core.composite as pccompo
import pycompo.core.utils as pcutil
from pycompo.core.sst_features import set_global_feature_id

warnings.filterwarnings(action='ignore')

# read in configuration file
config_file = "/home/m/m300738/libs/pycompo/config/settings_ngc5004_pc03.yaml"
config = pcutil.read_yaml_config(config_file)

start_time = config['data']['analysis_time'][0]
end_time = config['data']['analysis_time'][1]
analysis_times = [
    np.datetime64(t) for t in date_range(
        np.datetime64(start_time), np.datetime64(end_time), freq='MS',
        )
    ]
analysis_times = analysis_times[:4]

feature_var = config['data']['feature_var']
analysis_idf = f"{config['exp']}_{config['pycompo_name']}"


# ------------------------------------------------------------------------------
# build rainbelt
# --------------
if config['composite']['rainbelt_subsampling']['switch']:
    rainbelt = pccompo.get_rainbelt(analysis_times, config, quantile=0.8)


# ------------------------------------------------------------------------------
# create a single feature composite
# ---------------------------------
keep_props = [
    'radius_km', 'area_km2', 'bg_uas', 'bg_vas', 'bg_sfcwind', 'bg_sfcwind_dir',
    'ts_ano_mean', 'axis_major_length_idx', 'axis_minor_length_idx',
    'orientation_idx',
    ]

inpath = Path(f"{config['data']['outpath']}/{analysis_idf}/features/")

feature_props = []
feature_props_rainbelt = []
for i in range (len(analysis_times)-1):
    # read in data
    file_timestr = \
        f"{pcutil.np_datetime2file_datestr(analysis_times[i])}-" + \
        f"{pcutil.np_datetime2file_datestr(analysis_times[i+1])}"
    infile = inpath/Path(f"{analysis_idf}_features_{file_timestr}.nc")
    features = xr.open_dataset(infile).compute()
    feature_props.append(features[keep_props])

    # Precipitation-based geographic subsampling
    if config['composite']['rainbelt_subsampling']['switch']:
        features_rainbelt = pccompo.sample_features_geomask(features, rainbelt)
        feature_props_rainbelt.append(features_rainbelt[keep_props])

    # Basin-based geographic subsampling
    # TODO: Implement basin-based geographical subsampling
    
feature_props = xr.concat(set_global_feature_id(feature_props), dim='feature')
feature_props_rainbelt = xr.concat(
    set_global_feature_id(feature_props_rainbelt), dim='feature',
    )

# save feature composite data
outpath = Path(f"{config['data']['outpath']}/{analysis_idf}/")
outpath.mkdir(parents=True, exist_ok=True)

outfile = Path(f"{analysis_idf}_feature_props_all.nc")
feature_props.to_netcdf(str(outpath/outfile))
if config['composite']['rainbelt_subsampling']['switch']:
    outfile = Path(f"{analysis_idf}_feature_props_rainbelt_all.nc")
    feature_props_rainbelt.to_netcdf(str(outpath/outfile))