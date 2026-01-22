import sys
import warnings
import xarray as xr
from pathlib import Path

import pycompo.core.utils as pcutil
from pycompo.core.composite import get_rainbelt, sample_features_geomask
from pycompo.core.sst_features import set_global_feature_id

warnings.filterwarnings(action='ignore')


def run_combine_feature_props(config: dict, outfiles: dict) -> None:
    KEEP_PROPS = [
        'radius_km', 'area_km2', 'bg_uas', 'bg_vas', 'bg_sfcwind',
        'bg_sfcwind_dir', 'ts_ano_detect_mean', 'axis_major_length_idx', 
        'axis_minor_length_idx', 'orientation_idx', 'centroid_sphere',
        ]
    ana_times = pcutil.create_analysis_times(config)
    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    ipath = Path(f"{config['data']['outpath']}/{ana_idf}/features/")
    rainbelt_switch = config['composite']['rainbelt_subsampling']['switch']

    if rainbelt_switch:
        rainbelt = get_rainbelt(ana_times, config, quantile=0.8).compute()

    # --------------------------------------------------------------------------
    # create a single feature composite
    # ---------------------------------
    featprops, featprops_rainbelt = [], []
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ifile = ipath/Path(f"{ana_idf}_features_{fdate_str}.nc")
        features = xr.open_dataset(ifile).compute()
        featprops.append(features[KEEP_PROPS])

        if rainbelt_switch:
            features_rainbelt = sample_features_geomask(features, rainbelt)
            featprops_rainbelt.append(features_rainbelt[KEEP_PROPS])

        # Basin-based geographic subsampling
        # TODO: Implement basin-based geographical subsampling
        
    featprops = xr.concat(set_global_feature_id(featprops), dim='feature')
    featprops_rainbelt = xr.concat(
        set_global_feature_id(featprops_rainbelt), dim='feature',
        )

    featprops.to_netcdf(str(outfiles['alltrops']))
    if rainbelt_switch:
        featprops_rainbelt.to_netcdf(str(outfiles['rainbelt']))


# ------------------------------------------------------------------------------
# run script
# ----------
config_file = sys.argv[1]
config = pcutil.read_yaml_config(config_file)

ana_idf = f"{config['exp']}_{config['pycompo_name']}"
opath = Path(f"{config['data']['outpath']}/{ana_idf}/")
opath.mkdir(parents=True, exist_ok=True)

ofiles = {"alltrops": opath/Path(f"{ana_idf}_feature_props_alltrops_all.nc")}
ofiles_exist = ofiles['alltrops'].exists()
if config['composite']['rainbelt_subsampling']['switch']:
    ofiles['rainbelt'] = opath/Path(f"{ana_idf}_feature_props_rainbelt_all.nc")
    ofiles_exist = ofiles_exist and ofiles['rainbelt'].exists()

if not ofiles_exist:
    print("Combining feature properties from all time steps ...")
    run_combine_feature_props(config, ofiles)