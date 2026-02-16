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
        'bg_sfcwind_dir', 'bg_tas-ts', 'ts_ano_detect_mean',
        'axis_major_length_idx', 'axis_minor_length_idx', 'orientation_idx',
        'centroid_sphere',
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
    featprops_at, featprops_rb = [], []
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ifile = ipath/Path(f"{ana_idf}_features_{fdate_str}.nc")
        features_at = xr.open_dataset(ifile).compute()
        featprops_at.append(features_at[KEEP_PROPS])

        if rainbelt_switch:
            features_rb = sample_features_geomask(features_at, rainbelt)
            featprops_rb.append(features_rb[KEEP_PROPS])
        
    featprops_at = xr.concat(set_global_feature_id(featprops_at), dim='feature')
    pcutil.sort_ds(featprops_at).to_netcdf(str(outfiles['alltrops']))
    if rainbelt_switch:
        featprops_rb = xr.concat(
            set_global_feature_id(featprops_rb), dim='feature',
            )
        pcutil.sort_ds(featprops_rb).to_netcdf(str(outfiles['rainbelt']))


# ------------------------------------------------------------------------------
# run script
# ----------
config = pcutil.read_yaml_config(sys.argv[1])
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
