import numpy as np
import sys
import warnings
import xarray as xr
from pathlib import Path
from pandas import date_range

import pycompo.core.composite as pccompo
import pycompo.core.utils as pcutils
import pycompo.core.significance_testing as pcsig

warnings.filterwarnings(action='ignore')


# ------------------------------------------------------------------------------
def main():
    # read in settings
    config_file = sys.argv[1]
    config = pcutils.read_yaml_config(config_file)

    start_time = config['data']['analysis_time'][0]
    end_time = config['data']['analysis_time'][1]
    analysis_times = [
        np.datetime64(t) for t in date_range(
            np.datetime64(start_time), np.datetime64(end_time), freq='MS',
            )
        ]
    analysis_idf = f"{config['exp']}_{config['pycompo_name']}"
    subgroup_vars = pcutils.get_subgroup_vars_dict(config)

    
    # --------------------------------------------------------------------------
    # preparations for composite subsampling
    # --------------------------------------
    # load feature_props
    inpath = Path(f"{config['data']['outpath']}/{analysis_idf}/")
    infile = Path(f"{analysis_idf}_feature_props_alltrops_all.nc")
    feature_props_alltrops = xr.open_dataset(str(inpath/infile))

    # build rainbelt if necessary
    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt = pccompo.get_rainbelt(analysis_times, config, quantile=0.8)
        rainbelt = rainbelt.compute()
        infile = Path(f"{analysis_idf}_feature_props_rainbelt_all.nc")
        feature_props_rainbelt = xr.open_dataset(str(inpath/infile))


    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    alltrops_subgroup_compo = {var: [] for var in subgroup_vars.keys()}
    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt_subgroup_compo = {var: [] for var in subgroup_vars.keys()}

    inpath = Path(f"{config['data']['outpath']}/{analysis_idf}/features/")
    for i in range (len(analysis_times)-1):
        # read in data
        file_timestr = \
            f"{pcutils.np_datetime2file_datestr(analysis_times[i])}-" + \
            f"{pcutils.np_datetime2file_datestr(analysis_times[i+1])}"
        infile = inpath/Path(f"{analysis_idf}_features_{file_timestr}.nc")
        features_alltrops = xr.open_dataset(infile).compute()
        
        # subsample based on BG conditions
        for var, thresholds in subgroup_vars.items():
            alltrops_subgroup_compo[var].append(
                pccompo.get_subgroup_compos_per_ts(
                    features_alltrops,
                    feature_props_alltrops[var],
                    thresholds,
                    )
                )

        # Precipitation-based geographic subsampling
        if config['composite']['rainbelt_subsampling']['switch']:
            features_rainbelt = pccompo.sample_features_geomask(
                features_alltrops, rainbelt,
                )
            
            # subsample based on BG conditions
            for var, thresholds in subgroup_vars.items():
                rainbelt_subgroup_compo[var].append(
                    pccompo.get_quartile_compos_per_ts(
                        features_rainbelt,
                        feature_props_rainbelt[var],
                        thresholds,
                        )
                    )
                
    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    outpath = Path(f"{config['data']['outpath']}/{analysis_idf}/")
    outpath.mkdir(parents=True, exist_ok=True)

    for var in subgroup_vars:
        alltrops_subgroup_compo[var] = pccompo.get_full_quartile_compos(
            alltrops_subgroup_compo[var],
        )
        outfile = Path(f"{analysis_idf}_composite_alltrops_{var}_quartiles.nc")
        alltrops_subgroup_compo[var].to_netcdf(str(outpath/outfile))
            
    if config['composite']['rainbelt_subsampling']['switch']:
        for var in subgroup_vars:
            rainbelt_subgroup_compo[var] = pccompo.get_full_quartile_compos(
                rainbelt_subgroup_compo[var],
                )
            outfile = Path(f"{analysis_idf}_composite_rainbelt_{var}_quartiles.nc")
            rainbelt_subgroup_compo[var].to_netcdf(str(outpath/outfile))

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()