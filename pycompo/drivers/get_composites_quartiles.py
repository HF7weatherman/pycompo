import numpy as np
import sys
import warnings
import xarray as xr
from pathlib import Path
from pandas import date_range

import pycompo.core.composite as pccompo
import pycompo.core.utils as pcutils
import pycompo.core.sigtest as pcsig

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
    subgroup_vars = config['composite']['subgroup_vars']

    
    # --------------------------------------------------------------------------
    # preparations for composite subsampling
    # --------------------------------------
    # load feature_props
    inpath = Path(f"{config['data']['outpath']}/{analysis_idf}/")
    infile = Path(f"{analysis_idf}_feature_props_alltrops_all.nc")
    feature_props_alltrops = xr.open_dataset(str(inpath/infile))
    feature_props_alltrops_quartiles = feature_props_alltrops.quantile(
        [0.25, 0.50, 0.75]
        )

    # build rainbelt if necessary
    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt = pccompo.get_rainbelt(analysis_times, config, quantile=0.8)
        rainbelt = rainbelt.compute()
        infile = Path(f"{analysis_idf}_feature_props_rainbelt_all.nc")
        feature_props_rainbelt = xr.open_dataset(str(inpath/infile))
        feature_props_rainbelt_quartiles = feature_props_rainbelt.quantile(
            [0.25, 0.50, 0.75]
            )


    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    alltrops_quartile_compo = {var: [] for var in subgroup_vars}
    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt_quartile_compo = {var: [] for var in subgroup_vars}

    inpath = Path(f"{config['data']['outpath']}/{analysis_idf}/features/")
    for i in range (len(analysis_times)-1):
        # read in data
        file_timestr = \
            f"{pcutils.np_datetime2file_datestr(analysis_times[i])}-" + \
            f"{pcutils.np_datetime2file_datestr(analysis_times[i+1])}"
        infile = inpath/Path(f"{analysis_idf}_features_{file_timestr}.nc")
        features_alltrops = xr.open_dataset(infile).compute()
        
        # subsample based on BG conditions
        for var in subgroup_vars:
            alltrops_quartile_compo[var].append(
                pccompo.get_quartile_compos_per_ts(
                    features_alltrops,
                    feature_props_alltrops_quartiles[var], var,
                    )
                )

        # Precipitation-based geographic subsampling
        if config['composite']['rainbelt_subsampling']['switch']:
            features_rainbelt = pccompo.sample_features_geomask(
                features_alltrops, rainbelt,
                )
            
            # subsample based on BG conditions
            for var in subgroup_vars:
                rainbelt_quartile_compo[var].append(
                    pccompo.get_quartile_compos_per_ts(
                        features_rainbelt,
                        feature_props_rainbelt_quartiles[var], var,
                        )
                    )
                
    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    outpath = Path(f"{config['data']['outpath']}/{analysis_idf}/")
    outpath.mkdir(parents=True, exist_ok=True)

    for var in subgroup_vars:
        alltrops_quartile_compo[var] = pccompo.get_full_quartile_compos(
            alltrops_quartile_compo[var],
        )
        outfile = Path(f"{analysis_idf}_composite_alltrops_{var}_quartiles.nc")
        alltrops_quartile_compo[var].to_netcdf(str(outpath/outfile))
            
    if config['composite']['rainbelt_subsampling']['switch']:
        for var in subgroup_vars:
            rainbelt_quartile_compo[var] = pccompo.get_full_quartile_compos(
                rainbelt_quartile_compo[var],
                )
            outfile = Path(f"{analysis_idf}_composite_rainbelt_{var}_quartiles.nc")
            rainbelt_quartile_compo[var].to_netcdf(str(outpath/outfile))

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()