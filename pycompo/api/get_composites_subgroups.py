import sys
import warnings
import xarray as xr
from pathlib import Path

import pycompo.core.composite as pccompo
import pycompo.core.utils as pcutil

warnings.filterwarnings(action='ignore')


# ------------------------------------------------------------------------------
def main():
    config_file = sys.argv[1]
    config = pcutil.read_yaml_config(config_file)

    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    ana_times = pcutil.create_analysis_times(config)
    subgroup_vars = pcutil.get_subgroup_vars_dict(config)

    opath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    opath.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # preparations for composite subsampling
    # --------------------------------------
    ipath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    ifile = Path(f"{ana_idf}_feature_props_alltrops_all.nc")
    featprops_alltrops = xr.open_dataset(str(ipath/ifile))

    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt = pccompo.get_rainbelt(ana_times, config, quantile=0.8)
        rainbelt = rainbelt.compute()
        ifile = Path(f"{ana_idf}_feature_props_rainbelt_all.nc")
        featprops_rainbelt = xr.open_dataset(str(ipath/ifile))


    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    alltrops_subcompo = {var: [] for var in subgroup_vars.keys()}
    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt_subcompo = {var: [] for var in subgroup_vars.keys()}

    ipath = Path(f"{config['data']['outpath']}/{ana_idf}/features/")
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ifile = ipath/Path(f"{ana_idf}_features_{fdate_str}.nc")
        feats_alltrops = xr.open_dataset(ifile).compute()
        
        # subsample based on BG conditions
        for var, thresholds in subgroup_vars.items():
            alltrops_subcompo[var].append(
                pccompo.get_subgroup_compos_per_ts(
                    feats_alltrops, featprops_alltrops[var], thresholds,
                    )
                )

        # Precipitation-based geographic subsampling
        if config['composite']['rainbelt_subsampling']['switch']:
            feats_rainbelt = pccompo.sample_features_geomask(
                feats_alltrops, rainbelt,
                )
            
            # subsample based on BG conditions
            for var, thresholds in subgroup_vars.items():
                rainbelt_subcompo[var].append(
                    pccompo.get_quartile_compos_per_ts(
                        feats_rainbelt, featprops_rainbelt[var], thresholds,
                        )
                    )
                
    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    for var in subgroup_vars:
        alltrops_subcompo[var] = pccompo.get_full_quartile_compos(
            alltrops_subcompo[var],
        )
        ofile = Path(f"{ana_idf}_composite_alltrops_{var}_quartiles.nc")
        alltrops_subcompo[var].to_netcdf(str(opath/ofile))
            
    if config['composite']['rainbelt_subsampling']['switch']:
        for var in subgroup_vars:
            rainbelt_subcompo[var] = pccompo.get_full_quartile_compos(
                rainbelt_subcompo[var],
                )
            ofile = Path(f"{ana_idf}_composite_rainbelt_{var}_quartiles.nc")
            rainbelt_subcompo[var].to_netcdf(str(opath/ofile))

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()