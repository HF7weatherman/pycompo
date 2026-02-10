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
    rainbelt_switch = config['composite']['rainbelt_subsampling']['switch']
    subgroup_vars = pcutil.get_subgroup_vars_dict(config)

    opath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    opath.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # preparations for composite subsampling
    # --------------------------------------
    ipath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    ifile = Path(f"{ana_idf}_feature_props_alltrops_all.nc")
    featprops_at = xr.open_dataset(str(ipath/ifile))

    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt = pccompo.get_rainbelt(ana_times, config, quantile=0.8)
        rainbelt = rainbelt.compute()
        ifile = Path(f"{ana_idf}_feature_props_rainbelt_all.nc")
        featprops_rb = xr.open_dataset(str(ipath/ifile))


    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    compo_bin_at = {var: [] for var in subgroup_vars.keys()}
    var_bin_at = {var: [] for var in subgroup_vars.keys()}
    N_feats_bin_at = {var: [] for var in subgroup_vars.keys()}
    if rainbelt_switch:
        compo_bin_rb = {var: [] for var in subgroup_vars.keys()}
        var_bin_rb = {var: [] for var in subgroup_vars.keys()}
        N_feats_bin_rb = {var: [] for var in subgroup_vars.keys()}

    ipath = Path(f"{config['data']['outpath']}/{ana_idf}/features/")
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ifile = ipath/Path(f"{ana_idf}_features_{fdate_str}.nc")
        feats_at = xr.open_dataset(ifile).compute()
        
        for var, thresholds in subgroup_vars.items():
            compo_bin, var_bin, N_feats_bin = \
                pccompo.get_binned_features(feats_at, thresholds, var)
            compo_bin_at[var].append(compo_bin)
            var_bin_at[var].append(var_bin)
            N_feats_bin_at[var].append(N_feats_bin)

        if rainbelt_switch:
            feats_rb = pccompo.sample_features_geomask(feats_at, rainbelt)
            for var, thresholds in subgroup_vars.items():
                compo_bin, var_bin, N_feats_bin = \
                    pccompo.get_binned_features(feats_rb, thresholds, var)
                compo_bin_rb[var].append(compo_bin)
                var_bin_rb[var].append(var_bin)
                N_feats_bin_rb[var].append(N_feats_bin)
                
    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    for var in subgroup_vars:
        subcompo_at[var] = pccompo.get_full_quartile_compos(subcompo_at[var])
        ofile = Path(f"{ana_idf}_composite_alltrops_{var}_quartiles.nc")
        subcompo_at[var].to_netcdf(str(opath/ofile))
            
    if rainbelt_switch:
        for var in subgroup_vars:
            subcompo_rb[var] = pccompo.get_full_quartile_compos(subcompo_rb[var])
            ofile = Path(f"{ana_idf}_composite_rainbelt_{var}_quartiles.nc")
            subcompo_rb[var].to_netcdf(str(opath/ofile))

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
