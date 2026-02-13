import sys
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple

import pycompo.core.composite as pccompo
import pycompo.core.utils as pcutil
import pycompo.core.sigtest as pcsig
from pycompo.core.sst_features import calc_asprat_idx

warnings.filterwarnings(action='ignore')


# ------------------------------------------------------------------------------
def main():
    config = pcutil.read_yaml_config(sys.argv[1])
    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    rainbelt_switch = config['composite']['rainbelt_subsampling']['switch']
    subgroup_vars = pcutil.get_subgroup_vars_dict(config)

    opath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    opath.mkdir(parents=True, exist_ok=True)
    ofile_at = {
        'compo': {
            var: Path(f"{ana_idf}_composite_alltrops_{var}_binned.nc")
            for var in subgroup_vars
            },
        'pvalue': {
            var: Path(f"{ana_idf}_pvalue_alltrops_{var}_binned.nc")
            for var in subgroup_vars
            },
    }
    ofiles_exist = all(
        [ofile_at['compo'][var].exists() for var in subgroup_vars]
        )

    if rainbelt_switch:
        ofile_rb = {
            'compo': {
                var: Path(f"{ana_idf}_composite_rainbelt_{var}_binned.nc")
                for var in subgroup_vars
                },
            'pvalue': {
                var: Path(f"{ana_idf}_pvalue_rainbelt_{var}_binned.nc")
                for var in subgroup_vars
                },
        }
        ofiles_exist = ofiles_exist and all(
            [ofile_rb['compo'][var].exists() for var in subgroup_vars]
            )
    
    if not ofiles_exist:
        print("Calculate binned composites...")
        run_get_binned_composites(config, ofile_at, ofile_rb if rainbelt_switch else None)


def run_get_binned_composites(
        config: dict,
        ofile_at: dict,
        ofile_rb: dict | None,
        ) -> None:
    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    ana_times = pcutil.create_analysis_times(config)
    rainbelt_switch = config['composite']['rainbelt_subsampling']['switch']
    subgroup_vars = pcutil.get_subgroup_vars_dict(config)

    if rainbelt_switch:
        rainbelt = pccompo.get_rainbelt(ana_times, config, quantile=0.8)
        rainbelt = rainbelt.compute()

    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    compo_bin_at = {var: [] for var in subgroup_vars.keys()}
    var_bin_at = {var: [] for var in subgroup_vars.keys()}
    N_feats_bin_at = {var: [] for var in subgroup_vars.keys()}
    pvalue_bin_at = {}
    popmeans_at = []
    if rainbelt_switch:
        compo_bin_rb = {var: [] for var in subgroup_vars.keys()}
        var_bin_rb = {var: [] for var in subgroup_vars.keys()}
        N_feats_bin_rb = {var: [] for var in subgroup_vars.keys()}
        pvalue_bin_rb = {}
        popmeans_rb = []

    ipath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ifile = ipath/Path(f"features/{ana_idf}_features_{fdate_str}.nc")
        feats_at = xr.open_dataset(ifile).compute()
        feats_at['asprat_idx'] = calc_asprat_idx(feats_at)
        
        for var, thresholds in subgroup_vars.items():
            compo_bin, var_bin, N_feats_bin = \
                pccompo.get_binned_features(feats_at, thresholds, var)
            compo_bin_at[var].append(compo_bin)
            var_bin_at[var].append(var_bin)
            N_feats_bin_at[var].append(N_feats_bin)

        # read in population means
        ifile = ipath/Path(f"popmeans/{ana_idf}_popmeans_alltrops_{fdate_str}.nc")
        popmeans_at.append(xr.open_dataset(ifile).mean(dim='time').compute())

        if rainbelt_switch:
            feats_rb = pccompo.sample_features_geomask(feats_at, rainbelt)
            for var, thresholds in subgroup_vars.items():
                compo_bin, var_bin, N_feats_bin = \
                    pccompo.get_binned_features(feats_rb, thresholds, var)
                compo_bin_rb[var].append(compo_bin)
                var_bin_rb[var].append(var_bin)
                N_feats_bin_rb[var].append(N_feats_bin)

            # read in population means
            ifile = ipath/Path(
                f"popmeans/{ana_idf}_popmeans_rainbelt_{fdate_str}.nc"
                )
            popmeans_rb.append(
                xr.open_dataset(ifile).mean(dim='time').compute()
                )
                

    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    for var in subgroup_vars:
        compo_bin_at[var], pvalue_bin_at[var] = \
            build_yearly_compo_pvalue_binned(
                compo_bin_at[var], var_bin_at[var], N_feats_bin_at[var], 
                popmeans_at,
                )
        compo_bin_at[var].to_netcdf(str(opath/ofile_at['compo'][var])) #type:ignore
        pvalue_bin_at[var].to_netcdf(str(opath/ofile_at['pvalue'][var])) #type:ignore
            
        if rainbelt_switch:
            compo_bin_rb[var], pvalue_bin_rb[var] = \
            build_yearly_compo_pvalue_binned(
                compo_bin_rb[var], var_bin_rb[var], N_feats_bin_rb[var], 
                popmeans_rb,
                )
            compo_bin_rb[var].to_netcdf(str(opath/ofile_rb['compo'][var])) #type:ignore
            pvalue_bin_rb[var].to_netcdf(str(opath/ofile_rb['pvalue'][var])) #type:ignore


def _reorganize_binned_list(in_list: list[dict]) -> dict:
    out_dict = {key: [d[key] for d in in_list] for key in in_list[0]}
    for key, data in out_dict.items():
        if type(data[0]) == xr.Dataset:
            out_dict[key] = xr.concat(data, dim='month') #type: ignore
        elif type(data[0]) == int:
            out_dict[key] = xr.DataArray(np.array(data), dims=["month"]) #type: ignore
    return out_dict


def build_yearly_compo_pvalue_binned(
        compo: list[dict],
        variance: list[dict],
        N_features: list[dict],
        popmeans: list[xr.Dataset],
        ) -> Tuple[xr.Dataset, xr.Dataset]:
    popmeans_merged = xr.concat(popmeans, dim='month').mean(dim='month')
    compo_merged = _reorganize_binned_list(compo)
    variance_merged = _reorganize_binned_list(variance)
    N_features_merged = _reorganize_binned_list(N_features)

    keys = list(compo_merged.keys())
    sample_compo = compo_merged[keys[0]]
    compovars = [v for v in sample_compo.data_vars if sample_compo[v].ndim == 3]
    
    pvalue = {}
    for key in keys:
        _, pvalue[key] = pcsig.yearly_ttest_from_monthly_data(
            compo_merged[key][compovars], variance_merged[key][compovars],
            N_features_merged[key], popmean=popmeans_merged[compovars],
            )
        compo_merged[key] = compo_merged[key].mean(dim='month')

    compo_merged = xr.concat(
        compo_merged.values(), dim='bin', coords='minimal', compat='override',
        ).assign_coords(bin=keys)
    pvalue = xr.concat(
        pvalue.values(), dim='bin', coords='minimal', compat='override',
        ).assign_coords(bin=keys)

    return compo_merged, pvalue


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
