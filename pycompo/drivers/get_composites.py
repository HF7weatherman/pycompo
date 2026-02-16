import numpy as np
import sys
import warnings
import xarray as xr
from pathlib import Path
from typing import Tuple

import pycompo.core.utils as pcutil
import pycompo.core.sigtest as pcsig
from pycompo.core.composite import get_rainbelt, sample_features_geomask

warnings.filterwarnings(action='ignore')


# ------------------------------------------------------------------------------
def main():
    config = pcutil.read_yaml_config(sys.argv[1])
    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    opath = Path(f"{config['data']['outpath']}/{ana_idf}/")
    opath.mkdir(parents=True, exist_ok=True)

    ofiles = {
        "alltrops_pvalue": opath/Path(f"{ana_idf}_pvalue_alltrops.nc"),
        "alltrops_compo": opath/Path(f"{ana_idf}_composite_alltrops.nc"),
    }
    ofiles_exist = ofiles['alltrops_pvalue'].exists()
    ofiles_exist = ofiles['alltrops_compo'].exists()
    if config['composite']['rainbelt_subsampling']['switch']:
        ofiles['rainbelt_pvalue'] = opath/Path(f"{ana_idf}_pvalue_rainbelt.nc")
        ofiles['rainbelt_compo'] = opath/Path(f"{ana_idf}_composite_rainbelt.nc")
        ofiles_exist = ofiles_exist and ofiles['rainbelt_pvalue'].exists()
        ofiles_exist = ofiles_exist and ofiles['rainbelt_compo'].exists()

    if not ofiles_exist:
        print("Building the composites...")
        run_get_composites(config, ofiles)


def build_yearly_compo_pvalue(
        compo: list[xr.Dataset],
        popmeans: list[xr.Dataset],
        variance: list[xr.Dataset],
        N_features: list[np.float64],
        ) -> Tuple[xr.Dataset, xr.Dataset]:
    popmeans_merged = xr.concat(popmeans, dim='month').mean(dim='month')
    compo_merged = xr.concat(compo, dim='month')
    variance_merged = xr.concat(variance, dim='month')
    N_features_merged = xr.DataArray(np.array(N_features), dims=["month"])

    compovars = [v for v in compo_merged.data_vars if compo_merged[v].ndim == 3]
    
    _, pvalue = pcsig.yearly_ttest_from_monthly_data(
        compo_merged[compovars], variance_merged[compovars],
        N_features_merged, popmean=popmeans_merged[compovars],
        )
    compo_merged = compo_merged.mean(dim='month')
    return compo_merged, pvalue
    

def run_get_composites(config: dict, ofiles: dict) -> None:
    ana_times = pcutil.create_analysis_times(config)
    ana_idf = f"{config['exp']}_{config['pycompo_name']}"
    rainbelt_switch = config['composite']['rainbelt_subsampling']['switch']
    ipath_feats = Path(f"{config['data']['outpath']}/{ana_idf}/features/")
    ipath_popms = Path(f"{config['data']['outpath']}/{ana_idf}/popmeans/")
    at_compo, at_var, at_N_feats, at_popmeans = [], [], [], []

    if rainbelt_switch:
        rb_compo, rb_var, rb_N_feats, rb_popmeans = [], [], [], []
        rainbelt = get_rainbelt(ana_times, config, quantile=0.8).compute()

    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    for start_time, end_time in zip(ana_times, ana_times[1:]):
        fdate_str = pcutil.create_ftime_str(start_time, end_time)
        ifile = Path(f"{ana_idf}_features_{fdate_str}.nc")
        feats_at = xr.open_dataset(ipath_feats/ifile).compute()

        at_compo.append(feats_at.drop_vars("time").mean(dim='feature'))
        at_var.append(feats_at.drop_vars("time").var(dim='feature', ddof=1))
        at_N_feats.append(feats_at.drop_vars("time").sizes['feature'])

        # read in population means
        ifile = Path(f"{ana_idf}_popmeans_alltrops_{fdate_str}.nc")
        at_popmeans.append(
            xr.open_dataset(ipath_popms/ifile).mean(dim='time').compute()
            )

        if rainbelt_switch:
            feats_rb = sample_features_geomask(feats_at, rainbelt)
            rb_compo.append(feats_rb.drop_vars("time").mean(dim='feature'))
            rb_var.append(feats_rb.drop_vars("time").var(dim='feature', ddof=1))
            rb_N_feats.append(feats_rb.drop_vars("time").sizes['feature'])

            ifile = Path(f"{ana_idf}_popmeans_rainbelt_{fdate_str}.nc")
            rb_popmeans.append(
                xr.open_dataset(ipath_popms/ifile).mean(dim='time').compute()
                )
    
    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    at_compo, at_pvalue = build_yearly_compo_pvalue(
        at_compo, at_popmeans, at_var, at_N_feats,
        )
    at_pvalue.to_netcdf(str(ofiles['alltrops_pvalue']))
    at_compo.to_netcdf(str(ofiles['alltrops_compo']))

    if rainbelt_switch:
        rb_compo, rb_pvalue = build_yearly_compo_pvalue(
            rb_compo, rb_popmeans, rb_var, rb_N_feats,
            )
        rb_pvalue.to_netcdf(str(ofiles['rainbelt_pvalue']))
        rb_compo.to_netcdf(str(ofiles['rainbelt_compo']))


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
