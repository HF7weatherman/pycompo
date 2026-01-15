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

    # set and create outpath if necesary
    analysis_idf = f"{config['exp']}_{config['pycompo_name']}"
    outpath = Path(f"{config['data']['outpath']}/{analysis_idf}/")
    outpath.mkdir(parents=True, exist_ok=True)

    # set outfiles and check whether they already exist
    outfiles = {
        "alltrops_pvalue": outpath/Path(f"{analysis_idf}_pvalue_alltrops.nc"),
        "alltrops_compo": outpath/Path(f"{analysis_idf}_composite_alltrops.nc"),
    }
    outfiles_exist = outfiles['alltrops_pvalue'].exists()
    outfiles_exist = outfiles['alltrops_compo'].exists()
    if config['composite']['rainbelt_subsampling']['switch']:
        outfiles['rainbelt_pvalue'] = \
            outpath/Path(f"{analysis_idf}_pvalue_rainbelt.nc")
        outfiles['rainbelt_compo'] = \
            outpath/Path(f"{analysis_idf}_composite_rainbelt.nc")
        outfiles_exist = outfiles_exist and outfiles['rainbelt_pvalue'].exists()
        outfiles_exist = outfiles_exist and outfiles['rainbelt_compo'].exists()

    if not outfiles_exist:
        print("Combining feature properties from all time steps ...")
        run_get_composites(config, outfiles)


def run_get_composites(
        config: dict,
        outfiles: dict,
        ) -> None:
    start_time = config['data']['analysis_time'][0]
    end_time = config['data']['analysis_time'][1]
    analysis_times = [
        np.datetime64(t) for t in date_range(
            np.datetime64(start_time), np.datetime64(end_time), freq='MS',
            )
        ]
    analysis_idf = f"{config['exp']}_{config['pycompo_name']}"

    # build rainbelt if necessary
    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt = pccompo.get_rainbelt(analysis_times, config, quantile=0.8)
        rainbelt = rainbelt.compute()


    # --------------------------------------------------------------------------
    # create feature composite per time step
    # --------------------------------------
    inpath_feats = Path(f"{config['data']['outpath']}/{analysis_idf}/features/")
    alltrops_compo = []
    alltrops_var = []
    alltrops_N_features = []

    inpath_popms = Path(f"{config['data']['outpath']}/{analysis_idf}/popmeans/")
    alltrops_popmeans = []

    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt_compo = []
        rainbelt_var = []
        rainbelt_N_features = []
        rainbelt_popmeans = []

    for i in range (len(analysis_times)-1):
        file_timestr = \
            f"{pcutils.np_datetime2file_datestr(analysis_times[i])}-" + \
            f"{pcutils.np_datetime2file_datestr(analysis_times[i+1])}"
        
        # read in fetaure data
        infile = inpath_feats/Path(f"{analysis_idf}_features_{file_timestr}.nc")
        features_alltrops = xr.open_dataset(infile).compute()
        compo_vars = [
            v for v in features_alltrops.data_vars
            if features_alltrops[v].ndim == 3
            ]
    
        alltrops_compo.append(features_alltrops.mean(dim='feature'))
        alltrops_var.append(features_alltrops.var(dim='feature', ddof=1))
        alltrops_N_features.append(features_alltrops.sizes['feature'])

        # read in population means
        infile = inpath_popms/Path(
            f"{analysis_idf}_popmeans_alltrops_{file_timestr}.nc"
            )
        alltrops_popmeans.append(
            xr.open_dataset(infile).mean(dim='time').compute()
            )

        # Precipitation-based geographic subsampling
        if config['composite']['rainbelt_subsampling']['switch']:
            features_rainbelt = pccompo.sample_features_geomask(
                features_alltrops, rainbelt,
                )
            rainbelt_compo.append(features_rainbelt.mean(dim='feature'))
            rainbelt_var.append(features_rainbelt.var(dim='feature', ddof=1))
            rainbelt_N_features.append(features_rainbelt.sizes['feature'])

            # read in population means
            infile = inpath_popms/Path(
                f"{analysis_idf}_popmeans_rainbelt_{file_timestr}.nc"
                )
            rainbelt_popmeans.append(
                xr.open_dataset(infile).mean(dim='time').compute()
                )
    
    
    # --------------------------------------------------------------------------
    # merge to a full feature composite
    # ---------------------------------
    alltrops_popmeans = \
        xr.concat(alltrops_popmeans, dim='month').mean(dim='month')

    alltrops_compo = xr.concat(alltrops_compo, dim='month')
    alltrops_var = xr.concat(alltrops_var, dim='month')
    alltrops_N_features = xr.DataArray(
        np.array(alltrops_N_features), dims=["month"],
        )
    _, alltrops_pvalue = pcsig.yearly_ttest_from_monthly_data(
        alltrops_compo[compo_vars], alltrops_var[compo_vars],
        alltrops_N_features, popmean=alltrops_popmeans[compo_vars],
        )
    alltrops_pvalue.to_netcdf(str(outfiles['alltrops_pvalue']))
    alltrops_compo = alltrops_compo.mean(dim='month')
    alltrops_compo.to_netcdf(str(outfiles['alltrops_compo']))

    if config['composite']['rainbelt_subsampling']['switch']:
        rainbelt_popmeans = \
            xr.concat(rainbelt_popmeans, dim='month').mean(dim='month')
        rainbelt_compo = xr.concat(rainbelt_compo, dim='month')
        rainbelt_var = xr.concat(rainbelt_var, dim='month')
        rainbelt_N_features = xr.DataArray(
            np.array(rainbelt_N_features), dims=["month"],
            )
        _, rainbelt_pvalue = pcsig.yearly_ttest_from_monthly_data(
            rainbelt_compo[compo_vars], rainbelt_var[compo_vars], 
            rainbelt_N_features, popmean=rainbelt_popmeans[compo_vars],
            )
        rainbelt_pvalue.to_netcdf(str(outfiles['rainbelt_pvalue']))
        rainbelt_compo = rainbelt_compo.mean(dim='month')
        rainbelt_compo.to_netcdf(str(outfiles['rainbelt_compo']))


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()