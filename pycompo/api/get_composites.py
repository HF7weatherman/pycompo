from pathlib import Path
import xarray as xr
import warnings

from pyorg.core.geometry import get_cells_area

import pycompo.core.coord as pccoord
import pycompo.core.ellipse as pcellipse
import pycompo.core.feature_cutout as pcfeatcut
import pycompo.core.filter as pcfilter
import pycompo.core.sst_features as pcsst
import pycompo.core.utils as pcutils
import pycompo.core.wind as pcwind

from pycompo.core.composite import interpolate2compo_coords
from pycompo.core.utils import read_yaml_config

warnings.filterwarnings(action='ignore')


# ------------------------------------------------------------------------------
def main():
    # read in settings
    config_file = "/home/m/m300738/libs/pycompo/config/settings_template.yaml"
    config = read_yaml_config(config_file)

    start_time = config['data']['analysis_time'][0]
    end_time = config['data']['analysis_time'][1]

    # read in data
    infiles = []
    for var in [config['data']['feature_var']] + config['data']['wind_vars']:
        inpath = Path(config['data']['inpaths'][var])
        in_pattern = \
            f"{config['exp']}_tropical_{var}_20200801T000000Z-20200901T000000Z.nc"
        infiles.extend(sorted([str(f) for f in inpath.rglob(in_pattern)]))
    dset = xr.open_mfdataset(infiles, parallel=True).squeeze()
    for var in config['data']['wind_vars']: dset[var] = dset[var].compute()
    dset['cell_area'] = get_cells_area(dset)

    # 2) BG trend removal and filtering
    # a) Substract climatology
    if config['detrend']['switch']:
        feature_var = config['data']['feature_var']

        # Read in data for building the climatology
        inpath = Path(config['data']['inpaths'][feature_var])
        in_pattern = f"{config['exp']}_tropical_{feature_var}_*.nc"
        infiles = sorted([str(f) for f in inpath.rglob(in_pattern)])
        dset_clim = xr.open_mfdataset(infiles, parallel=True).squeeze()
        
        # Detrend dataset with multiyear monthly climatology
        climatology = pcfilter.build_hourly_climatology(
            dset_clim, clim_baseyear=str(config['detrend']['clim_baseyear'])
            )
        rolling_climatology = pcutils.circ_roll_avg(
            climatology, config['detrend']['clim_avg_days'], config['data']['spd'],
            )
        dset[f'{feature_var}_detrend'] = \
            dset[feature_var] - rolling_climatology[feature_var]
        dset[f'{feature_var}_detrend'] = dset[f'{feature_var}_detrend'].compute()
        config['data']['feature_var'] = f'{feature_var}_detrend'

    # b) Scale separation using a Gaussian filter
    dset = xr.merge([
        dset,
        pcfilter.get_gaussian_filter_bg_ano(
            dset[config['data']['feature_var']], **config['filter'],
            )
        ])
    dset = dset.sel(lat=slice(*config['lat_range']), drop=True)
    orig_coords = pccoord.get_coords_orig(dset)

    # 4) Detect SST anomaly features
    dset[f"{config['data']['feature_var']}_feature"], feature_props = \
        pcsst.extract_sst_features(
            dset[f"{config['data']['feature_var']}_ano"], **config['feature']
        )

    # 5) Create feature-centric Cartesian coordinate
        # a) Coordinate transformation to Cartesian coordinate
        # b) Rotation to align major axis with abcissa (in geophys. coord. space)
        # c) Normalize with major axis length (in geophys. coord. space)

    # 6) Create wind-aligned normalized coordinate system
    dset, feature_props, feature_data = pcfeatcut.get_featcen_data_cutouts(
        dset, feature_props, config['cutout']['search_RadRatio'],
        )

    feature_ellipse = pcellipse.get_ellipse_params(feature_props, orig_coords)

    feature_props = pcwind.calc_feature_bg_wind(
        feature_props, feature_data, config['data']['wind_vars'],
        )
    feature_data = pccoord.add_featcen_coords(
        orig_coords, feature_data, feature_props, feature_ellipse,
        )

    # 8) Write the output
    analysis_identifier = f"{config['exp']}_{config['pycompo_name']}"

    # save feature props
    outpath = Path(f"{config['data']['outpath']}/{analysis_identifier}/")
    outpath.mkdir(parents=True, exist_ok=True)
    outfile = Path(
        f"{analysis_identifier}_feature_props_{start_time}-{end_time}.nc"
        )
    feature_props.attrs["identifier"] = analysis_identifier
    feature_props.to_netcdf(str(outpath/outfile))

    # save feature data
    outpath = Path(
        f"{config['data']['outpath']}/{analysis_identifier}/" + \
        f"{analysis_identifier}_feature_data_{start_time}-{end_time}/"
        )
    outpath.mkdir(parents=True, exist_ok=True)

    for data in feature_data[:3]:
        feature_id = data['feature_id'].values
        outfile = Path(
            f"{analysis_identifier}_feature_data_{start_time}-{end_time}_" + \
            f"feature{feature_id}.nc"
            )
        data.attrs["identifier"] = analysis_identifier
        data.drop(['height_2', 'uas', 'vas']).to_netcdf(str(outpath/outfile))

#### BUILD THE COMPOSITES
    # 1) Read in data
        # a) SST cutouts data
        # b) Data to build composites from
        # 7) Remap to the same Cartesian coordinate system to be able to align data and create composites
compo_varlst = [
    f"{config['data']['feature_var']}_ano", "pr_ano", "prw_ano", "hfls_ano",
    "hfss_ano", "sfcwind_ano",
    ]
feature_compo_data = {
    var: interpolate2compo_coords(
        feature_data,
        (config['composite']['compo_x'], config['composite']['compo_y']),
        var
        ) for var in compo_varlst
    }
feature_compo_data = xr.merge([feature_compo_data[var] for var in compo_varlst])

# 8) Filter features that should be used for composites:
    # a) Size-based sampling
    # b) Geographic sampling (only within a monthly moving window of the 48mm-contour of PRW)
    # c) Analysis-time sampling
    # d) BG-wind-based sampling

# 9) Construction of feature-based composites