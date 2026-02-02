from scipy import stats
import numpy as np
import xarray as xr

from typing import Tuple

from pycompo.core.utils import area_weighted_avg

# ------------------------------------------------------------------------------
# Functions for local significance test
# -------------------------------------
def calc_compo_ttest(
        feature_compo_data: xr.Dataset,
        popmean: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
    """Calculate t-test for each grid point in the composite dataset.

    Parameters
    ----------
    feature_compo_data : xr.Dataset
        Composite dataset with dimensions (feature, y, x).
    popmean : float, optional
        Mean value to compare against (default is 0.0).

    Returns
    -------
    tstat : xr.Dataset
        Dataset of t-statistics for each variable.
    pvalue : xr.Dataset
        Dataset of p-values for each variable.
    """
    tstat, pvalue = _calc_compo_ttest(feature_compo_data, popmean)
    tstat = _ttest_dict2xarray(tstat, feature_compo_data)
    pvalue = _ttest_dict2xarray(pvalue, feature_compo_data)
    return tstat, pvalue


def _calc_compo_ttest(
        feature_compo_data: xr.Dataset,
        popmean: xr.Dataset,
    ) -> Tuple[dict, dict]:
    """
    Performs a one-sample t-test for each variable in the input xr.Dataset along 
    the 'feature' dimension.

    Parameters
    ----------
    feature_compo_data : xr.Dataset
        An xr.Dataset where the first dimension must be 'feature'.
        Each variable in the dataset will be tested independently.
    popmean : float, optional
        The population mean to test against. Default is 0.0.

    Returns
    -------
    tstat : dict
        Dictionary mapping variable names to arrays of t-statistics.
    pvalue : dict
        Dictionary mapping variable names to arrays of p-values.

    Raises
    ------
    ValueError
        If the first dimension of `feature_compo_data` is not 'feature'.
    """
    if not list(feature_compo_data.dims.keys())[0] == 'feature':
        raise ValueError(
            "The first dimension of feature_compo_data must be 'feature'."
            )
    ttest_results = {
        var: stats.ttest_1samp(
            feature_compo_data[var], popmean=popmean[var].values, axis=0,
            )
        for var in feature_compo_data.data_vars
        }
    tstat = {var: data.statistic for var, data in ttest_results.items()}
    pvalue = {var: data.pvalue for var, data in ttest_results.items()}

    return tstat, pvalue


def _ttest_dict2xarray(
        data_dict: dict,
        feature_compo_data: xr.Dataset
    ) -> xr.Dataset:
    return xr.Dataset(
        data_vars = {
            var: (('x', 'y'), data) if data.ndim == 2 else 
            (('x', 'y', 'height'), data) for var, data in data_dict.items()
            },
        coords = {
            **{
                'En_rota2_featcen_x': feature_compo_data['En_rota2_featcen_x'],
                'En_rota2_featcen_y': feature_compo_data['En_rota2_featcen_y']
                },
            **(
                {'height': feature_compo_data['height']} if
                any(d.ndim == 3 for d in data_dict.values()) else {}
                )
            }
        )


def get_local_significance(
        pvalue: xr.Dataset,
        alpha: float=0.05
        ) -> xr.Dataset:
    return pvalue < alpha


# ------------------------------------------------------------------------------
# Functions for field significance test
# -------------------------------------
def get_field_significance(
        pvalue: xr.Dataset,
        alpha_FDR: float=0.1
        ) -> xr.Dataset:
    return xr.Dataset(
        data_vars = {
            var: \
                (
                    ('x', 'y'),
                    multiple_hypothesis_test_with_FDR(data.data, alpha_FDR)
                ) if data.ndim == 2 else
                (
                    ('x', 'y', 'height'),
                    multiple_hypothesis_test_with_FDR(data.data, alpha_FDR)
                ) for var, data in pvalue.data_vars.items()
            },
        coords = {
            **{
                'En_rota2_featcen_x': pvalue['En_rota2_featcen_x'],
                'En_rota2_featcen_y': pvalue['En_rota2_featcen_y']
                },
            **(
                {'height': pvalue['height']} if len(pvalue.dims) == 3 else {}
                )
            }
        )

def multiple_hypothesis_test_with_FDR(
        p_field: np.ndarray,
        alpha_FDR: float=0.1,
        apply_heuristics: bool=True,
        logging: bool=False,
        **kwargs
        ) -> np.ndarray:
    """
    Perform a False Discovery Rate (FDR) correction on a 2D field of p-values,
    with additional logic for segment selection and minimum segment length.

    Reference: Wilks (2016) - "The stippling shows statistically significant
    grid points: how research results are routinely overstated and 
    over-interpreted, and what to do about it."

    Parameters
    ----------
    p_field : np.ndarray
        2D array of p-values.
    alpha_FDR : float, optional
        False discovery rate level (default: 0.1).
    gap_threshold : int, optional
        Minimum gap between indices to split segments (default: 10).
    minimum_segment_length : int, optional
        Minimum length for a segment to be considered significant (default: 10).

    Returns
    -------
    sigmask : np.ndarray
        Boolean array of the same shape as p_field, True for statistically
        significant grid points, False otherwise.
    """

    # Step 1: Apply Benjamini-Hochberg procedure
    p_vector_ascend, p_vector_select, selmask = \
        _benjamini_hochberg_procedure(p_field, alpha_FDR)
    
    if p_vector_select.size == 0:
        if logging: print("None of the p-values are smaller than pFDR.")
        return np.zeros_like(p_field, dtype=bool)

    # Step 2: Apply heuristics
    if apply_heuristics:
        p_vector_select = fdr_heuristics(
            p_vector_ascend, p_vector_select, selmask, logging, **kwargs
            )
        if p_vector_select.size == 0:
            if logging: print("None of the p-values are smaller than pFDR.")
            return np.zeros_like(p_field, dtype=bool)

    p_FDR = np.max(p_vector_select)
    sigmask = p_field <= p_FDR
    return sigmask


def _benjamini_hochberg_procedure(
        p_field: np.ndarray,
        alpha_FDR: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Benjamini-Hochberg procedure to control the FDR."""
    N_points = p_field.size
    p_vector = p_field.flatten()
    p_vector_ascend = np.sort(p_vector)
    idx = np.arange(1, N_points + 1)

    selmask = p_vector_ascend - alpha_FDR * idx / N_points <= 1e-5
    p_vector_select = p_vector_ascend[selmask]
    
    return p_vector_ascend, p_vector_select, selmask


def fdr_heuristics(
        p_vector_ascend: np.ndarray,
        p_vector_select: np.ndarray,
        selmask: np.ndarray,
        logging: bool,
        gap_threshold: int=10,
        minimum_segment_length: int=10
        ) -> np.ndarray:
    # Step 1: identify large consecutive segments
    idx_select = np.where(selmask)[0]
    edge_idxs = np.where(np.diff(idx_select) >= gap_threshold)[0]

    if edge_idxs.size > 0:
        dim1Dist = np.concatenate(
            ([edge_idxs[0]], np.diff(edge_idxs),
            [len(idx_select) - edge_idxs[-1]])
            )
    else:
        dim1Dist = np.array([len(idx_select)])

    # Step 2: split into segments
    segments = np.split(p_vector_select, np.cumsum(dim1Dist[:-1]))
    N_segments = len(segments)
    if logging: print(f"nseg = {N_segments}")

    if N_segments > 1:
        if logging: print("special case:")
        seg_lengths = np.array([len(s) for s in segments])
        maxlen = np.max(seg_lengths)
        maxid = np.argmax(seg_lengths)
        p_segment_select = segments[maxid]

        if logging: print(f"Segment length = {maxlen}")

        if maxlen > minimum_segment_length:
            # find number of insignificant p-values before this segment
            pid = np.argmin(np.abs(p_segment_select[0] - p_vector_ascend))
            insig_datalen = pid
            if insig_datalen > round(0.1 * maxlen):
                if logging: print(
                    f"Segment rejected due to insig_datalen = {insig_datalen}"
                    )
                p_segment_select = np.array([])
        else:
            p_segment_select = np.array([])

    else:
        p_segment_select = p_vector_select
        if len(p_segment_select) < minimum_segment_length:
            p_segment_select = np.array([])

    return p_segment_select


# ------------------------------------------------------------------------------
# Manual significance test
# ------------------------
def yearly_ttest_from_monthly_data(
        mon_mean: xr.Dataset,
        mon_var: xr.Dataset,
        n_per_mon: xr.DataArray,
        popmean: float=0.0,
        ):
    year_mean, year_var = _monthly2yearly_stats(mon_mean, mon_var, n_per_mon)
    t_stat, p_value = _manual_t_test(
        year_mean, year_var, n_per_mon.sum(dim='month'), popmean
        )
    return t_stat, p_value


def _monthly2yearly_stats(
        mon_mean: xr.Dataset,
        mon_var: xr.Dataset,
        n_per_mon: xr.DataArray,
        ) -> Tuple[xr.Dataset, xr.Dataset]:
    year_mean = (n_per_mon * mon_mean).sum(dim='month') / \
        n_per_mon.sum(dim='month')
    year_var = (
        ((n_per_mon - 1) * mon_var) + (n_per_mon * (mon_mean - year_mean)**2)
        ).sum(dim='month') / (n_per_mon.sum(dim='month') - 1)
    
    return year_mean, year_var
    

def _manual_t_test(
        mean: np.ndarray,
        variance: np.ndarray,
        n_sample: int,
        popmean: float=0.0
        ) -> Tuple[np.ndarray, np.ndarray]:
    std_error = np.sqrt(variance / n_sample)
    t_stat = (mean - popmean) / std_error
    p_value = {
        var: stats.t.sf(np.abs(t_stat[var].values), df=n_sample-1) * 2
        for var in t_stat.data_vars
        }
    p_value = xr.Dataset(
        data_vars = {
            var: (('x', 'y'), data) if data.ndim == 2 else 
            (('x', 'y', 'height'), data) for var, data in p_value.items()
            },
        coords = {
            **{
                'En_rota2_featcen_x': t_stat['En_rota2_featcen_x'],
                'En_rota2_featcen_y': t_stat['En_rota2_featcen_y']
                },
            **(
                {'height': t_stat['height']} if
                any(d.ndim == 3 for d in p_value.values()) else {}
                )
            }
        )
    return t_stat, p_value


# ------------------------------------------------------------------------------
# Functions to calculate population mean
# --------------------------------------
def calc_popmeans(
        dset: xr.Dataset,
        feature_var: str,
        ) -> xr.Dataset:
    popmeans = area_weighted_avg(dset.drop('cell_area'), dset['cell_area'])
    popmeans[f'downwind_{feature_var}_ano_grad'] = \
        popmeans[f'd{feature_var}_ano_dx'] * 0. 
    popmeans[f'crosswind_{feature_var}_ano_grad'] = \
        popmeans[f'd{feature_var}_ano_dy'] * 0.
    return popmeans