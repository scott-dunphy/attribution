import pandas as pd
import numpy as np

def _compute_weights(segment_df, total_denom):
    """Compute segment weights from Denom / total_denom."""
    if total_denom == 0:
        return pd.Series(0.0, index=segment_df.index)
    return segment_df['Denom'] / total_denom


def _get_query_level_for_dimension(dimension):
    """Map dimension name to Query_Level filter."""
    return {
        'PropertyType': 'By_PropertyType',
        'CBSAName': 'By_CBSA',
        'PropertyType_CBSAName': 'By_PropertyType_CBSA',
        'HeldSold': 'By_HeldSold',
    }[dimension]


def _get_merge_cols(dimension):
    """Return the column(s) to merge on for a given dimension."""
    if dimension == 'PropertyType':
        return ['PropertyType']
    elif dimension == 'CBSAName':
        return ['CBSAName']
    elif dimension == 'HeldSold':
        return ['HeldSold']
    else:
        return ['PropertyType', 'CBSAName']


def brinson_fachler_single_period(portfolio_df, benchmark_df, year, quarter, dimension):
    """
    Compute Brinson-Hood-Beebower (MSCI) attribution for a single period and dimension.

    Returns DataFrame with columns:
        merge_cols + [w_p, w_b, r_p, r_b, R_b, Allocation, Selection, Total_Active]
    """
    query_level = _get_query_level_for_dimension(dimension)
    merge_cols = _get_merge_cols(dimension)

    # Get total rows
    p_total = portfolio_df[
        (portfolio_df['Query_Level'] == 'Total') &
        (portfolio_df['Year'] == year) &
        (portfolio_df['Quarter'] == quarter)
    ]
    b_total = benchmark_df[
        (benchmark_df['Query_Level'] == 'Total') &
        (benchmark_df['Year'] == year) &
        (benchmark_df['Quarter'] == quarter)
    ]

    if p_total.empty or b_total.empty:
        return pd.DataFrame()

    p_total_row = p_total.iloc[0]
    b_total_row = b_total.iloc[0]

    R_p = p_total_row['Total_Return']
    R_b = b_total_row['Total_Return']
    p_total_denom = p_total_row['Denom']
    b_total_denom = b_total_row['Denom']

    # Get segment rows
    p_segs = portfolio_df[
        (portfolio_df['Query_Level'] == query_level) &
        (portfolio_df['Year'] == year) &
        (portfolio_df['Quarter'] == quarter)
    ].copy()
    b_segs = benchmark_df[
        (benchmark_df['Query_Level'] == query_level) &
        (benchmark_df['Year'] == year) &
        (benchmark_df['Quarter'] == quarter)
    ].copy()

    # Filter out 'All' values from the merge dimension(s)
    for col in merge_cols:
        if col in p_segs.columns:
            p_segs = p_segs[p_segs[col] != 'All']
        if col in b_segs.columns:
            b_segs = b_segs[b_segs[col] != 'All']
        else:
            # Benchmark doesn't have this dimension — ensure column exists for merge
            b_segs[col] = pd.Series(dtype='object')

    # Compute weights
    p_segs['w_p'] = _compute_weights(p_segs, p_total_denom)
    p_segs['r_p'] = p_segs['Total_Return']
    p_segs['r_p_income'] = p_segs['Income_Return']
    p_segs['r_p_capital'] = p_segs['Capital_Return']

    b_segs['w_b'] = _compute_weights(b_segs, b_total_denom)
    b_segs['r_b'] = b_segs['Total_Return']
    b_segs['r_b_income'] = b_segs['Income_Return']
    b_segs['r_b_capital'] = b_segs['Capital_Return']

    # Merge
    p_cols = merge_cols + ['w_p', 'r_p', 'r_p_income', 'r_p_capital', 'Prop_Count']
    b_cols = merge_cols + ['w_b', 'r_b', 'r_b_income', 'r_b_capital']

    merged = pd.merge(
        p_segs[p_cols].rename(columns={'Prop_Count': 'Prop_Count_P'}),
        b_segs[b_cols],
        on=merge_cols,
        how='outer'
    )

    # Fill missing -- benchmark first so portfolio fallbacks can reference it
    merged['w_p'] = merged['w_p'].fillna(0.0)
    merged['w_b'] = merged['w_b'].fillna(0.0)
    merged['r_b'] = merged['r_b'].fillna(R_b)
    merged['r_b_income'] = merged['r_b_income'].fillna(0.0)
    merged['r_b_capital'] = merged['r_b_capital'].fillna(0.0)
    merged['r_p'] = merged['r_p'].fillna(merged['r_b'])
    merged['r_p_income'] = merged['r_p_income'].fillna(merged['r_b_income'])
    merged['r_p_capital'] = merged['r_p_capital'].fillna(merged['r_b_capital'])
    merged['Prop_Count_P'] = merged['Prop_Count_P'].fillna(0).astype(int)

    # Brinson-Hood-Beebower (MSCI) formulas
    merged['Allocation'] = (merged['w_p'] - merged['w_b']) * (merged['r_b'] - R_b)
    merged['Selection'] = merged['w_p'] * (merged['r_p'] - merged['r_b'])
    merged['Total_Active'] = merged['Allocation'] + merged['Selection']

    # Absolute contribution
    merged['Contribution_P'] = merged['w_p'] * merged['r_p']
    merged['Contribution_B'] = merged['w_b'] * merged['r_b']

    merged['R_b'] = R_b
    merged['R_p'] = R_p
    merged['Year'] = year
    merged['Quarter'] = quarter

    return merged


# ---------------------------------------------------------------------------
# Multi-period linking
# ---------------------------------------------------------------------------

def _compound_segment_returns(detail_df, merge_cols):
    """Compute cumulative compounded segment returns per segment."""
    records = []
    for keys, group in detail_df.groupby(merge_cols):
        g = group.sort_values(['Year', 'Quarter'])
        cum_rp = 1.0
        cum_rb = 1.0
        for _, row in g.iterrows():
            cum_rp *= (1 + row['r_p'])
            cum_rb *= (1 + row['r_b'])

        if isinstance(keys, tuple):
            rec = dict(zip(merge_cols, keys))
        else:
            rec = {merge_cols[0]: keys}
        rec['r_p_cum'] = cum_rp - 1
        rec['r_b_cum'] = cum_rb - 1
        records.append(rec)

    return pd.DataFrame(records)


def _link_effects(detail_df, merge_cols, cum_rp, cum_rb, method='scale'):
    """
    Link single-period BHB (MSCI) effects to multi-period cumulative
    attribution.

    method='scale': Proportional scaling — scale all segment effects so they
        sum exactly to the cumulative geometric active return.
    method='residual': No scaling — show raw effects and add a Residual/Other
        row for the difference between raw sum and cumulative active return.

    1. Sum each segment's raw effects across all periods.
    2. Reconcile via chosen method.
    3. Compute cumulative compounded segment returns and contributions.
    """
    cum_active = cum_rp - cum_rb
    effect_cols = ['Allocation', 'Selection', 'Total_Active']

    # Sum effects across periods per segment
    agg_dict = {col: 'sum' for col in effect_cols}
    agg_dict['w_p'] = 'mean'
    agg_dict['w_b'] = 'mean'
    summary = detail_df.groupby(merge_cols, as_index=False).agg(agg_dict)

    if method == 'scale':
        # Proportional scaling: scale effects so they sum to cum_active
        raw_total = summary['Total_Active'].sum()
        if abs(raw_total) > 1e-12:
            scale = cum_active / raw_total
            for col in effect_cols:
                summary[col] = summary[col] * scale
        elif abs(cum_active) > 1e-12:
            n_segs = len(summary)
            if n_segs > 0:
                per_seg = cum_active / n_segs
                summary['Total_Active'] = per_seg
                summary['Allocation'] = per_seg / 2
                summary['Selection'] = per_seg / 2
    else:
        # Residual method: leave raw effects, add residual row
        raw_total = summary['Total_Active'].sum()
        residual = cum_active - raw_total
        if abs(residual) > 1e-12:
            residual_row = {col: 0.0 for col in summary.columns if col not in merge_cols}
            for col in merge_cols:
                residual_row[col] = 'Residual/Other'
            residual_row['Total_Active'] = residual
            residual_row['Allocation'] = 0.0
            residual_row['Selection'] = 0.0
            residual_row['w_p'] = 0.0
            residual_row['w_b'] = 0.0
            summary = pd.concat([summary, pd.DataFrame([residual_row])], ignore_index=True)

    # Compute cumulative compounded segment returns
    seg_returns = _compound_segment_returns(detail_df, merge_cols)
    summary = summary.merge(seg_returns, on=merge_cols, how='left')

    # Fill NaN for residual row's returns
    summary['r_p_cum'] = summary['r_p_cum'].fillna(0.0)
    summary['r_b_cum'] = summary['r_b_cum'].fillna(0.0)

    # Segment contributions: average weight * cumulative compounded segment return.
    summary['Contribution_P'] = summary['w_p'] * summary['r_p_cum']
    summary['Contribution_B'] = summary['w_b'] * summary['r_b_cum']

    return summary


def _cumulative_return(detail_df, return_col):
    """Compute cumulative compounded return from period detail."""
    period_returns = detail_df.groupby(['Year', 'Quarter'])[return_col].first().sort_index()
    cum = 1.0
    for r in period_returns:
        cum *= (1 + r)
    return cum - 1


def _annualize(cum_return, n_quarters):
    """Annualize a cumulative return over n_quarters."""
    if n_quarters <= 0:
        return 0.0
    n_years = n_quarters / 4.0
    if n_years <= 1.0:
        return cum_return  # Don't annualize sub-annual periods
    if cum_return <= -1.0:
        return -1.0  # Cap at total loss; values below -1.0 indicate data error
    return (1 + cum_return) ** (1 / n_years) - 1


# ---------------------------------------------------------------------------
# Trailing period helpers
# ---------------------------------------------------------------------------

TRAILING_PERIOD_DEFS = {
    '1Q': 1,
    '1Y': 4,
    '3Y': 12,
    '5Y': 20,
    '7Y': 28,
    '10Y': 40,
    'SI': None,  # Since inception = all available
}

TRAILING_PERIOD_ORDER = ['1Q', '1Y', '3Y', '5Y', '7Y', '10Y', 'SI']


def get_trailing_periods(all_periods, as_of, trailing_key):
    """
    Given sorted list of (year, quarter) tuples, an as_of (year, quarter),
    and a trailing period key, return the list of periods in that window.
    """
    # Filter to periods <= as_of
    eligible = [p for p in all_periods if p <= as_of]
    if not eligible:
        return []

    n_quarters = TRAILING_PERIOD_DEFS.get(trailing_key)
    if n_quarters is None:
        return eligible  # Since inception
    return eligible[-n_quarters:]


def get_available_trailing_periods(all_periods, as_of):
    """Return dict of trailing_key -> bool indicating which periods have enough data."""
    eligible = [p for p in all_periods if p <= as_of]
    n = len(eligible)
    available = {}
    for key in TRAILING_PERIOD_ORDER:
        req = TRAILING_PERIOD_DEFS[key]
        if req is None:
            available[key] = n >= 1
        else:
            available[key] = n >= req
    return available


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_attribution(portfolio_df, benchmark_df, periods, dimension, method='scale'):
    """
    Run Brinson-Fachler attribution across multiple periods for a given dimension.

    method: 'scale' for proportional scaling, 'residual' for raw effects + residual row.
    """
    results = []
    for year, quarter in periods:
        df = brinson_fachler_single_period(portfolio_df, benchmark_df, year, quarter, dimension)
        if not df.empty:
            results.append(df)

    if not results:
        return {'detail': pd.DataFrame(), 'summary': pd.DataFrame(),
                'cum_rp': 0.0, 'cum_rb': 0.0, 'ann_rp': 0.0, 'ann_rb': 0.0,
                'n_quarters': 0}

    detail = pd.concat(results, ignore_index=True)
    merge_cols = _get_merge_cols(dimension)
    n_quarters = len(results)  # Count periods that actually produced data

    # Cumulative compounded returns
    cum_rp = _cumulative_return(detail, 'R_p')
    cum_rb = _cumulative_return(detail, 'R_b')

    # Annualized returns
    ann_rp = _annualize(cum_rp, n_quarters)
    ann_rb = _annualize(cum_rb, n_quarters)

    # Linked multi-period summary
    summary = _link_effects(detail, merge_cols, cum_rp, cum_rb, method=method)
    summary = summary.sort_values('Total_Active', ascending=False)

    return {
        'detail': detail,
        'summary': summary,
        'cum_rp': cum_rp,
        'cum_rb': cum_rb,
        'ann_rp': ann_rp,
        'ann_rb': ann_rb,
        'n_quarters': n_quarters,
    }


def has_held_sold(portfolio_df):
    """Check if portfolio has By_HeldSold data."""
    return 'By_HeldSold' in portfolio_df['Query_Level'].values


def run_full_attribution(portfolio_df, benchmark_df, periods, method='scale'):
    """Run attribution for all dimensions. Returns dict of results."""
    results = {
        'by_property_type': run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType', method=method),
        'by_cbsa': run_attribution(portfolio_df, benchmark_df, periods, 'CBSAName', method=method),
        'by_property_type_cbsa': run_attribution(portfolio_df, benchmark_df, periods, 'PropertyType_CBSAName', method=method),
    }
    if has_held_sold(portfolio_df):
        results['by_held_sold'] = run_attribution(portfolio_df, benchmark_df, periods, 'HeldSold', method=method)
    return results


def run_all_trailing_periods(portfolio_df, benchmark_df, all_periods, as_of):
    """
    Run attribution for every trailing period window, for all dimensions.

    Returns dict:
        { trailing_key: { 'by_property_type': ..., 'by_cbsa': ..., ... } }
    """
    available = get_available_trailing_periods(all_periods, as_of)
    results = {}
    for key in TRAILING_PERIOD_ORDER:
        if available[key]:
            periods = get_trailing_periods(all_periods, as_of, key)
            results[key] = run_full_attribution(portfolio_df, benchmark_df, periods)
        else:
            results[key] = None
    return results, available
