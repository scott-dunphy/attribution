"""
Aggregate property-level portfolio data into the Query_Level format
required by the Brinson-Fachler attribution engine.

Input: one row per property per quarter
Output: DataFrame matching the benchmark format with Query_Level rows for
        Total, By_PropertyType, By_CBSA, By_PropertyType_CBSA
"""
import pandas as pd
import numpy as np


PORTFOLIO_REQUIRED_COLUMNS = [
    'Year', 'Quarter', 'PropertyType', 'CBSAName',
    'NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
    'Income_Return', 'Capital_Return', 'Total_Return',
]

DOLLAR_COLS = ['NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom']
RETURN_COLS = ['Income_Return', 'Capital_Return', 'Total_Return']


def validate_property_file(df):
    """Validate a property-level portfolio file. Returns list of errors."""
    errors = []
    missing = set(PORTFOLIO_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")
    return errors


def is_property_level(df):
    """Detect whether a DataFrame is property-level (no Query_Level column)
    or already in the aggregated benchmark format."""
    if 'Query_Level' not in df.columns:
        return True
    # If Query_Level exists but all values are NaN or empty, treat as property-level
    levels = df['Query_Level'].dropna().unique()
    if len(levels) == 0:
        return True
    return False


def _weighted_return(group, return_col, weight_col='Denom'):
    """Compute Denom-weighted return for a group."""
    total_denom = group[weight_col].sum()
    if total_denom == 0:
        return 0.0
    return (group[return_col] * group[weight_col]).sum() / total_denom


def _aggregate_group(group, property_type='All', cbsa_name='All',
                     query_level='Total', held_sold='All'):
    """Aggregate a group of property rows into a single summary row."""
    year = group['Year'].iloc[0]
    quarter = group['Quarter'].iloc[0]

    row = {
        'Query_Level': query_level,
        'Year': int(year),
        'YYYYQ': int(year * 10 + quarter),
        'Quarter': int(quarter),
        'PropertyType': property_type,
        'CBSAName': cbsa_name,
        'HeldSold': held_sold,
        'Prop_Count': len(group),
    }

    # Sum dollar columns
    for col in DOLLAR_COLS:
        row[col] = group[col].sum()

    # Weighted average returns
    for col in RETURN_COLS:
        row[col] = _weighted_return(group, col)

    return row


def aggregate_properties(df):
    """
    Take a property-level DataFrame and produce the full aggregated DataFrame
    with all Query_Level types.

    Returns a DataFrame in the same format as the NCREIF benchmark data,
    plus optional By_HeldSold rows if a Sold column is present.
    """
    # Clean up input
    df = df.copy()
    for col in DOLLAR_COLS + RETURN_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df['Year'] = df['Year'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)

    # Net Sale Price: when present and non-zero, use as ending MV for that quarter
    if 'Net Sale Price' in df.columns:
        df['Net Sale Price'] = pd.to_numeric(df['Net Sale Price'], errors='coerce').fillna(0.0)
        has_nsp = df['Net Sale Price'] != 0
        df.loc[has_nsp, 'MV'] = df.loc[has_nsp, 'Net Sale Price']

    # Determine Held/Sold classification if Sold column exists
    has_sold = 'Sold' in df.columns
    if has_sold:
        df['Sold'] = pd.to_numeric(df['Sold'], errors='coerce').fillna(0).astype(int)
        # If ANY quarter for a property has Sold=1, classify ALL quarters as Sold
        if 'PropertyID' in df.columns:
            sold_props = set(df.loc[df['Sold'] == 1, 'PropertyID'].unique())
            df['_HeldSold'] = df['PropertyID'].apply(
                lambda pid: 'Sold' if pid in sold_props else 'Held'
            )
        else:
            df['_HeldSold'] = df['Sold'].apply(lambda x: 'Sold' if x == 1 else 'Held')

    rows = []
    periods = df.groupby(['Year', 'Quarter'])

    for (year, quarter), period_df in periods:
        # Total
        rows.append(_aggregate_group(
            period_df, 'All', 'All', 'Total'
        ))

        # By PropertyType
        for pt, pt_group in period_df.groupby('PropertyType'):
            rows.append(_aggregate_group(
                pt_group, pt, 'All', 'By_PropertyType'
            ))

        # By CBSA
        for cbsa, cbsa_group in period_df.groupby('CBSAName'):
            rows.append(_aggregate_group(
                cbsa_group, 'All', cbsa, 'By_CBSA'
            ))

        # By PropertyType x CBSA
        for (pt, cbsa), cross_group in period_df.groupby(['PropertyType', 'CBSAName']):
            rows.append(_aggregate_group(
                cross_group, pt, cbsa, 'By_PropertyType_CBSA'
            ))

        # By HeldSold (only if Sold column exists)
        if has_sold:
            for hs, hs_group in period_df.groupby('_HeldSold'):
                rows.append(_aggregate_group(
                    hs_group, 'All', 'All', 'By_HeldSold', held_sold=hs
                ))

    result = pd.DataFrame(rows)

    # Ensure column order matches benchmark format
    col_order = [
        'Query_Level', 'Year', 'YYYYQ', 'Quarter', 'PropertyType', 'CBSAName',
        'HeldSold',
        'NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
        'Income_Return', 'Capital_Return', 'Total_Return', 'Prop_Count'
    ]
    result = result[col_order]
    return result
