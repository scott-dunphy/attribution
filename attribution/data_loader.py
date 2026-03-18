import pandas as pd
from .aggregator import is_property_level, validate_property_file, aggregate_properties

REQUIRED_COLUMNS = [
    'Query_Level', 'Year', 'YYYYQ', 'Quarter', 'PropertyType', 'CBSAName',
    'NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
    'Income_Return', 'Capital_Return', 'Total_Return', 'Prop_Count'
]

VALID_QUERY_LEVELS = {'By_CBSA', 'By_PropertyType', 'By_PropertyType_CBSA', 'By_HeldSold', 'Total'}


def load_file(filepath):
    """Load an attribution file. Auto-detects property-level vs aggregated format."""
    df = pd.read_excel(filepath)

    if is_property_level(df):
        return load_property_file(df)
    else:
        return load_aggregated_file(df)


def load_property_file(df):
    """Load a property-level file and aggregate it."""
    errors = validate_property_file(df)
    if errors:
        raise ValueError('; '.join(errors))

    # Cast types
    for col in ['NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
                'Income_Return', 'Capital_Return', 'Total_Return']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    aggregated = aggregate_properties(df)
    return aggregated


def load_aggregated_file(df):
    """Load an already-aggregated file (benchmark format)."""
    errors = validate_schema(df)
    if errors:
        raise ValueError('; '.join(errors))
    df['Year'] = df['Year'].astype(int)
    df['Quarter'] = df['Quarter'].astype(int)
    df['YYYYQ'] = df['YYYYQ'].astype(int)
    for col in ['NOI', 'CapEx', 'MV', 'MVLag1', 'PSales', 'Denom',
                'Income_Return', 'Capital_Return', 'Total_Return']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df['Prop_Count'] = pd.to_numeric(df['Prop_Count'], errors='coerce').fillna(0).astype(int)
    if 'HeldSold' not in df.columns:
        df['HeldSold'] = 'All'
    return df


def validate_schema(df):
    """Return list of validation errors, empty if valid."""
    errors = []
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")
        return errors
    levels = set(df['Query_Level'].dropna().unique())
    unexpected = levels - VALID_QUERY_LEVELS
    if unexpected:
        errors.append(f"Unexpected Query_Level values: {', '.join(sorted(unexpected))}")
    if 'Total' not in levels:
        errors.append("No 'Total' rows found in data")
    return errors


def get_available_periods(df):
    """Return sorted list of (Year, Quarter) tuples from Total rows."""
    total = df[df['Query_Level'] == 'Total']
    periods = total[['Year', 'Quarter']].drop_duplicates()
    periods = periods.sort_values(['Year', 'Quarter'])
    return list(periods.itertuples(index=False, name=None))


def get_common_periods(portfolio_df, benchmark_df):
    """Return periods present in both files."""
    p_periods = set(get_available_periods(portfolio_df))
    b_periods = set(get_available_periods(benchmark_df))
    common = sorted(p_periods & b_periods)
    return common
